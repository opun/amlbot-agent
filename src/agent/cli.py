import asyncio
import sys
import json
import termios
from typing import Optional

from agents import gen_trace_id, trace
from agents.mcp import MCPServerStdio
from agent.models import TracerConfig
from agent.mcp_client import MCPClient
from agent.tracer import CryptoTracer
from agent.reporting import build_report
from agent.theft_detection import parse_case_description_with_llm

def flush_input():
    """Flush standard input to avoid skipping prompts due to pasted multi-line text."""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

async def run_trace(
    description: Optional[str] = None,
    victim_address: Optional[str] = None,
    blockchain: str = "eth",
    asset: Optional[str] = None,
    date: Optional[str] = None,
    tx_hashes: Optional[str] = None,
    tx_hash: Optional[str] = None,
    theft_asset: Optional[str] = None
):
    if victim_address and tx_hash:
        print(f"Starting trace for transaction {tx_hash} (victim: {victim_address}) on {blockchain}...")
    elif victim_address:
        print(f"Starting trace for {victim_address} on {blockchain}...")
    elif tx_hash:
        print(f"Starting trace for transaction {tx_hash} on {blockchain}...")
    else:
        print(f"Starting trace on {blockchain}...")

    # Setup MCP Server
    async with MCPServerStdio(
        name="AMLBot MCP Server",
        params={
            "command": "docker",
            "args": ["run", "-i", "--rm", "-e", "USER_ID=a2fa961b7f4977981e3796916328d930", "mcp-server-amlbot:local"]
            # "args": ["run", "-i", "-e", "USER_ID=a2fa961b7f4977981e3796916328d930", "mcp-server-amlbot:local"]
        },
        client_session_timeout_seconds=300.0,
    ) as server:

        # Create trace context for OpenAI Tracing
        # Note: The tracer.py also generates a trace_id for the CaseMeta, but this one is for the outer span.
        trace_id = gen_trace_id()
        print(f"\nView trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")

        with trace(workflow_name="Crypto Tracer Agent", trace_id=trace_id):
            client = MCPClient(server)
            tracer = CryptoTracer(client)

            # Parse case description with LLM if provided
            parsed_info = {}
            if description:
                print("Parsing case description with AI...")
                parsed_info = await parse_case_description_with_llm(description)
                print(f"Extracted information: {json.dumps({k: v for k, v in parsed_info.items() if v is not None}, indent=2)}")

            # Merge parsed info with explicit parameters (explicit parameters take precedence)
            final_description = description or parsed_info.get("description")
            final_victim_address = victim_address or parsed_info.get("victim_address")
            final_blockchain = blockchain if blockchain != "eth" or "blockchain_name" not in parsed_info else parsed_info.get("blockchain_name", "eth")
            final_asset = asset or parsed_info.get("asset_symbol")
            final_date = date or parsed_info.get("approx_date")
            final_tx_hash = tx_hash or parsed_info.get("tx_hash")
            final_theft_asset = theft_asset or parsed_info.get("theft_asset")

            # Handle known_tx_hashes
            known_hashes = []
            if tx_hashes:
                known_hashes.extend(tx_hashes.split(","))
            if parsed_info.get("known_tx_hashes"):
                known_hashes.extend(parsed_info["known_tx_hashes"])
            # If tx_hash is provided, add it to known_tx_hashes if not already there
            if final_tx_hash and final_tx_hash not in known_hashes:
                known_hashes.append(final_tx_hash)

            config = TracerConfig(
                description=final_description,
                victim_address=final_victim_address,
                blockchain_name=final_blockchain,
                asset_symbol=final_asset,
                approx_date=final_date,
                known_tx_hashes=known_hashes,
                tx_hash=final_tx_hash,
                theft_asset=final_theft_asset
            )

            print("Running tracer...")
            result = await tracer.trace(config)

            print("Building report...")
            report = build_report(result)

            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(report["summary_text"])

            print("\n" + "="*50)
            print("JSON GRAPH (Saved to result.json)")
            print("="*50)

            with open("result.json", "w") as f:
                json.dump(report, f, indent=2)

            print("Done.")

def main():
    # Check if running in interactive mode (no args provided)
    if len(sys.argv) < 3:
        print("Crypto Tracer Agent - Interactive Mode")
        print("=" * 40)
        print("\nYou can provide a case description and the system will automatically extract")
        print("all required information using AI. Alternatively, you can manually specify parameters.")
        print("\nTwo input modes are supported:")
        print("Mode 1: Case Description (optional), Wallet Address, Blockchain, Date")
        print("Mode 2: Case Description (optional), Transaction Hash, Blockchain, Theft Asset")
        print()

        try:
            # First, get case description - this will be parsed by LLM
            description = input("Enter case description (optional, will be parsed by AI): ").strip() or None

            # If description provided, try to extract info from it
            parsed_info = {}
            if description:
                print("\nParsing case description with AI...")
                try:
                    parsed_info = asyncio.run(parse_case_description_with_llm(description))
                    if parsed_info:
                        print(f"\nExtracted information:")
                        for key, value in parsed_info.items():
                            if value is not None:
                                print(f"  {key}: {value}")
                except Exception as e:
                    print(f"Warning: Could not parse description with AI: {e}")
                    print("You can still provide parameters manually.")

            # Allow user to modify extracted information
            if parsed_info:
                flush_input()
                if input("\nDo you want to modify the extracted information? (y/n, default: n): ").strip().lower() == "y":
                    print("\nPress Enter to keep current value, or type new value.")

                    # Helper to get input with default
                    def get_input(label, key):
                        current = parsed_info.get(key)
                        val = input(f"{label} [{current if current is not None else ''}]: ").strip()
                        if val:
                            parsed_info[key] = val

                    get_input("Victim Address", "victim_address")
                    get_input("Transaction Hash", "tx_hash")
                    get_input("Blockchain", "blockchain_name")
                    get_input("Asset Symbol", "asset_symbol")
                    get_input("Theft Asset", "theft_asset")
                    get_input("Approximate Date (YYYY-MM-DD)", "approx_date")
                    get_input("Trace Mode (address/transaction)", "trace_mode")

                    # Handle known_tx_hashes (list)
                    current_hashes = ",".join(parsed_info.get("known_tx_hashes", []))
                    hashes_val = input(f"Known Tx Hashes (comma-separated) [{current_hashes}]: ").strip()
                    if hashes_val:
                        parsed_info["known_tx_hashes"] = [h.strip() for h in hashes_val.split(",") if h.strip()]

            # Mode selection
            print("\nSelect Input Mode:")
            print("1. Address Mode (Trace starting from victim address)")
            print("2. Transaction Mode (Trace starting from theft transaction hash)")

            # Determine default mode from parsed_info
            detected_mode = parsed_info.get("trace_mode")
            if detected_mode in ["transaction", "2"]:
                default_mode = "2"
            elif detected_mode in ["address", "1"]:
                default_mode = "1"
            elif parsed_info.get("tx_hash") and not parsed_info.get("victim_address"):
                default_mode = "2"
            else:
                default_mode = "1"

            flush_input()
            mode = input(f"Enter mode (1/2, default: {default_mode}): ").strip() or default_mode

            if mode == "2":
                # Mode 2: tx_hash mode
                tx_hash = parsed_info.get("tx_hash") or input("Enter transaction hash: ").strip()
                while not tx_hash:
                    print("Transaction hash is required for Mode 2.")
                    tx_hash = input("Enter transaction hash: ").strip()

                chain = parsed_info.get("blockchain_name") or input("Enter blockchain (e.g., eth, trx, btc): ").strip()
                while not chain:
                    print("Blockchain is required.")
                    chain = input("Enter blockchain: ").strip()

                theft_asset = parsed_info.get("theft_asset") or input("Enter theft asset symbol (e.g., USDT, ETH): ").strip()
                while not theft_asset:
                    print("Theft asset is required for Mode 2.")
                    theft_asset = input("Enter theft asset symbol: ").strip()

                victim = parsed_info.get("victim_address")
                asset = parsed_info.get("asset_symbol")
                date = parsed_info.get("approx_date")
                tx_hashes = ",".join(parsed_info.get("known_tx_hashes", [])) if parsed_info.get("known_tx_hashes") else None
            else:
                # Mode 1: address mode
                victim = parsed_info.get("victim_address") or input("Enter victim address: ").strip()
                while not victim:
                    print("Victim address is required for Mode 1.")
                    victim = input("Enter victim address: ").strip()

                chain = parsed_info.get("blockchain_name") or input("Enter blockchain (default: eth): ").strip() or "eth"
                asset = parsed_info.get("asset_symbol") or input("Enter asset symbol (optional): ").strip() or None
                date = parsed_info.get("approx_date") or input("Enter approximate date (YYYY-MM-DD, optional): ").strip() or None
                tx_hashes_input = input("Enter known transaction hashes (comma-separated, optional): ").strip()
                if tx_hashes_input:
                    tx_hashes = tx_hashes_input
                elif parsed_info.get("known_tx_hashes"):
                    tx_hashes = ",".join(parsed_info["known_tx_hashes"])
                else:
                    tx_hashes = None
                tx_hash = parsed_info.get("tx_hash")
                theft_asset = parsed_info.get("theft_asset")

            # Validate that either victim_address or tx_hash is provided
            if not victim and not tx_hash:
                print("Error: Either victim address (Mode 1) or transaction hash (Mode 2) must be provided.")
                sys.exit(1)

            # Note: It is okay to provide both. Tracer will use victim_address and prioritize tx_hash.

            print("\nInitializing trace with provided parameters...")
            asyncio.run(run_trace(
                description=description,
                victim_address=victim,
                blockchain=chain,
                asset=asset,
                date=date,
                tx_hashes=tx_hashes,
                tx_hash=tx_hash,
                theft_asset=theft_asset
            ))

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)
    else:
        # Command line arguments mode
        # Format: description victim_address blockchain [asset] [date] [tx_hashes] [--tx_hash HASH] [--theft_asset ASSET] [--threshold PERCENT]
        # Or: description --tx_hash HASH blockchain theft_asset [--threshold PERCENT]
        description = sys.argv[1] if sys.argv[1] != "--tx_hash" else None

        # Check if Mode 2 (tx_hash mode)
        if "--tx_hash" in sys.argv:
            tx_hash_idx = sys.argv.index("--tx_hash")
            tx_hash = sys.argv[tx_hash_idx + 1] if tx_hash_idx + 1 < len(sys.argv) else None
            blockchain = sys.argv[2] if sys.argv[1] != "--tx_hash" else sys.argv[3]

            theft_asset_idx = sys.argv.index("--theft_asset") if "--theft_asset" in sys.argv else -1
            theft_asset = sys.argv[theft_asset_idx + 1] if theft_asset_idx >= 0 and theft_asset_idx + 1 < len(sys.argv) else None

            victim = None
            asset = None
            date = None
            tx_hashes = None
        else:
            # Mode 1: address mode
            victim = sys.argv[2]
            blockchain = sys.argv[3] if len(sys.argv) > 3 else "eth"
            asset = sys.argv[4] if len(sys.argv) > 4 else None
            date = sys.argv[5] if len(sys.argv) > 5 else None
            tx_hashes = sys.argv[6] if len(sys.argv) > 6 else None
            tx_hash = None
            theft_asset = None

        # Validate that either victim_address or tx_hash is provided
        if not victim and not tx_hash:
            print("Error: Either victim address (Mode 1) or transaction hash (Mode 2) must be provided.")
            sys.exit(1)

        asyncio.run(run_trace(
            description=description,
            victim_address=victim,
            blockchain=blockchain,
            asset=asset,
            date=date,
            tx_hashes=tx_hashes,
            tx_hash=tx_hash,
            theft_asset=theft_asset
        ))

if __name__ == "__main__":
    main()
