import asyncio
import sys
import json
from typing import Optional

from agents import gen_trace_id, trace
from agents.mcp import MCPServerStdio
from agent.models import TracerConfig
from agent.mcp_client import MCPClient
from agent.tracer import CryptoTracer
from agent.reporting import build_report

async def run_trace(
    description: str,
    victim_address: str,
    blockchain: str = "eth",
    asset: Optional[str] = None,
    date: Optional[str] = None,
    tx_hashes: Optional[str] = None
):
    print(f"Starting trace for {victim_address} on {blockchain}...")

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

            config = TracerConfig(
                description=description,
                victim_address=victim_address,
                blockchain_name=blockchain,
                asset_symbol=asset,
                approx_date=date,
                known_tx_hashes=tx_hashes.split(",") if tx_hashes else []
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

        try:
            description = input("Enter case description: ").strip()
            while not description:
                print("Description is required.")
                description = input("Enter case description: ").strip()

            victim = input("Enter victim address: ").strip()
            while not victim:
                print("Victim address is required.")
                victim = input("Enter victim address: ").strip()

            chain = input("Enter blockchain (default: eth): ").strip() or "eth"
            asset = input("Enter asset symbol (default: USDT): ").strip() or "USDT"
            date = input("Enter approximate date (YYYY-MM-DD, optional): ").strip() or None
            tx_hashes = input("Enter known transaction hashes (comma-separated, optional): ").strip() or None

            print("\nInitializing trace with provided parameters...")
            asyncio.run(run_trace(description, victim, chain, asset, date, tx_hashes))

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)
    else:
        # Command line arguments mode
        description = sys.argv[1]
        victim = sys.argv[2]
        chain = sys.argv[3] if len(sys.argv) > 3 else "eth"
        asset = sys.argv[4] if len(sys.argv) > 4 else None
        date = sys.argv[5] if len(sys.argv) > 5 else None
        tx_hashes = sys.argv[6] if len(sys.argv) > 6 else None

        asyncio.run(run_trace(description, victim, chain, asset, date, tx_hashes))

if __name__ == "__main__":
    main()
