"""
Main entry point for the multi-agent crypto tracing system.
"""

import asyncio
import json
import sys
from pathlib import Path
from agents.mcp import MCPServerStdio
from agents import gen_trace_id, trace
from agent.multi_agent_system import run_multi_agent_trace


async def main():
    """
    Main entry point that sets up the MCP server and runs the multi-agent trace.
    """
    # Get inputs from user or command line
    if len(sys.argv) > 1:
        # Command line mode
        case_description = sys.argv[1]
        victim_address = sys.argv[2] if len(sys.argv) > 2 else None
        tx_hashes = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else None
        visualization_file = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        # Interactive mode
        print("Multi-Agent Crypto Tracing System")
        print("=" * 50)
        case_description = input("Enter case description: ").strip()
        victim_address = input("Enter victim address (optional, press Enter to skip): ").strip() or None
        tx_hashes_input = input("Enter transaction hashes (comma-separated, optional): ").strip()
        tx_hashes = tx_hashes_input.split(',') if tx_hashes_input else None
        visualization_file = input("Enter path to visualization JSON file (optional): ").strip() or None

    # Load visualization JSON if provided
    visualization_json = None
    if visualization_file:
        try:
            with open(visualization_file, 'r') as f:
                visualization_json = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load visualization JSON: {e}", file=sys.stderr)

    # Set up MCP server
    async with MCPServerStdio(
        name="AMLBot MCP Server",
        params={
            "command": "docker",
            # "args": ["run", "-i", "--rm", "-e", "USER_ID=a2fa961b7f4977981e3796916328d930", "mcp-server-amlbot:local"]
            "args": ["run", "-i", "-e", "USER_ID=a2fa961b7f4977981e3796916328d930", "mcp-server-amlbot:local"]
        },
        client_session_timeout_seconds=60.0,
    ) as server:
        trace_id = gen_trace_id()
        print(f"\nView trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")

        # Run the multi-agent trace
        try:
            result = await run_multi_agent_trace(
                mcp_server=server,
                case_description=case_description,
                victim_address=victim_address,
                tx_hashes=tx_hashes,
            )

            # Output results
            print("\n" + "=" * 50)
            print("TRACE RESULTS")
            print("=" * 50)
            print("\nSummary:")
            print(result.get("summary_text", "No summary available"))
            print("\n" + "=" * 50)
            print("Graph JSON:")
            print(json.dumps(result.get("graph", {}), indent=2))

            # Optionally save to file
            output_file = input("\nSave results to file? (Enter filename or press Enter to skip): ").strip()
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {output_file}")

        except Exception as e:
            print(f"Error during trace: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
