import asyncio

from pydantic import BaseModel

from agents import Agent, Runner, trace, gen_trace_id
from agents.run_context import RunContextWrapper
from agents.mcp import MCPServer, MCPServerStdio



"""
Simple tracer agent for tracing simple cases.
"""


async def run(mcp_server: MCPServer, case_description: str, seed_wallets: str):
    chain_trace_core_agent = Agent(
        name="chain_trace_core_agent",
        instructions="""You are ChainTrace-Core
    Goal: receive {case_description} and {seed_wallets}.  
    Return JSON request list for MCP that pulls all outgoing txs up to 10 hops, any bridge / swap / mixer / CEX / high-risk entity on path, with USD values.  
    Ignore irrelevant tokens.  
    Depth-first, newest-first. 
    Use the tools for getting the data.
        """,
        mcp_servers=[mcp_server],
    )

    trace_fetch_agent = Agent(
        name="trace_fetch_agent",
        instructions="""You are ChainTrace-Fetcher.  
Input: {mcp_responses}.  
Merge, dedupe, keep fields: tx_hash, from, to, token, amount, fiat, timestamp, entity_type, risk_score.  
Stop expanding once entity_type ∈ {bridge, mixer, cex, p2p_exchange, blocked}.  
Return JSON array “edges” (objects: {src, dst, amount_usd, risk, entity_type, ts}).""",
        # output_type=OutlineCheckerOutput,
        mcp_servers=[mcp_server],
    )

    trace_classifier_agent = Agent(
        name="trace_classifier_agent",
        instructions="""
        You are “RiskTagger”.  
Input: {edges}.  
Add label to each unique address:  
• “Victim”, “Scammer”, “Bridge”, “Mixer”, “CEX”, “P2P”, “Unknown”.  
Rule-of-thumb: first hop = “Scammer”; any owner info ⇒ use.  
Return two arrays: “nodes” (id, label, risk) and “edges”."""
        # output_type=str,
    )

    graph_builder_agent = Agent(
        name="graph_builder_agent",
        instructions="""
        You are “GraphMaker”.  
Input: {nodes, edges}.  
Produce Mermaid-flowchart code with vertical layout (top = victim).  
Colour:  
    Low risk ≤0.3 - green,  
    0.3<risk≤0.6 - yellow,  
    >0.6 - red.  
Edge label = amount USD.  
Return only the Mermaid code block.""",
    )

    # Ensure the entire workflow is a single trace
    with trace("Simple case tracer"):
        # 1. 
        chain_trace_core_agent_result = await Runner.run(
            chain_trace_core_agent,
            input=f"{case_description} {seed_wallets}",
        )
        print("Outline generated")

        # 2. 
        trace_fetch_agent_result = await Runner.run(
            trace_fetch_agent,
            input=chain_trace_core_agent_result.final_output,
        )

        # 3. 
        trace_classifier_result = await Runner.run(
            trace_classifier_agent,
            input=trace_fetch_agent_result.final_output,
        )

        print("Trace is fetched and classified.")

        # 4. Generate the graph
        graph_builder_agent_result = await Runner.run(
            graph_builder_agent,
            input=trace_classifier_result.final_output,
        )   

        print(f"Graph: {graph_builder_agent_result.final_output}")

async def main():    

    case_description = input("Give the simple case description:")
    seed_wallets = input("Give the victim wallet address:")

    
    async with MCPServerStdio(
        name="Filesystem Server, via npx",
        params={
            "command": "docker",
            "args": ["run","-i","--rm","mcp-server-amlbot"]
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Filesystem Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server, case_description, seed_wallets)


if __name__ == "__main__":
    asyncio.run(main())