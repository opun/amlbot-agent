"""
Multi-Agent Crypto Tracing System

Implements four specialized agents for crypto tracing investigations:
1. Router/Case Orchestrator
2. Tracing Agent
3. Classification/Bridge Agent
4. Reporting/Diagram Agent
"""

import json
from typing import Any, Dict, Optional, List
from agents import Agent, Runner, trace
from agents.mcp import MCPServer


# ============================================================================
# Router / Case Orchestrator Function
# ============================================================================

def build_case_context(
    case_description: str,
    victim_address: Optional[str] = None,
    tx_hashes: Optional[List[str]] = None,
    chains: Optional[List[str]] = None,
    asset_symbol: Optional[str] = None,
    approx_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a case_context object from user input.
    This function can be enhanced with an LLM to parse case descriptions.
    """
    import re
    from datetime import datetime

    # Extract case_id from description or generate one
    case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Try to extract date from description
    if not approx_date:
        # Simple date extraction (can be enhanced)
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        match = re.search(date_pattern, case_description)
        if match:
            approx_date = match.group()
        else:
            approx_date = datetime.now().strftime('%Y-%m-%d')

    # Try to extract asset from description
    if not asset_symbol:
        asset_pattern = r'\b(USDT|ETH|BTC|TRX|SOL|MATIC)\b'
        match = re.search(asset_pattern, case_description, re.IGNORECASE)
        if match:
            asset_symbol = match.group().upper()
        else:
            asset_symbol = "USDT"  # Default

    # Try to extract chains from description
    if not chains:
        chain_keywords = {
            'ethereum': 'eth',
            'eth': 'eth',
            'tron': 'trx',
            'trx': 'trx',
            'bitcoin': 'btc',
            'btc': 'btc',
            'solana': 'sol',
            'sol': 'sol',
            'polygon': 'matic',
            'matic': 'matic',
            'zksync': 'zksync',
        }
        chains = []
        desc_lower = case_description.lower()
        for keyword, chain in chain_keywords.items():
            if keyword in desc_lower and chain not in chains:
                chains.append(chain)
        if not chains:
            chains = ['eth']  # Default

    # Determine case type from description
    desc_lower = case_description.lower()
    if any(word in desc_lower for word in ['job', 'eaccountable', 'investment platform', 'salary', 'commission']):
        experiment_case_type = "job_investment_platform"
    elif any(word in desc_lower for word in ['whatsapp', 'instagram', 'social media', 'romance', 'bitfinexpro']):
        experiment_case_type = "social_media_investment"
    elif any(word in desc_lower for word in ['rental', 'rent', 'whatsapp rental']):
        experiment_case_type = "bridge_easy_rental"
    elif any(word in desc_lower for word in ['wallet compromise', 'phishing', 'malware']):
        experiment_case_type = "bridge_mid_wallet_compromise"
    elif any(word in desc_lower for word in ['device', 'remote control', 'infected']):
        experiment_case_type = "bridge_mid_device_compromise"
    elif any(word in desc_lower for word in ['otc', 'cash', 'broker']):
        experiment_case_type = "otc_cash_usdt"
    elif any(word in desc_lower for word in ['podcast', 'screen share', 'metamask', 'remote screen']):
        experiment_case_type = "podcast_remote_metamask"
    else:
        experiment_case_type = "other"

    narrative_summary = case_description[:200] + "..." if len(case_description) > 200 else case_description

    return {
        "case_id": case_id,
        "approx_date": approx_date,
        "chains": chains,
        "asset_symbol": asset_symbol,
        "victim_address": victim_address,
        "narrative_summary": narrative_summary,
        "experiment_case_type": experiment_case_type,
    }


# ============================================================================
# Tracing Agent
# ============================================================================

def create_tracing_agent(mcp_server: MCPServer) -> Agent:
    """
    Creates the Tracing Agent that reconstructs on-chain paths.
    """
    instructions = """You are a crypto tracing analyst.

Goal

Starting from the victim's address and approximate theft date, reconstruct the main path(s) of stolen funds until they reach:
• a bridge / cross-chain service,
• a centralized service (exchange, OTC, custodial wallet, swap aggregator),
• or a clear terminal destination (no further meaningful outgoing activity).

Inputs

You receive:
• case_context:
  - case_id
  - approx_date
  - chains
  - asset_symbol
  - victim_address (if known)
  - narrative_summary
  - experiment_case_type
• Possibly additional starting hints: addresses, tx hashes, CEX deposit addresses.

Available tools

• expert-search(hash, filter) – locate addresses/txs/entities.
• get-address(blockchain_name, address) – risk and entity profile for an address.
• all-txs(blockchain_name, address, filter, limit, offset) – list txs for an address.
• get-transaction(blockchain_name, address, tx_hash, token_id, path) – detailed tx info.
• get-position(blockchain_name, address, tx_hash, token_id, path) – follow the same funds forward from a tx position.
• token-stats(blockchain_name, address) – token distribution and volumes.
• get-extra-address-info(address, asset) – extra service tags and riskScore.

How to trace

1. Determine the source chain and asset:
• Usually from case_context.chains[0] and asset_symbol.
• Example: victim had USDT on Ethereum ("eth", "USDT").

2. Identify theft / drain txs:
• If you know victim_address:
  - Use all-txs around approx_date to find outgoing txs:
    * Matching asset_symbol,
    * Significant amounts (consistent with described loss),
    * At or shortly before/after the theft time.
  - For each candidate, call get-transaction to confirm:
    * from and to,
    * amount,
    * timestamp.
• If you only have hashes or CEX deposit addresses:
  - Use expert-search and get-transaction to locate the victim side.

3. For each confirmed theft tx:
• Create a path starting from the victim.
• Step 0: victim → first-hop address.
• Use get-address and get-extra-address-info on the first-hop:
  - If it already looks like CEX / service / bridge, record it and stop there for now (classification will refine).
  - Otherwise, treat it as a candidate intermediate / perpetrator wallet.

4. Walk forward (hop-by-hop):
• From each intermediate address, use get-position to find the next outgoing tx(s) that move a similar amount of the same asset.
• For each such tx:
  - Use get-transaction to confirm to/from, amount, time.
  - Add a step to the path.
  - Use get-address / get-extra-address-info on the new receiving address.
  - Continue until:
    * The funds reach a clear service (CEX / OTC / bridge contract).
    * The funds split into many small transfers (noise).
    * There are no more significant outgoing txs.

5. Focus:
• It is better to return 1–3 high-confidence paths than 20 noisy branches.
• Include the most likely primary path for the bulk of the victim's funds.

Do NOT:
• Call the bridge-analyze tool. Bridge classification is done by another agent.
• Rely on visualization JSONs. You work purely from tools and case context.

Output format

Return JSON only, with this structure:

{
  "case_meta": {
    "case_id": "<string>",
    "chains": ["eth", "trx"],
    "asset_symbol": "USDT",
    "approx_date": "2025-09-25",
    "victim_address": "<0x... or base58...>"
  },
  "paths": [
    {
      "path_id": "path-1",
      "description": "Main path of stolen funds from victim to first service or bridge",
      "steps": [
        {
          "step_index": 0,
          "from": "<victim-address>",
          "to": "<first-hop-address>",
          "tx_hash": "<hash>",
          "chain": "eth",
          "asset": "USDT",
          "amount_estimate": 66030.0,
          "time": "2025-09-25T12:34:56Z",
          "direction": "out"
        }
      ]
    }
  ],
  "entities": [
    {
      "address": "<victim-address>",
      "chain": "eth",
      "role": "victim",
      "risk_score": 0.0,
      "labels": ["Victim's wallet"],
      "notes": "Address controlled by victim per case context."
    }
  ],
  "annotations": []
}

• Use real values for time and amounts when available.
• If uncertain about an address role, set role: "intermediate" and explain in notes.
• Do not include prose outside the JSON."""

    return Agent(
        name="tracing_agent",
        instructions=instructions,
        mcp_servers=[mcp_server],
    )


# ============================================================================
# Classification / Bridge Agent
# ============================================================================

def create_classification_agent(mcp_server: MCPServer) -> Agent:
    """
    Creates the Classification/Bridge Agent that detects bridges and classifies services.
    """
    instructions = """You are a specialist in classifying entities and detecting cross-chain bridge operations within traced crypto paths.

Inputs

• The tracing result JSON:
  - case_meta
  - paths (with steps)
  - entities (addresses with initial roles and notes)
  - annotations (optional)
• Optionally, a visualization_json produced by the existing tracing system:
  - payload.items, payload.txs, payload.connects, payload.comments
  - helpers.isConnectionBasedMode, helpers.txList

Goals

1. Detect and label bridge operations (source chain → destination chain).
2. Classify services:
• Bridges (protocol-level labels, e.g. LayerZero, AllBridge, Bridgers).
• CEX deposits (Binance, HTX, Bybit, Crypto.com, etc.).
• OTC / swap / other service addresses.
• Unidentified services / mixers.
3. Identify perpetrator addresses where possible.
4. Enrich paths, entities, and annotations with:
• step types,
• service labels,
• bridge protocol info,
• comments aligned with visualization style.

Tools

• get-address(blockchain_name, address) – AML risk signals & entity classification.
• token-stats(blockchain_name, address) – token volumes per address.
• get-extra-address-info(address, asset) – extra tags and services.use_platform.
• all-txs, get-transaction – for additional context if needed.
• bridge-analyze(chain, tx_hash) – detect and describe bridge operations (LayerZero / AllBridge).

Bridge detection strategy

1. Identify candidate bridge steps among paths[*].steps[*]:
• Steps where:
  - The destination address is a known bridge contract or swap service (via get-extra-address-info.services.use_platform).
  - There is a strong cross-chain pattern from the case (e.g., Ethereum → Tron, Tron → Ethereum, chain → zkSync, etc.).

2. For each candidate step, call:
• bridge-analyze(chain = step.chain, tx_hash = step.tx_hash).

3. If bridge-analyze returns:
• result.status == "ok" and result.is_bridge == true:
  - Mark the source step as:
    * step_type = "bridge_in"
    * protocol = result.protocol (e.g. "layerzero", "allbridge").
  - Add / update an entity for result.destination_address on result.dst_chain:
    * role = "bridge_service" or "bridge_receiver" depending on context.
  - On the destination chain:
    * Locate the first incoming tx to destination_address with a matching amount/time (approximate match).
    * Mark that destination step as step_type = "bridge_out".

4. If bridge-analyze is not confident or is_bridge == false:
• Apply heuristics:
  - Look for near-equal amounts of the same token moving onto another chain shortly after.
  - Use get-extra-address-info to detect known bridge / swap platforms (e.g. "Bridgers.xyz Swap").
  - If confident enough, still mark:
    * step_type = "bridge_in" and step_type = "bridge_out" with a protocol or service_label based on best evidence.

Service classification

For each relevant address (from entities and path steps):

• Use get-address and get-extra-address-info:
  - If AML signals indicate a licensed or unlicensed exchange:
    * role = "cex_deposit".
    * Add labels like "Binance deposit address", "HTX deposit address", "Bybit deposit address", "Crypto.com deposit address", etc.
  - If services.use_platform indicates a specific bridge or swap (e.g. "Bridgers.xyz Swap"):
    * role = "bridge_service" (or "swap_service" if more appropriate).
    * Add labels like "Bridgers / Bridge to Tron".
  - If OTC / P2P patterns are clear:
    * role = "otc_service".
  - If riskScore is high and tags indicate obfuscation (mixer/multi-service):
    * role = "unidentified_service" or "mixer".
  - Otherwise, if not clearly a service:
    * role may remain "intermediate".

Use token-stats to confirm that high volumes of a given token pass through a suspected service address.

Perpetrator identification

• In simple cases (e.g., rental "bridge easy"):
  - The address that first receives from the victim and then forwards to bridge / CEX is often the perpetrator.
  - Mark such addresses:
    * role = "perpetrator",
    * add a label "Perpetrator's address" and notes such as:
      "Collected victim's funds and forwarded them to bridge and CEX."
• In more complex cases, you may have several candidate perpetrator addresses:
  - Choose those that appear as central intermediate aggregators and label accordingly.

Using visualization JSON (if provided)

• If visualization_json.helpers.isConnectionBasedMode === true:
  - Ignore the visualization completely.
• Otherwise:
  - Understand the mapping:
    * payload.items → addresses/entities.
    * payload.txs → transaction nodes.
    * payload.connects → edges (ignore connections of type "mergedEdge").
    * payload.comments → comment nodes with labels and text.
    * helpers.txList → full tx details for visualization.
  - You may:
    * Use comment labels as inspiration for your own labels and annotations.
    * e.g. "Bridge to Tron", "Bridge to zkSync", "FixedFloat deposit address", etc.
    * Use helpers.txList to double-check time/amount when it helps confirm your classification.
  - Do not override your reasoning solely to match visualization – they are reference, not ground truth.

Enrichment rules

1. Steps (paths[*].steps[*])
• Add:
  - step_type:
    * "direct_transfer"
    * "bridge_in"
    * "bridge_out"
    * "service_deposit"
    * "internal_transfer"
  - service_label (optional):
    * e.g. "Bridgers / Bridge to Tron", "Binance CEX deposit", "HTX deposit address", "FixedFloat deposit address", "Shuffle deposit address", "Stake deposit address", "Suspected unidentified service".
  - protocol (optional):
    * e.g. "layerzero", "allbridge".

2. Entities (entities[*])
• Refine role as one of:
  - "victim"
  - "perpetrator"
  - "bridge_service"
  - "cex_deposit"
  - "otc_service"
  - "unidentified_service"
  - "intermediate"
  - "cluster"
• Add labels and notes:
  - Example labels: "Victim's funds", "Perpetrator's address", "Bridge to Tron", "Bridge to zkSync", "Bridge to SOL", "Binance deposit address", "HTX deposit address", "Bybit deposit address", "Suspected unidentified service", "Funds present 2.2 ETH".

3. Annotations
• Add entries like:

{
  "id": "comment-1",
  "label": "Victim's funds",
  "related_addresses": ["<victim-address>"],
  "related_steps": ["path-1:0"],
  "text": "Victim's funds leaving the wallet on Ethereum towards the perpetrator."
}

Output format

• Return the same JSON structure you received, but enriched:
  - case_meta (unchanged or minimally extended),
  - paths (with added step_type, service_label, protocol fields),
  - entities (with refined role, labels, notes),
  - annotations (extended with new comments).
• Do not output prose outside JSON.
• Be conservative: if unsure, keep roles as "intermediate" and explain uncertainty in notes."""

    return Agent(
        name="classification_agent",
        instructions=instructions,
        mcp_servers=[mcp_server],
    )


# ============================================================================
# Reporting / Diagram Agent
# ============================================================================

def create_reporting_agent() -> Agent:
    """
    Creates the Reporting/Diagram Agent that produces narratives and graph JSONs.
    """
    instructions = """You are the reporting and visualization agent.

Inputs

• Enriched classification JSON:
  - case_meta
  - paths (with step_type, service_label, protocol, etc.)
  - entities (with refined roles, labels, notes)
  - annotations
• Optional: visualization JSON (for label inspiration only).

Goals

1. Produce a short, clear narrative describing:
• Off-chain scam story (WhatsApp rental, job/investment, OTC, podcast scam, etc.).
• High-level on-chain flow:
  - Where victim funds were,
  - How they left the victim's control,
  - Bridges (e.g., Ethereum → Tron, Tron → zkSync, Ethereum → SOL),
  - Key services involved (bridges, CEXs, OTC).
• Current known status of funds.

2. Produce a graph-oriented JSON summarizing nodes, edges, and comments:
• Rich enough to be transformed into your visualization format (items, txs, connects, comments).

Narrative (summary_text)

• 1–3 short paragraphs in simple English.
• Include:
  - The scam's social engineering channel and story.
  - The date and asset (if available).
  - Main path of the funds:
    * Victim → Perpetrator (or directly to service),
    * Bridge → Destination chain,
    * Deposits to CEX / OTC / other services,
    * Residual funds.
  - Use technical terms like "bridge", "CEX deposit", "Tron", "zkSync" – but keep sentences straightforward.

Graph JSON

Return a JSON object with this top-level shape:

{
  "summary_text": "<human-readable narrative>",
  "graph": {
    "case_meta": { ... },
    "nodes": [
      {
        "id": "<address-or-tx-or-comment-id>",
        "type": "address" | "tx" | "comment",
        "address": "0x..." | "T..." | null,
        "tx_hash": "<hash>" | null,
        "chain": "eth" | "trx" | "zksync" | "sol" | null,
        "role": "victim" | "perpetrator" | "bridge_service" | "cex_deposit" | "otc_service" | "unidentified_service" | "intermediate" | null,
        "labels": ["Victim's funds", "Binance deposit address", "Bridge to Tron"],
        "metadata": {
          "asset": "USDT",
          "amount_estimate": 12345.67,
          "protocol": "layerzero" | "allbridge" | null
        }
      }
    ],
    "edges": [
      {
        "id": "<edge-id>",
        "from": "<node-id>",
        "to": "<node-id>",
        "relation": "input" | "output" | "flow" | "comment_link",
        "chain": "eth" | "trx" | "zksync" | "sol" | null,
        "asset": "USDT" | "ETH" | "TRX" | null,
        "amount_estimate": 66030.0,
        "step_type": "direct_transfer" | "bridge_in" | "bridge_out" | "service_deposit" | "internal_transfer" | null
      }
    ],
    "comments": [
      {
        "id": "comment-1",
        "label": "Victim's funds",
        "text": "Victim's funds on Ethereum before being bridged to Tron.",
        "related_nodes": ["<victim-address-node-id>"],
        "related_edges": ["<edge-id>"]
      }
    ]
  }
}

Labeling style

• Use comment labels that match the existing ecosystem:
  - "Victim's funds"
  - "Perpetrator's address"
  - "Bridge to Tron"
  - "Bridge to zkSync"
  - "Bridge to SOL"
  - "Binance deposit address"
  - "HTX deposit address"
  - "Bybit deposit address"
  - "FixedFloat deposit address"
  - "Exolix deposit address"
  - "Bitcoinvn deposit address"
  - "Shuffle deposit address"
  - "Stake deposit address"
  - "Suspected unidentified service"
  - "Funds present <amount>"

Rules

• Do not fabricate chains, roles, or amounts; derive everything from the enriched classification JSON.
• If something is unknown, use null or omit the field from metadata instead of making it up.
• Ensure summary_text is self-contained and understandable without the graph.
• The entire response from this agent must be valid JSON with the fields above."""

    return Agent(
        name="reporting_agent",
        instructions=instructions,
    )


# ============================================================================
# Main Orchestration Function
# ============================================================================

async def run_multi_agent_trace(
    mcp_server: MCPServer,
    case_description: str,
    victim_address: Optional[str] = None,
    tx_hashes: Optional[List[str]] = None,
    chains: Optional[List[str]] = None,
    asset_symbol: Optional[str] = None,
    approx_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main orchestration function that runs the multi-agent tracing pipeline.

    Args:
        mcp_server: The MCP server instance with tools
        case_description: User's description of the case
        victim_address: Optional victim address
        tx_hashes: Optional list of transaction hashes
        visualization_json: Optional visualization JSON for reference
        chains: Optional list of chain slugs (e.g. ["eth", "trx"])
        asset_symbol: Optional asset symbol (e.g. "USDT", "ETH")
        approx_date: Optional approximate date (YYYY-MM-DD format)

    Returns:
        Final report with summary_text and graph JSON
    """
    # Build case context
    case_context = build_case_context(
        case_description=case_description,
        victim_address=victim_address,
        tx_hashes=tx_hashes,
        chains=chains,
        asset_symbol=asset_symbol,
        approx_date=approx_date,
    )

    # Create all agents
    tracing_agent = create_tracing_agent(mcp_server)
    classification_agent = create_classification_agent(mcp_server)
    reporting_agent = create_reporting_agent()

    # Step 1: Run tracing agent
    with trace("Tracing Agent"):
        tracing_input = {
            "case_context": case_context,
            "victim_address": victim_address,
            "tx_hashes": tx_hashes or [],
        }
        tracing_result = await Runner.run(
            tracing_agent,
            input=json.dumps(tracing_input, indent=2),
        )

        # Parse tracing result
        try:
            tracing_data = json.loads(tracing_result.final_output)
            print(f"Tracing data: {tracing_data}")
        except json.JSONDecodeError:
            # If agent didn't return JSON, try to extract it
            tracing_data = {
                "case_meta": case_context,
                "paths": [],
                "entities": [],
                "annotations": []
            }

    # Step 2: Run classification agent
    with trace("Classification Agent"):
        classification_input = {
            "case_context": case_context,
            "tracing_result": tracing_data,
        }
        classification_result = await Runner.run(
            classification_agent,
            input=json.dumps(classification_input, indent=2),
        )

        # Parse classification result
        try:
            classification_data = json.loads(classification_result.final_output)
        except json.JSONDecodeError:
            # If agent didn't return JSON, use tracing data as fallback
            classification_data = tracing_data
            print(f"Classification data: {classification_data}")
    # Step 3: Run reporting agent
    with trace("Reporting Agent"):
        reporting_input = {
            "case_context": case_context,
            "classification_result": classification_data,
        }
        reporting_result = await Runner.run(
            reporting_agent,
            input=json.dumps(reporting_input, indent=2),
        )

        # Parse final result
        try:
            final_report = json.loads(reporting_result.final_output)
        except json.JSONDecodeError:
            # If not JSON, create a basic report
            final_report = {
                "summary_text": reporting_result.final_output,
                "graph": {
                    "case_meta": case_context,
                    "nodes": [],
                    "edges": [],
                    "comments": []
                }
            }
            print(f"Final report: {final_report}")
    return final_report
