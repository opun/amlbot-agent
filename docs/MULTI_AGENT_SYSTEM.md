# Multi-Agent Crypto Tracing System

This document describes the multi-agent system for retail-level crypto tracing investigations.

## Overview

The system consists of four specialized agents that work together to trace stolen cryptocurrency funds:

1. **Router/Case Orchestrator** - Parses case descriptions and orchestrates the pipeline
2. **Tracing Agent** - Reconstructs on-chain paths from victim addresses
3. **Classification/Bridge Agent** - Detects bridges and classifies services
4. **Reporting/Diagram Agent** - Produces narratives and graph JSONs

## Architecture

```
User Input (case description, addresses, tx hashes)
    ↓
Router/Orchestrator (builds case_context)
    ↓
Tracing Agent (reconstructs paths using MCP tools)
    ↓
Classification Agent (detects bridges, classifies services)
    ↓
Reporting Agent (generates narrative + graph JSON)
    ↓
Final Report (summary_text + graph)
```

## Usage

### Basic Usage

```python
from agent.multi_agent_system import run_multi_agent_trace
from agents.mcp import MCPServerStdio

async with MCPServerStdio(...) as server:
    result = await run_multi_agent_trace(
        mcp_server=server,
        case_description="Victim lost funds in WhatsApp rental scam",
        victim_address="0x1234...",
        tx_hashes=["0xabcd..."],
    )

    print(result["summary_text"])
    print(json.dumps(result["graph"], indent=2))
```

### Command Line Usage

```bash
# Interactive mode
python -m agent.main_multi_agent

# Command line mode
python -m agent.main_multi_agent \
    "Case description" \
    "0x1234..." \
    "0xabcd,0xefgh" \
    "path/to/visualization.json"
```

## Case Types

The system recognizes these case archetypes:

- `job_investment_platform` - Job/investment platform scams (e.g., "eAccountable")
- `social_media_investment` - Social media/romance investment scams
- `bridge_easy_rental` - WhatsApp rental scams with wallet compromise
- `bridge_mid_wallet_compromise` - Wallet compromise with multiple bridges
- `bridge_mid_device_compromise` - Device compromise with wallet drain
- `otc_cash_usdt` - OTC cash-for-USDT scams
- `podcast_remote_metamask` - Remote screen-sharing MetaMask compromise
- `other` - Other types

## Tools Available

### MCP Server Tools

All agents have access to these tools via the MCP server:

- `expert-search` - Search for addresses/txs/entities
- `get-address` - Get address details and risk scores
- `token-stats` - Get token statistics
- `all-txs` - List transactions for an address
- `get-transaction` - Get detailed transaction info
- `get-position` - Follow funds forward from a transaction
- `get-extra-address-info` - Get service platform tags
- `bridge-analyze` - Detect bridge operations (LayerZero, AllBridge)

## Output Format

The system returns a JSON object with:

```json
{
  "summary_text": "Human-readable narrative of the case...",
  "graph": {
    "case_meta": {
      "case_id": "case-20250101-120000",
      "chains": ["eth", "trx"],
      "asset_symbol": "USDT",
      "approx_date": "2025-01-01",
      "victim_address": "0x1234...",
      "narrative_summary": "...",
      "experiment_case_type": "bridge_easy_rental"
    },
    "nodes": [
      {
        "id": "node-1",
        "type": "address",
        "address": "0x1234...",
        "chain": "eth",
        "role": "victim",
        "labels": ["Victim's funds"],
        "metadata": {
          "asset": "USDT",
          "amount_estimate": 66030.0
        }
      }
    ],
    "edges": [
      {
        "id": "edge-1",
        "from": "node-1",
        "to": "node-2",
        "relation": "flow",
        "chain": "eth",
        "asset": "USDT",
        "amount_estimate": 66030.0,
        "step_type": "direct_transfer"
      }
    ],
    "comments": [
      {
        "id": "comment-1",
        "label": "Victim's funds",
        "text": "Victim's funds on Ethereum before being bridged to Tron.",
        "related_nodes": ["node-1"],
        "related_edges": ["edge-1"]
      }
    ]
  }
}
```

## Visualization JSON Support

The system can accept visualization JSONs from the existing AML visualizer as reference. These are used for:

- Label inspiration (e.g., "Bridge to Tron", "Binance deposit address")
- Cross-checking amounts and timestamps
- Understanding path structures

**Important**: If `helpers.isConnectionBasedMode === true`, the visualization is ignored entirely.

## Agent Details

### Tracing Agent

- **Tools**: expert-search, get-address, all-txs, get-transaction, get-position, token-stats, get-extra-address-info
- **Output**: JSON with paths, entities, and annotations
- **Focus**: Reconstructs 1-3 high-confidence paths from victim to services/bridges

### Classification Agent

- **Tools**: get-address, token-stats, get-extra-address-info, all-txs, get-transaction, bridge-analyze
- **Input**: Tracing result + optional visualization JSON
- **Output**: Enriched paths with step types, service labels, bridge protocol info
- **Focus**: Detects bridges (LayerZero, AllBridge), classifies services (CEX, OTC, mixers)

### Reporting Agent

- **Tools**: None (pure reasoning)
- **Input**: Classification result + optional visualization JSON
- **Output**: Human-readable narrative + graph JSON
- **Focus**: Produces final report suitable for UI rendering

## Implementation Notes

- The system uses OpenAI's agents library for agent orchestration
- MCP server provides tools via Docker container
- All agents are stateless and can be run independently
- The router function builds case_context from user input (can be enhanced with LLM parsing)

## Future Enhancements

- Enhanced case_context parsing using LLM
- Support for more bridge protocols
- Real-time API endpoints for get_extra_address_info and bridge_analyze
- Batch processing for multiple cases
- Integration with existing AMLBot UI
