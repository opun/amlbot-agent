"""
Shared tool dispatch logic for MCP and HTTP tracers.
Maps tool names to client method calls via a dispatch table,
eliminating duplicated if/elif chains.
"""
from enum import StrEnum
from typing import Any

# Model id for the upstream bridge-detector service. The API exposes
# multiple analyzer versions ("bridge-analyzer-1", future "bridge-analyzer-2",
# …); we pin to v1 so behavior stays stable as the server rolls out newer
# models. Every ``bridge_analyze`` invocation includes this in the request
# body — the bridge-detector treats it as a required parameter.
BRIDGE_ANALYZER_MODEL = "bridge-analyzer-1"


class ToolName(StrEnum):
    EXPERT_SEARCH = "expert_search"
    GET_ADDRESS = "get_address"
    TOKEN_STATS = "token_stats"
    ALL_TXS = "all_txs"
    GET_TRANSACTION = "get_transaction"
    GET_POSITION = "get_position"
    GET_EXTRA_ADDRESS_INFO = "get_extra_address_info"
    BRIDGE_ANALYZE = "bridge_analyze"
    TOKEN_TRANSFERS = "token_transfers"


async def dispatch_tool(client: Any, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate client method."""
    dispatch = {
        ToolName.EXPERT_SEARCH: _call_expert_search,
        ToolName.GET_ADDRESS: _call_get_address,
        ToolName.TOKEN_STATS: _call_token_stats,
        ToolName.ALL_TXS: _call_all_txs,
        ToolName.GET_TRANSACTION: _call_get_transaction,
        ToolName.GET_POSITION: _call_get_position,
        ToolName.GET_EXTRA_ADDRESS_INFO: _call_get_extra_address_info,
        ToolName.BRIDGE_ANALYZE: _call_bridge_analyze,
        ToolName.TOKEN_TRANSFERS: _call_token_transfers,
    }

    try:
        key = ToolName(tool_name)
    except ValueError:
        raise ValueError(f"Unknown tool: {tool_name}") from None

    handler = dispatch[key]
    return await handler(client, arguments)


async def _call_expert_search(client: Any, args: dict[str, Any]) -> Any:
    return await client.expert_search(
        args["hash"],
        args.get("filter", "explorer"),
    )


async def _call_get_address(client: Any, args: dict[str, Any]) -> Any:
    return await client.get_address(
        args["blockchain_name"],
        args["address"],
    )


async def _call_token_stats(client: Any, args: dict[str, Any]) -> Any:
    return await client.token_stats(
        args["blockchain_name"],
        args["address"],
    )


async def _call_all_txs(client: Any, args: dict[str, Any]) -> Any:
    limit = max(1, min(int(args.get("limit", 20)), 100))
    offset = max(0, min(int(args.get("offset", 0)), 1000))
    return await client.all_txs(
        args["address"],
        args["blockchain_name"],
        args.get("filter"),
        limit,
        offset,
        args.get("direction", "asc"),
        args.get("order", "time"),
        args.get("transaction_type", "all"),
    )


async def _call_get_transaction(client: Any, args: dict[str, Any]) -> Any:
    return await client.get_transaction(
        args["address"],
        args["tx_hash"],
        args["blockchain_name"],
        args.get("token_id", 0),
        args.get("path", "0"),
    )


async def _call_get_position(client: Any, args: dict[str, Any]) -> Any:
    return await client.get_position(
        args["address"],
        args["tx_hash"],
        args["blockchain_name"],
        args.get("token_id", 0),
        args.get("path", "0"),
    )


async def _call_get_extra_address_info(client: Any, args: dict[str, Any]) -> Any:
    return await client.get_extra_address_info(
        args["address"],
        args["asset"],
    )


# Source of truth for accepted chain slugs:
# https://api.bridge-detector.amlbot.com/v1/bridge/chains
# Keep values aligned with that catalog to avoid ``unsupported_chain`` 400s.
_BRIDGE_CHAIN_MAP: dict[str, str] = {
    # Tron is exposed to bridge-detector as ``tron-mainnet`` (the
    # service's network identifier); plain ``"tron"`` is rejected.
    "trx": "tron-mainnet",
    "eth": "ethereum",
    "bsc": "bsc",
    "bnb": "bsc",
    "matic": "polygon",
    "arb": "arbitrum",
    "op": "optimism",
    "avax": "avalanche",
    "base": "base",
    "sol": "solana",
    "btc": "bitcoin",
}


def _bridge_chain(chain: str | None) -> str:
    if not chain:
        return ""
    normalized = chain.strip().lower()
    return _BRIDGE_CHAIN_MAP.get(normalized, normalized)


async def _call_bridge_analyze(client: Any, args: dict[str, Any]) -> Any:
    # No chain translation here — both ``MCPClient.bridge_analyze`` and
    # ``MCPHTTPClient.bridge_analyze`` apply ``_bridge_chain`` internally
    # so the mapping is uniform regardless of call path (dispatch table
    # vs direct ``client.bridge_analyze`` calls from ``api.py`` /
    # ``deterministic_tracer.py``). Passing it here too would be a
    # (harmless but confusing) no-op second application.
    #
    # ``model`` is plumbed through with a default so existing callers
    # (which don't yet supply it) keep working.
    return await client.bridge_analyze(
        args["chain"],
        args["tx_hash"],
        model=args.get("model") or BRIDGE_ANALYZER_MODEL,
    )


async def _call_token_transfers(client: Any, args: dict[str, Any]) -> Any:
    return await client.token_transfers(
        args["tx_hash"],
        args["blockchain_name"],
    )
