"""
Shared tool dispatch logic for MCP and HTTP tracers.
Maps tool names to client method calls via a dispatch table,
eliminating duplicated if/elif chains.
"""
from enum import StrEnum
from typing import Any


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


async def _call_bridge_analyze(client: Any, args: dict[str, Any]) -> Any:
    return await client.bridge_analyze(
        args["chain"],
        args["tx_hash"],
    )


async def _call_token_transfers(client: Any, args: dict[str, Any]) -> Any:
    return await client.token_transfers(
        args["tx_hash"],
        args["blockchain_name"],
    )
