"""
MCP Tracer - extends BaseTracer with local MCP stdio-based tool execution.
Uses MCPClient to call tools via local MCP server process.
"""
import logging
from typing import Any, Dict

from .base_tracer import BaseTracer
from .mcp_client import MCPClient

logger = logging.getLogger("tracer")


class MCPTracer(BaseTracer):
    """Tracer that uses local MCP client via stdio."""

    def __init__(self, client: MCPClient):
        super().__init__()
        self.client = client

    def _get_client(self):
        return self.client

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool via MCP stdio client."""
        logger.info(f"🔧 Executing {tool_name} via MCP")

        # MCP tool names use hyphens, convert from underscores
        mcp_tool_name = tool_name.replace("_", "-")

        if tool_name == "expert_search":
            return await self.client.expert_search(
                arguments["hash"],
                arguments.get("filter", "explorer")
            )
        elif tool_name == "get_address":
            return await self.client.get_address(
                arguments["address"],
                arguments["blockchain_name"]
            )
        elif tool_name == "token_stats":
            return await self.client.token_stats(
                arguments["blockchain_name"],
                arguments["address"]
            )
        elif tool_name == "all_txs":
            limit = int(arguments.get("limit", 20))
            offset = int(arguments.get("offset", 0))
            limit = max(1, min(limit, 100))
            offset = max(0, min(offset, 1000))
            return await self.client.all_txs(
                arguments["address"],
                arguments["blockchain_name"],
                arguments.get("filter"),
                limit,
                offset,
                arguments.get("direction", "asc"),
                arguments.get("order", "time"),
                arguments.get("transaction_type", "all")
            )
        elif tool_name == "get_transaction":
            return await self.client.get_transaction(
                arguments["address"],
                arguments["tx_hash"],
                arguments["blockchain_name"],
                arguments.get("token_id", 0),
                arguments.get("path", "0")
            )
        elif tool_name == "get_position":
            return await self.client.get_position(
                arguments["address"],
                arguments["tx_hash"],
                arguments["blockchain_name"],
                arguments.get("token_id", 0),
                arguments.get("path", "0")
            )
        elif tool_name == "get_extra_address_info":
            return await self.client.get_extra_address_info(
                arguments["address"],
                arguments["asset"]
            )
        elif tool_name == "bridge_analyze":
            return await self.client.bridge_analyze(
                arguments["chain"],
                arguments["tx_hash"]
            )
        elif tool_name == "token_transfers":
            return await self.client.token_transfers(
                arguments["tx_hash"],
                arguments["blockchain_name"]
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
