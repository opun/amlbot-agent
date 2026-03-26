"""
MCP Tracer - extends BaseTracer with local MCP stdio-based tool execution.
Uses MCPClient to call tools via local MCP server process.
"""
import logging
from typing import Any

from .base_tracer import BaseTracer
from .mcp_client import MCPClient
from .tool_dispatch import dispatch_tool

logger = logging.getLogger("tracer")


class MCPTracer(BaseTracer):
    """Tracer that uses local MCP client via stdio."""

    def __init__(self, client: MCPClient):
        super().__init__()
        self.client = client

    def _get_client(self):
        return self.client

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool via MCP stdio client."""
        logger.info(f"Executing {tool_name} via MCP")
        return await dispatch_tool(self.client, tool_name, arguments)
