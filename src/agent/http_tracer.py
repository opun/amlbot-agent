"""
HTTP Tracer - extends BaseTracer with HTTP-based tool execution.
Uses MCPHTTPClient to call remote MCP server via HTTP API.
"""
import logging
from typing import Any

from .base_tracer import BaseTracer
from .mcp_http_client import MCPHTTPClient
from .tool_dispatch import dispatch_tool

logger = logging.getLogger("tracer")


class HTTPTracer(BaseTracer):
    """Tracer that uses HTTP client to call MCP tools remotely."""

    def __init__(self, client: MCPHTTPClient):
        super().__init__()
        self.client = client

    def _get_client(self):
        return self.client

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool via HTTP API."""
        logger.info(f"Executing {tool_name} via HTTP")
        return await dispatch_tool(self.client, tool_name, arguments)
