"""
HTTP client for MCP Server AMLBot.
Replaces stdio-based MCP client with HTTP-based one.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-wide async cache + in-flight dedupe for MCP tool calls.
#
# Scoped to MCPHTTPClient at class level so repeated traces within the same
# process (even via separate MCPHTTPClient instances) share cached reads.
# Keys embed user_id so tenancy never leaks between sessions.
#
# In-flight dedupe collapses N concurrent identical lookups (common when a
# trace fans out into a hub address like a CEX deposit or bridge vault that
# multiple paths reach simultaneously) into a single MCP round-trip.
# ---------------------------------------------------------------------------

_CACHE_DEFAULT_TTL = 600.0  # 10 minutes
_CACHE_TTL_BY_TOOL: dict[str, float] = {
    "bridge_analyze": 3600.0,   # bridge classification is historical, rarely changes
    "bridge-analyze": 3600.0,
    "get_address": 600.0,
    "get-address": 600.0,
    "get_extra_address_info": 600.0,
    "get-extra-address-info": 600.0,
    "token_stats": 600.0,
    "token-stats": 600.0,
    "expert_search": 600.0,
    "expert-search": 600.0,
    "token_transfers": 600.0,
    "token-transfers": 600.0,
    "get_transaction": 600.0,
    "get-transaction": 600.0,
    "get_position": 600.0,
    "get-position": 600.0,
    "all_txs": 120.0,           # outgoing-tx queries touched most often
    "all-txs": 120.0,
}
# Write/side-effect endpoints must never be cached.
_NO_CACHE_TOOLS = frozenset({
    "save-visualization",
    "save_visualization",
})
_CACHE_MAX_ENTRIES = 512


class VisualizationAPIClient:
    """Client for the visualization API (save and share)."""

    def __init__(self, api_url: str | None = None, user_id: str | None = None):
        """
        Initialize visualization API client.

        Args:
            api_url: Base URL for the API (defaults to NEXT_PUBLIC_API_URL env var)
            user_id: User ID for authentication
        """
        self.api_url = (api_url or os.getenv("NEXT_PUBLIC_API_URL", "")).rstrip('/')
        self.user_id = user_id
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=32, keepalive_expiry=30.0),
            http2=True,
        )

    async def save_visualization(self, visualization_data: dict[str, Any]) -> dict[str, Any]:
        """
        Save visualization and get its ID.

        POST {api_url}/api/pro/visualizations

        Args:
            visualization_data: The visualization data to save

        Returns:
            Response with visualization ID
        """
        url = f"{self.api_url}/api/pro/visualizations"

        headers = {
            "Content-Type": "application/json"
        }
        if self.user_id:
            headers["X-User-Id"] = self.user_id

        logger.info(f"Saving visualization to {url}")

        response = await self.client.post(
            url,
            json=visualization_data,
            headers=headers
        )

        response.raise_for_status()
        result = response.json()

        logger.info(f"Visualization saved with ID: {result.get('id') or result.get('data', {}).get('id')}")

        return result

    async def create_share_link(self, visualization_id: str) -> dict[str, Any]:
        """
        Create a shareable link for a visualization.

        POST {api_url}/api/pro/visualizations/{id}/share

        Args:
            visualization_id: The ID of the saved visualization

        Returns:
            Response with share link information
        """
        url = f"{self.api_url}/api/pro/visualizations/{visualization_id}/share"

        headers = {
            "Content-Type": "application/json"
        }
        if self.user_id:
            headers["X-User-Id"] = self.user_id

        logger.info(f"Creating share link for visualization {visualization_id}")

        response = await self.client.post(
            url,
            headers=headers
        )

        response.raise_for_status()
        result = response.json()

        logger.info(f"Share link created: {result.get('url') or result.get('data', {}).get('url')}")

        return result

    async def save_and_share(self, visualization_data: dict[str, Any]) -> dict[str, Any]:
        """
        Save visualization and immediately create a share link.

        Args:
            visualization_data: The visualization data to save

        Returns:
            Combined result with visualization ID and share link
        """
        # First save the visualization
        save_result = await self.save_visualization(visualization_data)

        # Extract ID (handle different response formats)
        viz_id = (
            save_result.get("id") or
            save_result.get("data", {}).get("id") or
            save_result.get("visualization_id")
        )

        if not viz_id:
            raise ValueError("Failed to get visualization ID from save response")

        # Create share link
        share_result = await self.create_share_link(viz_id)

        # Combine results
        return {
            "visualization_id": viz_id,
            "save_result": save_result,
            "share_result": share_result,
            "share_url": (
                share_result.get("url") or
                share_result.get("data", {}).get("url") or
                share_result.get("share_url")
            )
        }

    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MCPHTTPClient:
    """HTTP client for MCP Server tools. Compatible with MCPClient interface."""

    # Class-level caches so multiple MCPHTTPClient instances (one per trace)
    # share results within the same process lifetime. Per-user partitioning
    # comes from _cache_key() embedding user_id.
    _cache: dict[str, tuple[float, Any]] = {}
    _inflight: dict[str, "asyncio.Future[Any]"] = {}

    def __init__(self, mcp_server_url: str, user_id: str):
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.user_id = user_id
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100, keepalive_expiry=30.0),
            http2=True,
        )

    def _cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        try:
            serialized = json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)
        except Exception:
            serialized = repr(arguments)
        return f"{self.user_id}::{tool_name}::{serialized}"

    @classmethod
    def _cache_get(cls, key: str) -> Any | None:
        entry = cls._cache.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if expiry < time.monotonic():
            cls._cache.pop(key, None)
            return None
        return value

    @classmethod
    def _cache_put(cls, key: str, value: Any, tool_name: str) -> None:
        ttl = _CACHE_TTL_BY_TOOL.get(tool_name, _CACHE_DEFAULT_TTL)
        if ttl <= 0:
            return
        # Lightweight eviction: when at capacity, drop the entry with the
        # earliest expiry. Not true LRU but cheap and good enough for our
        # small working set.
        if len(cls._cache) >= _CACHE_MAX_ENTRIES:
            try:
                oldest_key = min(cls._cache.items(), key=lambda kv: kv[1][0])[0]
                cls._cache.pop(oldest_key, None)
            except ValueError:
                pass
        cls._cache[key] = (time.monotonic() + ttl, value)

    @classmethod
    def clear_cache(cls) -> None:
        """Used by tests / forced refresh paths."""
        cls._cache.clear()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool via HTTP with TTL cache + in-flight dedupe."""
        if tool_name in _NO_CACHE_TOOLS:
            return await self._call_tool_uncached(tool_name, arguments)

        key = self._cache_key(tool_name, arguments)

        cached = self._cache_get(key)
        if cached is not None:
            return cached

        inflight = self._inflight.get(key)
        if inflight is not None:
            return await inflight

        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._inflight[key] = future
        try:
            result = await self._call_tool_uncached(tool_name, arguments)
            self._cache_put(key, result, tool_name)
            if not future.done():
                future.set_result(result)
            return result
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            # Always drop the in-flight entry so a failure doesn't pin the key.
            self._inflight.pop(key, None)

    async def _call_tool_uncached(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Raw HTTP POST to the MCP /api/tools/call endpoint."""
        url = f"{self.mcp_server_url}/api/tools/call"

        response = await self.client.post(
            url,
            json={
                "tool_name": tool_name,
                "arguments": arguments,
                "user_id": self.user_id
            },
            headers={
                "X-User-Id": self.user_id,
                "Content-Type": "application/json"
            }
        )

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise Exception(data.get("error", "Unknown error"))

        return data.get("result")

    async def all_txs(
        self, address: str, blockchain_name: str,
        filter_criteria: dict | None = None,
        limit: int = 100, offset: int = 0,
        direction: str = "asc", order: str = "time",
        transaction_type: str = "all"
    ) -> dict[str, Any]:
        """Get all transactions for an address."""
        return await self.call_tool("all-txs", {
            "address": address,
            "blockchain_name": blockchain_name,
            "filter": filter_criteria,
            "limit": limit,
            "offset": offset,
            "direction": direction,
            "order": order,
            "transaction_type": transaction_type
        })

    async def get_transaction(
        self, address: str, tx_hash: str,
        blockchain_name: str, token_id: int = 0, path: str = "0"
    ) -> dict[str, Any]:
        """Get transaction details."""
        return await self.call_tool("get-transaction", {
            "address": address,
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name,
            "token_id": token_id,
            "path": path
        })

    async def get_address(self, blockchain_name: str, address: str) -> dict[str, Any]:
        """Get address information."""
        return await self.call_tool("get-address", {
            "blockchain_name": blockchain_name,
            "address": address
        })

    async def token_stats(self, blockchain_name: str, address: str) -> dict[str, Any]:
        """Get token statistics."""
        return await self.call_tool("token-stats", {
            "blockchain_name": blockchain_name,
            "address": address
        })

    async def get_extra_address_info(self, address: str, asset: str) -> dict[str, Any]:
        """Get extra address information including service platform detection."""
        return await self.call_tool("get-extra-address-info", {
            "address": address,
            "asset": asset
        })

    async def bridge_analyze(self, chain: str, tx_hash: str) -> dict[str, Any]:
        """Analyze bridge transaction."""
        return await self.call_tool("bridge-analyze", {
            "chain": chain,
            "tx_hash": tx_hash
        })

    async def expert_search(self, hash: str, filter: str = "explorer") -> dict[str, Any]:
        """Expert search."""
        return await self.call_tool("expert-search", {
            "hash": hash,
            "filter": filter
        })

    async def token_transfers(self, tx_hash: str, blockchain_name: str) -> dict[str, Any]:
        """Get token transfers."""
        return await self.call_tool("token-transfers", {
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name
        })

    async def get_position(
        self, address: str, tx_hash: str,
        blockchain_name: str, token_id: int = 0, path: str = "0"
    ) -> dict[str, Any]:
        """Get position information for transaction tracing."""
        return await self.call_tool("get-position", {
            "address": address,
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name,
            "token_id": token_id,
            "path": path
        })

    async def bridge_analyzer(self, chain: str, tx_hash: str) -> dict[str, Any]:
        """Alias for bridge_analyze for compatibility."""
        return await self.bridge_analyze(chain, tx_hash)

    async def save_and_share_visualization(self, data: dict[str, Any]) -> dict[str, Any]:
        """Save and share visualization."""
        return await self.call_tool("save-visualization", data)

    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()
