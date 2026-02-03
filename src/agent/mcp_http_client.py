"""
HTTP client for MCP Server AMLBot.
Replaces stdio-based MCP client with HTTP-based one.
"""
import httpx
import os
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VisualizationAPIClient:
    """Client for the visualization API (save and share)."""

    def __init__(self, api_url: Optional[str] = None, user_id: Optional[str] = None):
        """
        Initialize visualization API client.

        Args:
            api_url: Base URL for the API (defaults to NEXT_PUBLIC_API_URL env var)
            user_id: User ID for authentication
        """
        self.api_url = (api_url or os.getenv("NEXT_PUBLIC_API_URL", "")).rstrip('/')
        self.user_id = user_id
        self.client = httpx.AsyncClient(timeout=60.0)

    async def save_visualization(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
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

    async def create_share_link(self, visualization_id: str) -> Dict[str, Any]:
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

    async def save_and_share(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
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

    def __init__(self, mcp_server_url: str, user_id: str):
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.user_id = user_id
        self.client = httpx.AsyncClient(timeout=60.0)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool via HTTP."""
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
        filter_criteria: Optional[Dict] = None,
        limit: int = 100, offset: int = 0,
        direction: str = "asc", order: str = "time",
        transaction_type: str = "all"
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
        """Get transaction details."""
        return await self.call_tool("get-transaction", {
            "address": address,
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name,
            "token_id": token_id,
            "path": path
        })

    async def get_address(self, blockchain_name: str, address: str) -> Dict[str, Any]:
        """Get address information."""
        return await self.call_tool("get-address", {
            "blockchain_name": blockchain_name,
            "address": address
        })

    async def token_stats(self, blockchain_name: str, address: str) -> Dict[str, Any]:
        """Get token statistics."""
        return await self.call_tool("token-stats", {
            "blockchain_name": blockchain_name,
            "address": address
        })

    async def get_extra_address_info(self, address: str, asset: str) -> Dict[str, Any]:
        """Get extra address information including service platform detection."""
        return await self.call_tool("get-extra-address-info", {
            "address": address,
            "asset": asset
        })

    async def bridge_analyze(self, chain: str, tx_hash: str) -> Dict[str, Any]:
        """Analyze bridge transaction."""
        return await self.call_tool("bridge-analyze", {
            "chain": chain,
            "tx_hash": tx_hash
        })

    async def expert_search(self, hash: str, filter: str = "explorer") -> Dict[str, Any]:
        """Expert search."""
        return await self.call_tool("expert-search", {
            "hash": hash,
            "filter": filter
        })

    async def token_transfers(self, tx_hash: str, blockchain_name: str) -> Dict[str, Any]:
        """Get token transfers."""
        return await self.call_tool("token-transfers", {
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name
        })

    async def bridge_analyzer(self, chain: str, tx_hash: str) -> Dict[str, Any]:
        """Alias for bridge_analyze for compatibility."""
        return await self.bridge_analyze(chain, tx_hash)

    async def save_and_share_visualization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save and share visualization."""
        return await self.call_tool("save-visualization", data)

    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()
