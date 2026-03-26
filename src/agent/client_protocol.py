"""
Protocol definition for MCP clients.
Allows both stdio-based MCPClient and HTTP-based MCPHTTPClient to be used interchangeably.
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Protocol for MCP client implementations."""

    async def all_txs(
        self,
        address: str,
        blockchain_name: str,
        filter_criteria: dict | None = None,
        limit: int = 100,
        offset: int = 0,
        direction: str = "asc",
        order: str = "time",
        transaction_type: str = "all"
    ) -> dict[str, Any]:
        """Get all transactions for an address."""
        ...

    async def get_transaction(
        self,
        address: str,
        tx_hash: str,
        blockchain_name: str,
        token_id: int = 0,
        path: str = "0"
    ) -> dict[str, Any]:
        """Get transaction details."""
        ...

    async def get_address(
        self,
        blockchain_name: str,
        address: str
    ) -> dict[str, Any]:
        """Get address information."""
        ...

    async def token_stats(
        self,
        blockchain_name: str,
        address: str
    ) -> dict[str, Any]:
        """Get token statistics."""
        ...

    async def get_extra_address_info(
        self,
        address: str,
        blockchain_name: str
    ) -> dict[str, Any]:
        """Get extra address information."""
        ...

    async def bridge_analyzer(
        self,
        chain: str,
        tx_hash: str
    ) -> dict[str, Any]:
        """Analyze bridge transaction."""
        ...

    async def expert_search(
        self,
        hash: str,
        filter: str = "explorer"
    ) -> dict[str, Any]:
        """Expert search."""
        ...

    async def token_transfers(
        self,
        tx_hash: str,
        blockchain_name: str
    ) -> dict[str, Any]:
        """Get token transfers."""
        ...
