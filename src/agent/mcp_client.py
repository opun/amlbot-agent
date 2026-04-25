"""
Stdio MCP client — direct tool calls via ``agents.mcp.MCPServer.call_tool``.

Mirrors the method shape of ``MCPHTTPClient`` so both satisfy
``client_protocol.MCPClientProtocol`` and feed ``tool_dispatch.dispatch_tool``
uniformly.  Previously this module round-tripped every tool through an
``Agent(...) / Runner.run(...)`` — pure LLM overhead, no decisions.
"""
import json
from typing import Any

from agents.mcp import MCPServer


class MCPClient:
    def __init__(self, mcp_server: MCPServer):
        self.server = mcp_server

    async def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.server.call_tool(tool_name, arguments)

        if result.isError:
            raise RuntimeError(
                f"MCP tool {tool_name!r} returned error: {result}"
            )

        if result.structuredContent is not None:
            return result.structuredContent

        for block in result.content or []:
            text = getattr(block, "text", None)
            if text is None:
                continue
            try:
                parsed: Any = json.loads(text)
            except json.JSONDecodeError:
                return {"raw_output": text}
            if isinstance(parsed, dict):
                return parsed
            return {"raw_output": text}
        return {}

    async def all_txs(
        self,
        address: str,
        blockchain_name: str,
        filter_criteria: dict | None = None,
        limit: int = 100,
        offset: int = 0,
        direction: str = "asc",
        order: str = "time",
        transaction_type: str = "all",
    ) -> dict[str, Any]:
        return await self._call("all-txs", {
            "address": address,
            "blockchain_name": blockchain_name,
            "filter": filter_criteria,
            "limit": limit,
            "offset": offset,
            "direction": direction,
            "order": order,
            "transaction_type": transaction_type,
        })

    async def get_transaction(
        self,
        address: str,
        tx_hash: str,
        blockchain_name: str,
        token_id: int = 0,
        path: str = "0",
    ) -> dict[str, Any]:
        return await self._call("get-transaction", {
            "address": address,
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name,
            "token_id": token_id,
            "path": path,
        })

    async def get_address(self, blockchain_name: str, address: str) -> dict[str, Any]:
        return await self._call("get-address", {
            "blockchain_name": blockchain_name,
            "address": address,
        })

    async def token_stats(self, blockchain_name: str, address: str) -> dict[str, Any]:
        return await self._call("token-stats", {
            "blockchain_name": blockchain_name,
            "address": address,
        })

    async def get_extra_address_info(self, address: str, blockchain_name: str) -> dict[str, Any]:
        # MCP tool parameter is literally named ``asset`` but receives the
        # chain/currency code (``ETH`` etc.) — same quirk as MCPHTTPClient.
        return await self._call("get-extra-address-info", {
            "address": address,
            "asset": blockchain_name,
        })

    async def bridge_analyzer(
        self, chain: str, tx_hash: str, model: str | None = None,
    ) -> dict[str, Any]:
        """Alias for ``bridge_analyze`` — preserved for external callers."""
        return await self.bridge_analyze(chain, tx_hash, model=model)

    async def bridge_analyze(
        self, chain: str, tx_hash: str, model: str | None = None,
    ) -> dict[str, Any]:
        # Translate our short codes (``trx``, ``eth``, …) to what the
        # bridge-detector API expects (``tron``, ``ethereum``, …). See the
        # matching override in ``MCPHTTPClient.bridge_analyze`` — both
        # clients feed ``tool_dispatch`` and ``api.py`` directly, so the
        # mapping lives on each client instead of only at dispatch time.
        #
        # ``model`` selects the analyzer version on the upstream service
        # (the bridge-detector treats it as a required body field). We
        # default to the pinned id from ``tool_dispatch`` so direct
        # callers that don't pass it explicitly stay on the same model.
        from agent.tool_dispatch import BRIDGE_ANALYZER_MODEL, _bridge_chain
        return await self._call("bridge-analyze", {
            "model": model or BRIDGE_ANALYZER_MODEL,
            "chain": _bridge_chain(chain),
            "tx_hash": tx_hash,
        })

    async def get_position(
        self,
        address: str,
        tx_hash: str,
        blockchain_name: str,
        token_id: int = 0,
        path: str = "0",
    ) -> dict[str, Any]:
        return await self._call("get-position", {
            "address": address,
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name,
            "token_id": token_id,
            "path": path,
        })

    async def expert_search(self, hash: str, filter: str = "explorer") -> dict[str, Any]:
        return await self._call("expert-search", {
            "hash": hash,
            "filter": filter,
        })

    async def token_transfers(self, tx_hash: str, blockchain_name: str) -> dict[str, Any]:
        return await self._call("token-transfers", {
            "tx_hash": tx_hash,
            "blockchain_name": blockchain_name,
        })

    async def save_and_share_visualization(self, data: dict[str, Any]) -> dict[str, Any]:
        return await self._call("save-visualization", data)
