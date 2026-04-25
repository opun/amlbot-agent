"""Regression: every ``bridge_analyze`` call must include the
``model`` parameter in the request body.

The bridge-detector upstream service treats ``model`` as a required
field — without it the call is rejected. We pin to
``BRIDGE_ANALYZER_MODEL`` (``"bridge-analyzer-1"``) so tracer behavior
stays stable as the server rolls out newer analyzer versions.

Coverage:
  * ``MCPClient.bridge_analyze`` and ``bridge_analyzer`` (stdio path)
  * ``MCPHTTPClient.bridge_analyze`` and ``bridge_analyzer`` (HTTP path)
  * ``tool_dispatch._call_bridge_analyze`` (the dispatch table that
    proxies tracer-side ``_call_tool("bridge_analyze", …)`` calls)

If a future caller forgets ``model``, the default kicks in — but the
dispatch table propagates an explicit ``model`` whenever the tracer
supplies one (so the constant lives in one place).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from mcp.types import CallToolResult

from agent.mcp_client import MCPClient
from agent.mcp_http_client import MCPHTTPClient
from agent.tool_dispatch import (
    BRIDGE_ANALYZER_MODEL,
    _call_bridge_analyze,
)


def _stdio_client() -> tuple[MCPClient, AsyncMock]:
    server = AsyncMock()
    server.call_tool.return_value = CallToolResult(
        content=[],
        structuredContent={"is_bridge": False},
        isError=False,
    )
    return MCPClient(server), server.call_tool


# ── stdio MCPClient ──────────────────────────────────────────────────────


def test_stdio_bridge_analyze_includes_default_model():
    client, call_tool = _stdio_client()
    asyncio.run(client.bridge_analyze(chain="ETH", tx_hash="0xdead"))
    tool_name, arguments = call_tool.await_args.args
    assert tool_name == "bridge-analyze"
    assert arguments["model"] == BRIDGE_ANALYZER_MODEL == "bridge-analyzer-1"
    assert arguments["chain"] == "ethereum"  # ETH → ethereum translation
    assert arguments["tx_hash"] == "0xdead"


def test_stdio_bridge_analyze_respects_explicit_model():
    client, call_tool = _stdio_client()
    asyncio.run(client.bridge_analyze(
        chain="trx", tx_hash="0xabc", model="bridge-analyzer-2",
    ))
    _, arguments = call_tool.await_args.args
    assert arguments["model"] == "bridge-analyzer-2"


def test_stdio_bridge_analyzer_alias_threads_model_through():
    client, call_tool = _stdio_client()
    asyncio.run(client.bridge_analyzer(
        chain="bsc", tx_hash="0xfeed", model="bridge-analyzer-1",
    ))
    _, arguments = call_tool.await_args.args
    assert arguments["model"] == "bridge-analyzer-1"
    assert arguments["chain"] == "binance-smart-chain"


# ── HTTP MCPHTTPClient ───────────────────────────────────────────────────


class _FakeHTTPClient:
    """Minimal stand-in that captures call_tool invocations.

    We don't go through ``httpx`` because the model-param fix is purely
    about how the request body is constructed before dispatch — the
    transport layer is irrelevant.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {"is_bridge": False}


def _http_client_with_capture() -> tuple[MCPHTTPClient, _FakeHTTPClient]:
    fake = _FakeHTTPClient()
    # MCPHTTPClient.call_tool is what we override; bind directly.
    inst = MCPHTTPClient.__new__(MCPHTTPClient)
    inst.call_tool = fake.call_tool  # type: ignore[method-assign]
    return inst, fake


def test_http_bridge_analyze_includes_default_model():
    client, fake = _http_client_with_capture()
    asyncio.run(client.bridge_analyze(chain="ETH", tx_hash="0xdead"))
    assert len(fake.calls) == 1
    tool_name, arguments = fake.calls[0]
    assert tool_name == "bridge-analyze"
    assert arguments["model"] == "bridge-analyzer-1"
    assert arguments["chain"] == "ethereum"
    assert arguments["tx_hash"] == "0xdead"


def test_http_bridge_analyze_respects_explicit_model():
    client, fake = _http_client_with_capture()
    asyncio.run(client.bridge_analyze(
        chain="trx", tx_hash="0xabc", model="bridge-analyzer-2",
    ))
    _, arguments = fake.calls[0]
    assert arguments["model"] == "bridge-analyzer-2"


def test_http_bridge_analyzer_alias_threads_model():
    client, fake = _http_client_with_capture()
    asyncio.run(client.bridge_analyzer(
        chain="bsc", tx_hash="0xfeed",
    ))
    _, arguments = fake.calls[0]
    assert arguments["model"] == "bridge-analyzer-1"


# ── tool_dispatch (proxy used by tracer-side _call_tool) ─────────────────


class _StubBridgeClient:
    """Captures whatever ``bridge_analyze`` keyword args the dispatch
    table forwards. We verify by-keyword propagation because the
    constant default is emitted when the caller doesn't supply
    ``model`` (e.g. legacy callers in ``api.py``)."""

    def __init__(self) -> None:
        self.last_call: tuple | None = None

    async def bridge_analyze(self, chain, tx_hash, *, model=None):
        self.last_call = (chain, tx_hash, model)
        return {"is_bridge": False}


def test_dispatch_passes_model_from_args():
    """If the tracer's _call_tool sends ``model`` in the args dict, the
    dispatch table must forward that exact value."""
    stub = _StubBridgeClient()
    asyncio.run(_call_bridge_analyze(stub, {
        "model": "bridge-analyzer-2",
        "chain": "trx",
        "tx_hash": "0xabc",
    }))
    assert stub.last_call == ("trx", "0xabc", "bridge-analyzer-2")


def test_dispatch_falls_back_to_default_model_when_args_missing_it():
    """Older callers that haven't been migrated still work — dispatch
    fills in the pinned default rather than calling without ``model``."""
    stub = _StubBridgeClient()
    asyncio.run(_call_bridge_analyze(stub, {
        "chain": "trx",
        "tx_hash": "0xabc",
    }))
    assert stub.last_call == ("trx", "0xabc", "bridge-analyzer-1")
