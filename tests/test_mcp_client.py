"""Tests for the stdio MCPClient (direct MCPServer.call_tool wrapper)."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from agent.mcp_client import MCPClient


def _make_client() -> tuple[MCPClient, AsyncMock]:
    server = AsyncMock()
    return MCPClient(server), server.call_tool


def test_all_txs_returns_structured_content():
    client, call_tool = _make_client()
    call_tool.return_value = CallToolResult(
        content=[],
        structuredContent={"txs": [{"hash": "0xabc"}]},
        isError=False,
    )

    result = asyncio.run(client.all_txs(
        address="0xdead",
        blockchain_name="ETH",
        limit=10,
    ))

    assert result == {"txs": [{"hash": "0xabc"}]}
    call_tool.assert_awaited_once()
    tool_name, arguments = call_tool.await_args.args
    assert tool_name == "all-txs"
    assert arguments["address"] == "0xdead"
    assert arguments["blockchain_name"] == "ETH"
    assert arguments["limit"] == 10
    # defaults preserved
    assert arguments["direction"] == "asc"
    assert arguments["filter"] is None


def test_get_transaction_falls_back_to_text_content():
    client, call_tool = _make_client()
    call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text='{"ok": true, "tx_hash": "0xff"}')],
        structuredContent=None,
        isError=False,
    )

    result = asyncio.run(client.get_transaction(
        address="0x1",
        tx_hash="0xff",
        blockchain_name="ETH",
    ))

    assert result == {"ok": True, "tx_hash": "0xff"}


def test_non_json_text_returns_raw_output():
    client, call_tool = _make_client()
    call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="no-json-here")],
        structuredContent=None,
        isError=False,
    )

    result = asyncio.run(client.expert_search(hash="0xabc"))
    assert result == {"raw_output": "no-json-here"}


def test_iserror_raises_runtime_error():
    client, call_tool = _make_client()
    call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="boom")],
        structuredContent=None,
        isError=True,
    )

    with pytest.raises(RuntimeError, match="bridge-analyze"):
        asyncio.run(client.bridge_analyze(chain="ETH", tx_hash="0xdead"))


def test_save_and_share_visualization_passes_payload_verbatim():
    client, call_tool = _make_client()
    call_tool.return_value = CallToolResult(
        content=[],
        structuredContent={"share_url": "https://example/share/abc"},
        isError=False,
    )

    payload = {
        "title": "Case 001",
        "type": "trace_graph",
        "payload": {"nodes": [1, 2]},
        "helpers": {},
        "extras": {},
    }
    result = asyncio.run(client.save_and_share_visualization(payload))

    assert result == {"share_url": "https://example/share/abc"}
    tool_name, arguments = call_tool.await_args.args
    assert tool_name == "save-visualization"
    assert arguments is payload


def test_bridge_analyzer_aliases_to_bridge_analyze():
    client, call_tool = _make_client()
    call_tool.return_value = CallToolResult(
        content=[],
        structuredContent={"bridge": "thorchain"},
        isError=False,
    )

    result = asyncio.run(client.bridge_analyzer(chain="ETH", tx_hash="0xdead"))

    assert result == {"bridge": "thorchain"}
    tool_name, _ = call_tool.await_args.args
    assert tool_name == "bridge-analyze"
