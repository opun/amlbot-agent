"""Regression: MCPHTTPClient must surface full error context, not
``Exception('')``.

Production log (2026-04-24, trace ``…_trx_USDT_e37253294b__trace_5c.jsonl``)
showed this useless traceback because MCP returned ``success=false`` with
an empty ``error`` field and the old code did
``Exception(data.get("error", "Unknown error"))`` — ``.get()`` returns
``""`` when the key IS present, never the default::

    ERROR:asyncio:Future exception was never retrieved
    future: <Future finished exception=Exception('')>
    ...
      File ".../mcp_http_client.py", line 334, in _call_tool_uncached
        raise exc
    Exception

Fix: in ``mcp_http_client._call_tool_uncached`` use ``data.get("error") or
"Unknown error"`` (falsy-coalesce) AND fall back to a self-describing
message that names the tool and dumps the full MCP body. Plus attach
``tool_name`` to the exception so ``base_tracer._call_tool``'s error
logger has a clean handle.

The companion asyncio "Future exception was never retrieved" noise is
silenced by attaching a ``done_callback`` to the in-flight dedupe
future that consumes the exception — the real raise path still
propagates to the caller.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.mcp_http_client import MCPHTTPClient


def _stub_response(status_code: int = 200, body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "http://mcp/api/tools/call")
    content = json.dumps(body or {}).encode("utf-8")
    return httpx.Response(
        status_code=status_code,
        request=request,
        content=content,
        headers={"content-type": "application/json"},
    )


class _StubHttpx:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._response


def _make_client(response):
    # Skip __init__ (httpx.AsyncClient + env lookups). Inject minimal state.
    c = MCPHTTPClient.__new__(MCPHTTPClient)
    c.client = _StubHttpx(response)
    c.mcp_server_url = "http://mcp"
    c.user_id = "u1"
    c._inflight = {}
    return c


class TestEmptyErrorField:
    def test_empty_error_string_yields_descriptive_message(self):
        """``data.get("error", "Unknown error")`` returns ``""`` when the
        key is present with an empty value. The fix uses falsy-coalesce
        and a tool-named fallback so the log isn't ``Exception('')``."""
        resp = _stub_response(200, {"success": False, "error": ""})
        c = _make_client(resp)
        with pytest.raises(Exception) as exc_info:
            asyncio.run(c._call_tool_uncached("token-stats", {"address": "T…"}))
        msg = str(exc_info.value)
        assert msg != ""
        assert "token-stats" in msg
        assert "success=false" in msg or "empty error" in msg

    def test_populated_error_string_is_preserved(self):
        """Non-empty error strings from MCP must pass through verbatim so
        the original upstream detail survives."""
        resp = _stub_response(200, {
            "success": False,
            "error": "Server error '500 Internal Server Error' for url 'https://api.bridge-detector.amlbot.com/v1/bridge/analyze'",
        })
        c = _make_client(resp)
        with pytest.raises(Exception) as exc_info:
            asyncio.run(c._call_tool_uncached("bridge-analyze", {"chain": "tron"}))
        assert "bridge-detector" in str(exc_info.value)

    def test_exception_carries_request_and_response_metadata(self):
        resp = _stub_response(200, {"success": False, "error": "boom"})
        c = _make_client(resp)
        with pytest.raises(Exception) as exc_info:
            asyncio.run(c._call_tool_uncached("get-address", {"address": "T…"}))
        exc = exc_info.value
        assert getattr(exc, "tool_name", None) == "get-address"
        assert getattr(exc, "response_status", None) == 200
        assert getattr(exc, "response_body", None) == {"success": False, "error": "boom"}
        req = getattr(exc, "request_body", None)
        assert req is not None
        assert req["tool_name"] == "get-address"


class TestInflightExceptionHandled:
    """The ``call_tool`` dedupe wrapper used to orphan exceptions on the
    in-flight future — Python's asyncio then printed ``Future exception
    was never retrieved`` at ERROR level, duplicating the real traceback
    already logged by ``base_tracer._call_tool``.

    The fix attaches a ``done_callback`` that reads ``f.exception()``,
    marking it as retrieved without interfering with the outer raise.
    """

    def test_failing_call_does_not_leave_unretrieved_future(self):
        resp = _stub_response(200, {"success": False, "error": "nope"})
        c = _make_client(resp)

        caught_warnings: list[str] = []

        async def _run():
            # Patch asyncio's default loop exception handler to surface
            # the "Future exception was never retrieved" message if the
            # regression returns.
            loop = asyncio.get_running_loop()
            orig_handler = loop.get_exception_handler()

            def _handler(_loop, ctx):
                caught_warnings.append(ctx.get("message", ""))

            loop.set_exception_handler(_handler)
            try:
                with pytest.raises(Exception):
                    await c.call_tool("bridge-analyze", {"chain": "tron"})
                # Give the event loop a chance to process orphaned
                # future finalizers.
                await asyncio.sleep(0)
            finally:
                loop.set_exception_handler(orig_handler)

        asyncio.run(_run())

        unretrieved = [w for w in caught_warnings if "never retrieved" in w]
        assert not unretrieved, (
            f"in-flight future left an unretrieved exception: {unretrieved}"
        )
