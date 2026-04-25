"""Regression: bridge-detector API expects its own chain vocabulary
(``tron-mainnet`` / ``ethereum`` / ``binance-smart-chain``), not the
SAILS short codes the rest of the codebase uses.

Production failure this guards against (seen on 2026-04-24 while tracing
``e37253294b…`` on TRX):

    [TRACE] ❌ Tool error: Server error '500 Internal Server Error'
        for url 'https://api.bridge-detector.amlbot.com/v1/bridge/analyze'

The MCP server proxied our ``chain=trx`` straight through to the
bridge-detector backend, which only accepts ``tron-mainnet``. Translation
now happens inside ``MCPClient.bridge_analyze`` /
``MCPHTTPClient.bridge_analyze`` so every call path (dispatch table,
``api.py``, ``deterministic_tracer.py``) gets the right value.
"""
from __future__ import annotations

from agent.tool_dispatch import _BRIDGE_CHAIN_MAP, _bridge_chain


class TestBridgeChainMap:
    def test_trx_is_translated_to_tron_mainnet(self):
        # The bridge-detector identifies Tron as ``tron-mainnet`` (its
        # network identifier). Plain ``"tron"`` is rejected by the
        # service, so we must emit ``tron-mainnet`` everywhere.
        assert _bridge_chain("trx") == "tron-mainnet"

    def test_eth_is_translated_to_ethereum(self):
        assert _bridge_chain("eth") == "ethereum"

    def test_bsc_is_translated_to_binance_smart_chain(self):
        assert _bridge_chain("bsc") == "binance-smart-chain"
        assert _bridge_chain("bnb") == "binance-smart-chain"

    def test_matic_is_translated_to_polygon(self):
        assert _bridge_chain("matic") == "polygon"

    def test_already_full_name_passes_through_idempotent(self):
        """Applying the mapping twice must stay idempotent so a
        direct-call path and a dispatch-table path can both normalize
        without corrupting each other."""
        for short, full in _BRIDGE_CHAIN_MAP.items():
            assert _bridge_chain(full) == full, (
                f"expected {full!r} to pass through, got {_bridge_chain(full)!r}"
            )

    def test_unknown_chain_passes_through_normalized(self):
        assert _bridge_chain("FantasyChain") == "fantasychain"

    def test_empty_input_is_empty(self):
        assert _bridge_chain("") == ""
        assert _bridge_chain(None) == ""

    def test_whitespace_is_stripped(self):
        assert _bridge_chain("  trx  ") == "tron-mainnet"
        assert _bridge_chain("TRX") == "tron-mainnet"


class TestTronMainnetReverseMapping:
    """Bridge-detector responses may echo back ``tron-mainnet`` in
    ``dst_chain``. Our parsers must normalize it to internal ``trx``
    so cross-chain HopJob construction picks the right registry/unit
    rules — same way ``tron`` already maps."""

    def test_base_tracer_normalize_chain_handles_tron_mainnet(self):
        from agent.base_tracer import BaseTracer

        # Both the short and the network-id forms must collapse to
        # the canonical SAILS code so downstream code paths line up.
        assert BaseTracer._normalize_chain("tron-mainnet") == "trx"
        assert BaseTracer._normalize_chain("Tron-Mainnet") == "trx"
        assert BaseTracer._normalize_chain("  TRON-MAINNET  ") == "trx"

    def test_currency_registry_normalize_external_chain_handles_tron_mainnet(self):
        from agent.currency_registry import normalize_external_chain

        assert normalize_external_chain("tron-mainnet") == "trx"
        assert normalize_external_chain("Tron-Mainnet") == "trx"
