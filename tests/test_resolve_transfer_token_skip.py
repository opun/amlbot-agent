"""Regression tests for the USDT-contract-as-destination bug.

Historical failure
------------------
When a trace is walking a chain's native asset (asset="ETH" on eth),
``_resolve_transfer`` used to jump straight to ``get_transaction`` and
skip ``token_transfers`` entirely, because native-asset token_transfers
responses are always empty.

But the hop-classifier can legitimately select a *token* tx on that same
native-asset hop (e.g. a victim who was originally tracked as "stolen
ETH" touches a USDT transfer two hops downstream). For that token tx:

  - ``get_transaction`` returns ``to = 0xdac17f958d2ee523a2206206994597c13d831ec7``
    (the USDT contract address), because from the native-ETH view the
    value left the sender and entered the contract.
  - The real recipient lives in the Transfer-event log, which only
    ``token_transfers`` surfaces.

Result: the trace dead-ends at the USDT contract instead of the real
recipient, and the frontend renders the terminal node as
"Tether USD (USDT)".

The fix lives in ``BaseTracer._should_skip_token_transfers``: we may only
skip token_transfers when BOTH (a) the trace asset is native AND (b) the
specific tx we're resolving is itself native (``token_id`` is 0 or
missing). If the tx is a token transfer we must call token_transfers
regardless of the trace-level asset.
"""
from __future__ import annotations

from agent.base_tracer import BaseTracer


class _StubTracer(BaseTracer):
    """BaseTracer has abstract ``execute_tool`` / ``_get_client`` hooks
    that are irrelevant to the pure decision logic under test. Stub them
    out so we can instantiate the class."""

    def _get_client(self):  # pragma: no cover - not exercised
        raise NotImplementedError

    async def execute_tool(self, tool_name, arguments):  # pragma: no cover
        raise NotImplementedError

    def __init__(self):
        pass


def _t() -> BaseTracer:
    return _StubTracer()


class TestShouldSkipTokenTransfers:
    # --- original behavior: native trace + native tx → skip is fine ---

    def test_native_eth_trace_with_native_tx_skips(self):
        """Stolen ETH, tx is a plain ETH transfer → token_transfers is
        guaranteed empty, skip it."""
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint="ETH",
            address_hint="0xabc",
            tx_token_id=0,
        ) is True

    def test_native_eth_trace_with_unknown_token_id_skips(self):
        """token_id not yet known (e.g. first hop, all_txs_map empty) →
        fall back to the native path rather than add a round-trip on
        every hop."""
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint="ETH",
            address_hint="0xabc",
            tx_token_id=None,
        ) is True

    def test_native_trx_trace_with_native_tx_skips(self):
        assert _t()._should_skip_token_transfers(
            chain="trx",
            asset_hint="TRX",
            address_hint="TRx1",
            tx_token_id=0,
        ) is True

    # --- new behavior: native trace + *token* tx must NOT skip ---

    def test_native_eth_trace_with_usdt_tx_does_not_skip(self):
        """Regression: stolen-ETH trace, classifier picks a USDT tx.
        We MUST call token_transfers or we end up at USDT contract."""
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint="ETH",
            address_hint="0x147ac0b39675769e55a0f0e7fdd3641b47963661",
            tx_token_id=94252,  # USDT on eth
        ) is False

    def test_native_trx_trace_with_trc20_tx_does_not_skip(self):
        """Same bug shape on TRON (TRC-20 USDT)."""
        assert _t()._should_skip_token_transfers(
            chain="trx",
            asset_hint="TRX",
            address_hint="TRxSome",
            tx_token_id=12345,
        ) is False

    def test_native_eth_trace_with_weth_tx_does_not_skip(self):
        """Any non-zero token_id on a native-asset hop triggers the
        token_transfers path."""
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint="ETH",
            address_hint="0xabc",
            tx_token_id=1,
        ) is False

    # --- non-native traces never skip ---

    def test_token_asset_trace_never_skips(self):
        """Stolen USDT trace → always use token_transfers."""
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint="USDT",
            address_hint="0xabc",
            tx_token_id=94252,
        ) is False

    def test_btc_trace_does_not_skip(self):
        """BTC/BCH/LTC aren't in the native-asset map (they're UTXO),
        so the skip path never activates."""
        assert _t()._should_skip_token_transfers(
            chain="btc",
            asset_hint="BTC",
            address_hint="bc1q...",
            tx_token_id=0,
        ) is False

    # --- missing address_hint disables skipping ---

    def test_missing_address_hint_does_not_skip(self):
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint="ETH",
            address_hint=None,
            tx_token_id=0,
        ) is False

    def test_missing_asset_hint_does_not_skip(self):
        assert _t()._should_skip_token_transfers(
            chain="eth",
            asset_hint=None,
            address_hint="0xabc",
            tx_token_id=0,
        ) is False
