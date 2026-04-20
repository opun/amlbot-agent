"""Regression guard for ETH/EVM gwei→display normalization.

The 0xe0c92b55 trace stopped after hop 1 because ``_normalize_amount``
was returning ETH values in gwei (base-unit, 140_570_190 for 0.14 ETH),
while downstream code mixed those with display-unit values from
``amount_coerced`` (already 0.14). The dust guard then compared
0.0077 < 1_405_701 (gwei × 1%) and treated every branch as dust.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import BaseTracer


class _T(BaseTracer):
    async def execute_tool(self, tool_name, arguments):
        raise NotImplementedError

    def _get_client(self):
        raise NotImplementedError


@pytest.fixture
def tracer():
    return _T()


class TestEthNormalization:
    """``_normalize_amount`` must bring ETH base-unit values down to
    display before they enter the tracer loop — mirroring what already
    happens for BTC (sat→BTC) and USDT (micro→USDT)."""

    def test_eth_gwei_is_divided(self, tracer):
        # 0.14057019 ETH ⇒ 140_570_190 gwei. After normalize: display.
        assert tracer._normalize_amount(140_570_190, "eth", "ETH") == pytest.approx(0.14057019)

    def test_eth_display_below_threshold_untouched(self, tracer):
        # 0.14 ETH (display) ⇒ 0.14. The 1e6 threshold skips small
        # display values so we don't double-normalize them.
        assert tracer._normalize_amount(0.14, "eth", "ETH") == pytest.approx(0.14)

    def test_eth_zero_ok(self, tracer):
        assert tracer._normalize_amount(0, "eth", "ETH") == 0.0

    def test_bnb_matic_same_rule(self, tracer):
        # Per currencies.json: BSC BNB native = unit 9 (gwei); MATIC
        # native = unit 18 (wei). The hand-written hardcode rule divides
        # every EVM chain by 10^9 — that was a bug only accidentally
        # correct for ETH/BSC. The registry gets each chain right.
        assert tracer._normalize_amount(1_500_000_000, "bsc", "BNB") == pytest.approx(1.5)
        # 2 MATIC in wei = 2e18. Divide by 10^18 = 2.0.
        assert tracer._normalize_amount(2 * 10**18, "matic", "MATIC") == pytest.approx(2.0)
        # 0.5 ETH on Arbitrum (18 decimals): 5e17 wei.
        assert tracer._normalize_amount(5 * 10**17, "arb", "ETH") == pytest.approx(0.5)

    def test_usdt_on_eth_still_uses_6_decimals(self, tracer):
        # USDT-on-ETH is still a 6-decimal token; chain-level EVM rule
        # must not swallow the asset-level rule.
        assert tracer._normalize_amount(60_000_000_000, "eth", "USDT") == pytest.approx(60_000.0)

    def test_btc_still_uses_8_decimals(self, tracer):
        assert tracer._normalize_amount(36_500_000, "btc", "BTC") == pytest.approx(0.365)

    def test_trx_native_still_uses_6_decimals(self, tracer):
        assert tracer._normalize_amount(60_000_000_000, "trx", "USDT") == pytest.approx(60_000.0)

    def test_dust_comparison_works_after_fix(self, tracer):
        """Simulates the 0xe0c92b55 case:
        stolen=0.14 ETH display, hop sends 0.0077 → 5.5% = not dust."""
        stolen = tracer._normalize_amount(140_570_190, "eth", "ETH")
        hop_amount = tracer._normalize_amount(0.0077, "eth", "ETH")  # already display
        # 0.0077 / 0.14057 = 5.48% → above 1% dust threshold.
        assert hop_amount >= stolen * 0.01
        # Before the fix, stolen stayed 140_570_190 and the comparison
        # flipped: 0.0077 < 1_405_701.9 → every hop flagged dust.
