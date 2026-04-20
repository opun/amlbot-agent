"""Currency registry + _normalize_amount correctness.

Registry is the authoritative source; these tests pin down the decimals
the tracer relies on for a wide range of chains and tokens. Adding a new
chain/token to ``currencies.json`` should automatically fix the
corresponding hop amounts without touching tracer code.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import BaseTracer
from agent.currency_registry import (
    get_registry,
    normalize_external_chain,
    parse_thorchain_token_prefix,
)


class _T(BaseTracer):
    async def execute_tool(self, tool_name, arguments):
        raise NotImplementedError

    def _get_client(self):
        raise NotImplementedError


@pytest.fixture
def tracer():
    return _T()


class TestRegistryLoad:
    def test_registry_has_entries(self):
        reg = get_registry()
        assert len(reg) > 1000, "currencies snapshot looks empty"

    def test_eth_native(self):
        rec = get_registry().lookup("eth", 0)
        assert rec is not None
        assert rec.unit == 9
        assert rec.symbol == "ETH"

    def test_matic_native_is_18_decimals(self):
        rec = get_registry().lookup("matic", 0)
        assert rec is not None
        assert rec.unit == 18, "MATIC native should be 18 decimals per SAILS registry"

    def test_btc_native(self):
        rec = get_registry().lookup("btc", 0)
        assert rec is not None
        assert rec.unit == 8

    def test_trx_usdt(self):
        rec = get_registry().lookup_by_symbol("trx", "USDT")
        assert rec is not None
        assert rec.unit == 6


class TestNormalizeAmountPerChain:
    """One test per popular native coin + its decimals from the registry."""

    def test_eth_9_dec(self, tracer):
        # 0.14057019 ETH = 140_570_190 in gwei-scale SAILS returns.
        assert tracer._normalize_amount(140_570_190, "eth", "ETH") == pytest.approx(0.14057019)

    def test_matic_18_dec(self, tracer):
        # 2 MATIC in wei
        assert tracer._normalize_amount(2 * 10**18, "matic", "MATIC") == pytest.approx(2.0)

    def test_arb_eth_18_dec(self, tracer):
        # 0.5 ETH on Arbitrum (registry says 18 decimals)
        assert tracer._normalize_amount(5 * 10**17, "arb", "ETH") == pytest.approx(0.5)

    def test_op_eth_18_dec(self, tracer):
        assert tracer._normalize_amount(3 * 10**17, "op", "ETH") == pytest.approx(0.3)

    def test_base_eth_18_dec(self, tracer):
        assert tracer._normalize_amount(1 * 10**18, "base", "ETH") == pytest.approx(1.0)

    def test_avax_18_dec(self, tracer):
        assert tracer._normalize_amount(25 * 10**17, "avax", "AVAX") == pytest.approx(2.5)

    def test_ftm_18_dec(self, tracer):
        assert tracer._normalize_amount(10 * 10**18, "ftm", "FTM") == pytest.approx(10.0)

    def test_bsc_bnb_9_dec(self, tracer):
        # BSC native BNB recorded as unit=9 in the snapshot
        assert tracer._normalize_amount(1_500_000_000, "bsc", "BNB") == pytest.approx(1.5)

    def test_btc_8_dec(self, tracer):
        assert tracer._normalize_amount(36_500_000, "btc", "BTC") == pytest.approx(0.365)

    def test_ltc_8_dec(self, tracer):
        assert tracer._normalize_amount(50_000_000, "ltc", "LTC") == pytest.approx(0.5)

    def test_doge_8_dec(self, tracer):
        assert tracer._normalize_amount(100_000_000, "doge", "DOGE") == pytest.approx(1.0)

    def test_trx_6_dec(self, tracer):
        assert tracer._normalize_amount(60_000_000_000, "trx", "USDT") == pytest.approx(60_000.0)


class TestNormalizeErc20Tokens:
    """ERC20 tokens beyond the short stablecoin list — the registry must
    handle these without the tracer having to know each token."""

    def test_weth_18_dec(self, tracer):
        assert tracer._normalize_amount(1 * 10**18, "eth", "WETH") == pytest.approx(1.0)

    def test_wbtc_8_dec(self, tracer):
        # WBTC is 8 decimals — notably unlike every other ERC20.
        assert tracer._normalize_amount(50_000_000, "eth", "WBTC") == pytest.approx(0.5)

    def test_dai_18_dec(self, tracer):
        assert tracer._normalize_amount(100 * 10**18, "eth", "DAI") == pytest.approx(100.0)

    def test_link_18_dec(self, tracer):
        assert tracer._normalize_amount(25 * 10**18, "eth", "LINK") == pytest.approx(25.0)

    def test_shib_18_dec(self, tracer):
        # 100k SHIB in wei-scale — SHIB is still 18 decimals.
        assert tracer._normalize_amount(100_000 * 10**18, "eth", "SHIB") == pytest.approx(100_000.0)


class TestNormalizeSafeguards:
    """Values that look already-display must pass through unchanged."""

    def test_display_eth_untouched(self, tracer):
        assert tracer._normalize_amount(0.14, "eth", "ETH") == pytest.approx(0.14)

    def test_display_matic_untouched(self, tracer):
        # 2.0 MATIC — below half-unit threshold (10^9), so no scaling.
        assert tracer._normalize_amount(2.0, "matic", "MATIC") == pytest.approx(2.0)

    def test_display_usdt_below_threshold(self, tracer):
        # 500 USDT in display-form. Safeguard floor at 1e6 for unit=6, so
        # 500 stays unchanged.
        assert tracer._normalize_amount(500.0, "eth", "USDT") == pytest.approx(500.0)

    def test_display_usdt_large_untouched(self, tracer):
        """Regression: 60k display USDT must NOT be double-normalized.

        Before the threshold fix the safeguard sat at 10^(unit//2)=10^3
        for USDT, so any display value ≥ 1000 got divided by 10^6 a
        second time (60000 → 0.06). That caused coverage-check failure
        in ``_fetch_outgoing_txs`` and wrongly tripped the dust guard,
        cutting off the trace after hop 1 on USDT-heavy TRON cases.
        """
        assert tracer._normalize_amount(60_000.0, "trx", "USDT") == pytest.approx(60_000.0)
        assert tracer._normalize_amount(60_000.0, "eth", "USDT") == pytest.approx(60_000.0)
        assert tracer._normalize_amount(999_999.0, "eth", "USDT") == pytest.approx(999_999.0)

    def test_raw_usdt_base_units_still_scales(self, tracer):
        """Counterpart to the above: values at 1e6+ ARE base-units and
        should still divide — the floor is 1e6 so 1e6 exactly trips."""
        # 60000 USDT in base-units (60000 * 10^6)
        assert tracer._normalize_amount(60_000_000_000.0, "trx", "USDT") == pytest.approx(60_000.0)
        # 1 USDT in base-units
        assert tracer._normalize_amount(1_000_000.0, "trx", "USDT") == pytest.approx(1.0)

    def test_zero_stays_zero(self, tracer):
        assert tracer._normalize_amount(0, "eth", "ETH") == 0.0

    def test_unknown_chain_fallthrough(self, tracer):
        # Unknown chain with known-symbol asset: tracer still works.
        # With no registry hit, falls through to hardcode rules which
        # don't match, then returns raw val unchanged.
        assert tracer._normalize_amount(12345.67, "fake_chain_xyz", "XYZ") == pytest.approx(12345.67)


class TestExternalChainMapping:
    def test_thorchain_style_names(self):
        assert normalize_external_chain("bitcoin") == "btc"
        assert normalize_external_chain("BITCOIN") == "btc"
        assert normalize_external_chain("ethereum") == "eth"
        assert normalize_external_chain("Polygon") == "matic"
        assert normalize_external_chain("POL") == "matic"
        assert normalize_external_chain("BNB") == "bsc"

    def test_unknown_name_returns_none(self):
        assert normalize_external_chain("Atlantis") is None
        assert normalize_external_chain("") is None
        assert normalize_external_chain(None) is None

    def test_thorchain_token_prefix(self):
        assert parse_thorchain_token_prefix("eth.eth") == "eth"
        assert parse_thorchain_token_prefix("btc.btc") == "btc"
        assert parse_thorchain_token_prefix("bsc.bnb") == "bsc"
        assert parse_thorchain_token_prefix("thor.rune") is None  # not in map
        assert parse_thorchain_token_prefix("nocolon") is None
        assert parse_thorchain_token_prefix(None) is None
