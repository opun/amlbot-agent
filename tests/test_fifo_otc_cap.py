"""Tests for FIFO ledger, OTC-like classification, and global cap logic."""
import pytest

from agent.base_tracer import FIFOLedger, BaseTracer
from agent.models import TracerConfig, TraceStats


class TestFIFOLedger:
    def test_basic_fifo_single_inflow_single_outflow(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        ledger.record_inflow("addr_a", 100_000, 100_000)
        attr = ledger.attribute_outflow("addr_a", 100_000)
        assert attr == pytest.approx(100_000)

    def test_fifo_partial_outflow(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        ledger.record_inflow("addr_a", 100_000, 100_000)
        attr = ledger.attribute_outflow("addr_a", 60_000)
        assert attr == pytest.approx(60_000)
        attr2 = ledger.attribute_outflow("addr_a", 40_000)
        assert attr2 == pytest.approx(40_000)

    def test_fifo_mixed_funds_proportional(self):
        """When an address has mixed inflows (theft + non-theft),
        outflows should be attributed proportionally using FIFO."""
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        ledger.record_inflow("addr_a", 80_000, 80_000)
        ledger.record_inflow("addr_a", 120_000, 0)
        attr = ledger.attribute_outflow("addr_a", 80_000)
        assert attr == pytest.approx(80_000)
        attr2 = ledger.attribute_outflow("addr_a", 60_000)
        assert attr2 == pytest.approx(0)

    def test_fifo_mixed_funds_partial_theft_in_first_entry(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        ledger.record_inflow("addr_a", 100_000, 50_000)  # 50% theft
        attr = ledger.attribute_outflow("addr_a", 60_000)
        assert attr == pytest.approx(30_000)  # 50% of 60k

    def test_fifo_multiple_entries_fifo_order(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        ledger.record_inflow("addr_a", 40_000, 40_000)
        ledger.record_inflow("addr_a", 60_000, 20_000)
        attr = ledger.attribute_outflow("addr_a", 50_000)
        # First 40k from entry 1 (100% theft), then 10k from entry 2 (20/60 = 33.3%)
        expected = 40_000 + 10_000 * (20_000 / 60_000)
        assert attr == pytest.approx(expected)

    def test_outflow_from_empty_address_returns_zero(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        attr = ledger.attribute_outflow("unknown_addr", 50_000)
        assert attr == 0.0


class TestGlobalCap:
    def test_claim_terminal_basic(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        clamped = ledger.claim_terminal(100_000)
        assert clamped == pytest.approx(100_000)
        assert not ledger.cap_exceeded

    def test_claim_terminal_exceeded_clips(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        c1 = ledger.claim_terminal(100_000)
        assert c1 == pytest.approx(100_000)
        c2 = ledger.claim_terminal(5_000)
        assert c2 == pytest.approx(3_000)
        assert ledger.cap_exceeded

    def test_cap_with_zero_stolen_amount_no_limit(self):
        ledger = FIFOLedger(stolen_amount=0, tolerance=0.03)
        c1 = ledger.claim_terminal(1_000_000)
        assert c1 == pytest.approx(1_000_000)
        assert not ledger.cap_exceeded

    def test_cap_tolerance_respected(self):
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.05)
        assert ledger.cap == pytest.approx(105_000)
        c1 = ledger.claim_terminal(105_000)
        assert c1 == pytest.approx(105_000)
        assert ledger.cap_exceeded

    def test_attribute_outflow_does_not_consume_cap(self):
        """attribute_outflow should NOT affect total_traced (cap is only for terminals)."""
        ledger = FIFOLedger(stolen_amount=100_000, tolerance=0.03)
        ledger.record_inflow("addr_a", 100_000, 100_000)
        attr = ledger.attribute_outflow("addr_a", 100_000)
        assert attr == pytest.approx(100_000)
        assert ledger.total_traced == 0.0
        assert not ledger.cap_exceeded


class TestEndToEndFIFO:
    """Simulates the real case: stolen 3,086 USDT mixed with larger flows."""

    def test_real_case_tcbydh(self):
        """
        Stolen: 3,086 USDT
        Path: victim → perp (3,086) → addr_b (41,053) → addr_c (41,053)
              → OTC (117,100) → addr_e (125,200) → ... → HTX (125,000)

        Without FIFO: 125,000 attributed (40x over-attribution)
        With FIFO: only ~3,086 attributed at HTX terminal
        """
        ledger = FIFOLedger(stolen_amount=3_086, tolerance=0.03)

        # Step 0: theft — victim sends 3,086 to perpetrator
        ledger.record_inflow("perp", 3_086, 3_086)

        # Step 1: perp sends 41,053 (mixed with own funds)
        attr1 = ledger.attribute_outflow("perp", 41_053)
        assert attr1 == pytest.approx(3_086)  # only theft portion
        ledger.record_inflow("addr_b", 41_053, attr1)

        # Step 2: addr_b forwards 41,053
        attr2 = ledger.attribute_outflow("addr_b", 41_053)
        assert attr2 == pytest.approx(3_086)
        ledger.record_inflow("addr_c", 41_053, attr2)

        # Step 3: addr_c sends 117,100 (further mixing)
        attr3 = ledger.attribute_outflow("addr_c", 117_100)
        # 3,086 / 41,053 * 41,053 = 3,086 (all from first entry, fully drained)
        # remaining 117,100 - 41,053 = 76,047 comes from empty queue = 0
        assert attr3 == pytest.approx(3_086)
        ledger.record_inflow("otc", 117_100, attr3)

        # Step 4: OTC sends 125,200
        attr4 = ledger.attribute_outflow("otc", 125_200)
        # 3,086/117,100 * 117,100 = 3,086 from first entry, rest from empty = 0
        assert attr4 == pytest.approx(3_086)
        ledger.record_inflow("addr_e", 125_200, attr4)

        # Steps 5-7: pass-through (same proportion)
        attr5 = ledger.attribute_outflow("addr_e", 125_200)
        assert attr5 == pytest.approx(3_086)
        ledger.record_inflow("addr_f", 125_200, attr5)

        attr6 = ledger.attribute_outflow("addr_f", 125_000)
        # slightly less outflow: 3,086/125,200 * 125,000 ≈ 3,081
        expected6 = 3_086 / 125_200 * 125_000
        assert attr6 == pytest.approx(expected6, rel=0.01)
        ledger.record_inflow("htx", 125_000, attr6)

        # TERMINAL: funds reach HTX — NOW claim against cap
        attr_terminal = ledger.attribute_outflow("htx", 125_000)
        claimed = ledger.claim_terminal(attr_terminal)

        # Cap = 3,086 * 1.03 = 3,178.58
        assert claimed <= 3_086 * 1.03
        assert claimed == pytest.approx(attr_terminal)
        assert ledger.total_traced == pytest.approx(claimed)

    def test_cap_prevents_double_claiming(self):
        """Two paths from same theft, cap prevents over-attribution."""
        ledger = FIFOLedger(stolen_amount=10_000, tolerance=0.03)

        # Path 1: 6,000 theft share reaches CEX-A
        ledger.record_inflow("perp", 10_000, 10_000)
        attr_path1 = ledger.attribute_outflow("perp", 6_000)
        assert attr_path1 == pytest.approx(6_000)
        c1 = ledger.claim_terminal(attr_path1)
        assert c1 == pytest.approx(6_000)

        # Path 2: remaining 4,000 theft share reaches CEX-B
        attr_path2 = ledger.attribute_outflow("perp", 4_000)
        assert attr_path2 == pytest.approx(4_000)
        c2 = ledger.claim_terminal(attr_path2)
        # Cap = 10,300; already claimed 6,000; headroom = 4,300
        assert c2 == pytest.approx(4_000)

        assert ledger.total_traced == pytest.approx(10_000)
        assert not ledger.cap_exceeded

    def test_cap_clips_when_over(self):
        """When terminal claims exceed cap, clip the excess."""
        ledger = FIFOLedger(stolen_amount=5_000, tolerance=0.03)
        # Cap = 5,150

        c1 = ledger.claim_terminal(5_000)
        assert c1 == pytest.approx(5_000)
        c2 = ledger.claim_terminal(500)
        assert c2 == pytest.approx(150)  # only 150 headroom left
        assert ledger.cap_exceeded


class TestOTCClassification:
    def test_otc_like_detected(self):
        result = BaseTracer._classify_otc_like(
            total_in_volume=55_000_000,
            tx_count=1200,
            counterparty_count=300,
            address_age_days=200,
            outbound_distribution={"HTX": 45_000_000, "other": 10_000_000},
        )
        assert result["otc_like"] is True
        assert result["dominant_cex"] == "HTX"
        assert result["dominant_cex_share"] == pytest.approx(45_000_000 / 55_000_000, rel=1e-3)

    def test_not_otc_like_low_volume(self):
        result = BaseTracer._classify_otc_like(
            total_in_volume=10_000,
            tx_count=1200,
            counterparty_count=300,
            address_age_days=200,
            outbound_distribution={"HTX": 8_000, "other": 2_000},
        )
        assert result["otc_like"] is False

    def test_not_otc_like_too_few_txs(self):
        result = BaseTracer._classify_otc_like(
            total_in_volume=1_000_000,
            tx_count=50,
            counterparty_count=10,
            address_age_days=365,
            outbound_distribution={"HTX": 900_000},
        )
        assert result["otc_like"] is False

    def test_not_otc_like_too_young(self):
        result = BaseTracer._classify_otc_like(
            total_in_volume=5_000_000,
            tx_count=500,
            counterparty_count=100,
            address_age_days=30,
            outbound_distribution={"HTX": 4_000_000},
        )
        assert result["otc_like"] is False


class TestTracerConfigDefaults:
    def test_default_thresholds(self):
        config = TracerConfig()
        assert config.cex_single_cluster_threshold == 0.60
        assert config.traced_amount_tolerance == 0.03
        assert config.stolen_amount is None

    def test_custom_thresholds(self):
        config = TracerConfig(
            cex_single_cluster_threshold=0.75,
            traced_amount_tolerance=0.05,
            stolen_amount=500_000.0,
        )
        assert config.cex_single_cluster_threshold == 0.75
        assert config.traced_amount_tolerance == 0.05
        assert config.stolen_amount == 500_000.0


class TestTraceStatsFields:
    def test_new_fields_optional(self):
        stats = TraceStats(initial_amount_estimate=100_000, explored_paths=3)
        assert stats.total_traced_amount is None
        assert stats.stolen_amount is None

    def test_new_fields_populated(self):
        stats = TraceStats(
            initial_amount_estimate=100_000,
            explored_paths=3,
            total_traced_amount=95_000,
            stolen_amount=100_000,
        )
        assert stats.total_traced_amount == 95_000
        assert stats.stolen_amount == 100_000
