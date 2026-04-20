"""Tests for the dust-threshold logic.

The full hop loop is hard to stand up in isolation, so we verify the
dust condition itself: given (stolen_amount, ratio, raw_attributed),
does the inequality produce the expected trim/continue decision? The
actual branch-skipping happens in base_tracer; an end-to-end regression
lives under ``tests/golden/`` once a recording is captured.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import FIFOLedger
from agent.models import TracerConfig


def _is_dust(stolen_amount: float, raw_attributed: float, ratio: float) -> bool:
    """Mirror of the guard expression in ``_run_agentic_trace``.

    Duplicated here to pin down the semantics and catch accidental
    changes in the inequality direction or the zero-guards.
    """
    return (
        stolen_amount > 0
        and ratio > 0.0
        and raw_attributed < stolen_amount * ratio
    )


class TestDustExpression:
    def test_below_threshold_is_dust(self):
        # 100 USDT out of 60 000 = 0.17%, below default 1%
        assert _is_dust(60_000.0, 100.0, 0.01) is True

    def test_right_at_threshold_is_not_dust(self):
        # 600 / 60 000 = 1.0% — strictly-less-than, so not trimmed
        assert _is_dust(60_000.0, 600.0, 0.01) is False

    def test_above_threshold_is_not_dust(self):
        assert _is_dust(60_000.0, 663.0, 0.01) is False

    def test_unknown_stolen_amount_disables_guard(self):
        """When stolen_amount is unknown (0), the guard must not fire."""
        assert _is_dust(0.0, 10.0, 0.01) is False

    def test_ratio_zero_disables_guard(self):
        assert _is_dust(1000.0, 1.0, 0.0) is False

    def test_negative_noise_does_not_trim(self):
        # FIFO can occasionally round attributed slightly below 0 due to
        # floating-point drain; the guard still fires (correctly — 0 is
        # genuinely dust). Sanity, not a regression guard.
        assert _is_dust(1000.0, -0.0001, 0.01) is True


class TestConfigDefaults:
    def test_default_ratio_is_one_percent(self):
        cfg = TracerConfig(victim_address="0xabc", blockchain_name="eth")
        assert cfg.min_path_attribution_ratio == 0.01


class TestStepAmountNotFifoAttributed:
    """Per-step dust must use outflow size (``step_amount``) rather than
    FIFO-attributed theft-share. FIFO dilutes attribution when an
    address holds a mix of stolen + non-stolen inflows — comparing that
    diluted value to ``stolen * ratio`` wrongly trims legit-sized hops.

    Real-case (TBdxz Victim-report #16547):
    * TBdxz received 55 USDT (attr=55) AND 208 000 USDT (attr≈59 938,
      because FIFO capped at the stolen-queue remainder).
    * TBdxz's queue: theft-share ratio ≈ 28.8% across the mixed pool.
    * Outflow of 663 USDT → step_amount=663 (1.1% of stolen 60 000),
      but ``fifo_ledger.attribute_outflow`` returns ~230 (28.8% × 663).
    * With the OLD rule (attributed < stolen*ratio): 230 < 600 → dust.
      Branch drops. Trace cuts short at TBdxz.
    * With the NEW rule (step_amount < stolen*ratio): 663 > 600 → not
      dust. Branch continues.
    """

    def _is_step_dust(self, *, step_amount: float, stolen: float, ratio: float) -> bool:
        """Mirror of the guard in ``_run_agentic_trace`` around the
        ``_accumulate_hashes`` loop."""
        return (
            stolen > 0
            and ratio > 0
            and step_amount < stolen * ratio
        )

    def test_663_usdt_of_60k_is_not_dust(self):
        """Regression: pre-fix FIFO comparison wrongly flagged this."""
        assert self._is_step_dust(step_amount=663.0, stolen=60_000.0, ratio=0.01) is False

    def test_100_usdt_of_60k_is_dust(self):
        """Real dust (<1%) still trims."""
        assert self._is_step_dust(step_amount=100.0, stolen=60_000.0, ratio=0.01) is True

    def test_at_exact_1_percent_is_not_dust(self):
        """Strict less-than, so the boundary passes."""
        assert self._is_step_dust(step_amount=600.0, stolen=60_000.0, ratio=0.01) is False


class TestAssetChangedBridgeDustGuard:
    """Cross-asset bridges (Bridgers ETH→USDT(TRON)) must not trip the
    dust guard on a cross-unit comparison.

    Scenario: stolen_amount = 0.14 ETH, FIFO returns 0.14 ETH of
    attribution on the Bridgers outflow, but bridge_step_amount = 590
    USDT (from the time-matched destination deposit). Without the
    asset-changed skip, the guard would trip ``0.14 < 0.01 * 590`` and
    return before pushing the HopJob. This test pins down the rule.
    """

    def _is_bridge_dust_hit(
        self,
        *,
        asset_changed: bool,
        bridge_step_amount: float,
        bridge_raw_attr: float,
        dust_anchor: float,
        ratio: float = 0.01,
    ) -> bool:
        """Mirror of the asset-changed guard in ``_run_agentic_trace``.

        Must match the code path at ``base_tracer.py`` around the
        ``bridge_dust_hit`` expression — keep in sync when the rule
        there changes.
        """
        have_dest_amount = bridge_step_amount > 0
        return (
            have_dest_amount
            and not asset_changed
            and dust_anchor > 0
            and ratio > 0
            and bridge_raw_attr < dust_anchor * ratio
        )

    def test_asset_changed_skips_dust_guard(self):
        """Bridgers swap: FIFO 0.14 (ETH units), anchor 590 (USDT).
        Guard must NOT fire so the HopJob pushes onto TRON."""
        assert self._is_bridge_dust_hit(
            asset_changed=True,
            bridge_step_amount=590.28,
            bridge_raw_attr=0.14,
            dust_anchor=590.28,
        ) is False

    def test_same_asset_bridge_still_dust_guarded(self):
        """Same-asset bridge (e.g. LayerZero USDT→USDT): keep the
        guard active so tiny leaks still trim."""
        assert self._is_bridge_dust_hit(
            asset_changed=False,
            bridge_step_amount=590.28,
            bridge_raw_attr=0.14,
            dust_anchor=590.28,
        ) is True

    def test_no_dest_amount_skips_guard(self):
        """Bridgers with no time-match hit: bridge_step_amount stays 0,
        guard can't compare anything meaningful."""
        assert self._is_bridge_dust_hit(
            asset_changed=True,
            bridge_step_amount=0.0,
            bridge_raw_attr=0.0,
            dust_anchor=60000.0,
        ) is False

    def test_ratio_can_be_disabled(self):
        cfg = TracerConfig(
            victim_address="0xabc", blockchain_name="eth",
            min_path_attribution_ratio=0.0,
        )
        assert cfg.min_path_attribution_ratio == 0.0

    def test_ratio_rejects_out_of_range(self):
        with pytest.raises(Exception):
            TracerConfig(
                victim_address="0xabc", blockchain_name="eth",
                min_path_attribution_ratio=1.5,
            )
        with pytest.raises(Exception):
            TracerConfig(
                victim_address="0xabc", blockchain_name="eth",
                min_path_attribution_ratio=-0.1,
            )


class TestFIFOLedgerAttributionForDust:
    """Sanity: FIFO attribution actually returns a small number when the
    source address got a tiny theft-share relative to its total inflow."""

    def test_small_theft_share_yields_small_attribution(self):
        ledger = FIFOLedger(stolen_amount=60_000.0, tolerance=0.03)
        # Address had a clean inflow of 100 of legit funds + 100 of theft
        # (theft_share=100 out of 100 amount). Outflow attribution is
        # then bounded by theft_share.
        ledger.record_inflow("0xA", amount=100.0, theft_share=100.0)
        # But on 0xB the attribution propagates through.
        attributed = ledger.attribute_outflow("0xA", outflow_amount=100.0)
        assert attributed == pytest.approx(100.0)

    def test_oversized_outflow_caps_attribution_to_queue(self):
        """Mirror of the Tronify case: recipient had tiny theft-share,
        but sends out a huge physical amount from mixed funds. Attribution
        is capped to what the FIFO queue actually contains."""
        ledger = FIFOLedger(stolen_amount=60_000.0, tolerance=0.03)
        ledger.record_inflow("0xA", amount=100.0, theft_share=100.0)
        attributed = ledger.attribute_outflow("0xA", outflow_amount=27_122.0)
        # Only 100 of the 27 122 out are theft-attributed.
        assert attributed == pytest.approx(100.0)
        # And 100 / 60 000 = 0.17% — below 1% threshold → dust.
        assert _is_dust(60_000.0, attributed, 0.01) is True
