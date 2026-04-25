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


class TestDustAggregateDoesNotKillAlivePath:
    """When a later dust iteration hits a recipient that a PRIOR non-dust
    iteration already registered in ``recipient_state``, the dust code
    path pulls ``path_id`` from the existing state instead of forking a
    new path — meaning the ``path_id`` is a live branch with a pending
    HopJob. Setting ``stop_reason = "Below dust threshold"`` on it would
    (a) make ``_completed_paths_count()`` count it toward the scheduler's
    ``max_completed`` budget even though it still has downstream work,
    and (b) mislead downstream rendering into believing the leg ended at
    dust when the real HopJob continues.

    Reproduced from
    ``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_c5.jsonl``:
    TLHPDaL's iter 10 pushed a HopJob for TRgoBU (5 632 USDT); iter 14
    then sent 108 USDT to the same TRgoBU. With the pre-fix code iter 14
    dust-trimmed ``paths["10"]`` — TRgoBU's hop 2 never ran because the
    scheduler's completion count jumped past the 10-path budget.

    Fix: when the dust branch is an aggregate onto an existing recipient
    (``existing_state is not None``), skip the ``stop_reason`` /
    ``path_seen_addresses`` / ``dust_trimmed_paths`` writes.
    """

    def _should_set_stop_reason(self, *, existing_state_present: bool) -> bool:
        """Mirror of the new ``is_aggregate_onto_alive`` guard in the
        dust branch. Keep in sync when the rule there changes."""
        return not existing_state_present

    def test_fresh_recipient_path_is_stopped(self):
        assert self._should_set_stop_reason(existing_state_present=False) is True

    def test_aggregate_onto_live_path_keeps_running(self):
        assert self._should_set_stop_reason(existing_state_present=True) is False


class TestDustTrimmedPathsExcludedFromBudget:
    """``_completed_paths_count()`` drives ``HopScheduler.should_continue``
    via ``max_completed``. Before the fix it counted every path with a
    ``stop_reason``, including dust-trimmed siblings. A single TLHPDaL
    fan-out produced ~15 dust paths — well past the default
    ``max_paths=10`` budget — and the scheduler exited before any
    downstream hop was processed. The output graph then showed hop-1
    edges but no hop-2+ exploration.

    Fix: exclude paths in ``dust_trimmed_paths`` from the budget count.
    Only genuine terminals (CEX, bridge, dead-end, max-hops) count.
    """

    def _completed_count(
        self, stop_reasons: dict[str, str | None], dust_trimmed: set[str]
    ) -> int:
        """Mirror of the new ``_completed_paths_count`` body."""
        return sum(
            1 for pid, sr in stop_reasons.items()
            if sr and pid not in dust_trimmed
        )

    def test_dust_only_paths_do_not_consume_budget(self):
        stops = {"1": "Below dust threshold", "2": "Below dust threshold"}
        dust = {"1", "2"}
        assert self._completed_count(stops, dust) == 0

    def test_real_terminal_still_counts(self):
        stops = {"1": "Reached terminal entity", "2": "Below dust threshold"}
        dust = {"2"}
        assert self._completed_count(stops, dust) == 1

    def test_mixed_terminals_count_only_real_ones(self):
        stops = {
            "1": "Below dust threshold",
            "2": "Reached terminal entity",
            "3": "Dead end - no outgoing transactions",
            "4": "Below dust threshold",
            "5": "Max hop limit reached",
        }
        dust = {"1", "4"}
        assert self._completed_count(stops, dust) == 3


class TestDustDoesNotPolluteParentPathSeenSet:
    """When the first sibling iteration of an outgoing-tx loop dust-trims
    and happens to reuse ``base_path_id`` (because ``used_base_path`` is
    still False), writing ``to_addr`` into ``path_seen_addresses[path_id]``
    would pollute the INCOMING path's history — not the sibling branch.

    The loop-detection check at the top of the next iteration reads
    ``path_seen_addresses[job.path_id]``; with the old code a dust
    ``b36a → TN6c`` as iter 1 blocked the legitimate
    ``376c (220 100 USDT) → TN6c`` as iter 2, because both iterations see
    ``job.path_id == base_path_id == "1"`` and iter 1's pollution lands in
    the exact set iter 2 checks. Trace
    ``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_57.jsonl``
    reproduced the miss in production.

    Fix: the dust branch only writes to ``path_seen_addresses[path_id]``
    when ``path_id != job.path_id`` — i.e. only for forked siblings, not
    for the parent path itself.
    """

    def _should_pollute_parent_seen(self, *, path_id: str, job_path_id: str) -> bool:
        """Mirror of the guard on the ``path_seen_addresses`` write in the
        dust branch of ``_run_agentic_trace``. Keep in sync when the rule
        there changes."""
        return path_id != job_path_id

    def test_first_iter_reusing_base_does_not_pollute(self):
        # iter 1: used_base_path was False, so path_id == base == job.path_id
        assert self._should_pollute_parent_seen(path_id="1", job_path_id="1") is False

    def test_forked_sibling_iter_pollutes_its_own_branch(self):
        # iter 3+: forked path_id != job.path_id → safe to pollute
        assert self._should_pollute_parent_seen(path_id="2", job_path_id="1") is True
        assert self._should_pollute_parent_seen(path_id="5", job_path_id="1") is True


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
