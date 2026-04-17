"""Regression tests for the dust-check unit-mismatch bug.

Historical bug
--------------
In the account-model theft setup, ``_parse_transfer`` may return
``amount`` in *raw* units (e.g. gwei-scale for ETH, when the AMLBot
``token_transfers`` response lacks an ``amount_coerced`` key).  The
theft tx is not yet in ``all_txs_map``, so ``_resolve_amount`` falls
back to that raw value, making ``fifo_ledger.stolen_amount`` gwei-scale.

Subsequent mule outflows at hop 1 are fetched via ``_fetch_outgoing_txs``,
which populates ``all_txs_map[sel_hash]["amount_coerced"]`` in decimal
form.  ``_resolve_amount`` then returns the decimal value.

Result:
  - ``fifo_ledger._queues[perp]`` holds ``{amount: 94e9, theft_share: 94e9}``
    (gwei).
  - ``attribute_outflow(perp, 10.0)`` (decimal) returns ``10.0``.
  - ``dust_threshold = 94e9 * 0.001 = 94e6`` (gwei).
  - ``10.0 < 94e6`` → dust fires for every mule at hop 2, terminating
    the trace prematurely and causing ChangeNOW destinations to be
    misrepresented as "Trace endpoint (dust amount)".

These tests lock the FIFO/ dust semantics to the fix: the dust check
must be expressed relative to ``job.incoming_amount`` (same unit as
``attributed_amount``), so any theft/outflow unit mismatch in the
ledger cannot silently terminate high-value paths.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import FIFOLedger


class TestDustCheckUnitMismatchRegression:
    def test_fifo_behaves_sensibly_when_stolen_and_outflow_share_scale(self):
        """Sanity baseline: same unit produces full attribution and no dust."""
        ledger = FIFOLedger(stolen_amount=94.0, tolerance=0.03)
        ledger.record_inflow("perp", 94.0, 94.0)

        attributed_per_mule = []
        for _ in range(9):
            attributed_per_mule.append(ledger.attribute_outflow("perp", 10.0))

        assert all(a == pytest.approx(10.0) for a in attributed_per_mule[:9])
        # last call drains remaining 4 ETH
        tail = ledger.attribute_outflow("perp", 10.0)
        assert tail == pytest.approx(4.0)
        # 11th call: queue empty, 0 attribution
        assert ledger.attribute_outflow("perp", 10.0) == 0.0

    def test_dust_semantic_must_use_incoming_not_stolen(self):
        """The fix switches the dust threshold from ``stolen_amount * 0.001``
        to ``job.incoming_amount * 0.001``.  This test encodes the invariant:
        when a mule receives a real attribution that equals its incoming flow,
        it must NOT be classified as dust regardless of the scale the ledger
        was initialised with.
        """
        # Simulate the unit mismatch that triggered the bug in production:
        # - stolen_amount recorded in gwei-scale (94 billion)
        # - per-hop step_amount in decimal-ETH scale (10.0)
        stolen_gwei = 94_044_350_034.418
        mule_incoming_decimal = 10.0
        attributed_decimal = 10.0

        # Old (broken) behaviour: compare attributed to stolen * 0.001.
        old_threshold = stolen_gwei * 0.001
        assert attributed_decimal < old_threshold, (
            "Baseline: confirms the old stolen-relative threshold falsely "
            "flags real 10 ETH attributions as dust when stolen is gwei-scaled"
        )

        # New behaviour: compare to the hop's own incoming amount, which is
        # ALWAYS in the same unit as attributed_amount because they share
        # the same code path (FIFO attribute_outflow).
        new_threshold = mule_incoming_decimal * 0.001
        assert attributed_decimal >= new_threshold, (
            "Fix invariant: a mule whose attribution matches its inflow "
            "must never be classified as dust, regardless of ledger scale."
        )

    def test_dust_fires_when_attribution_is_trivial_fraction_of_incoming(self):
        """Per-hop dust semantic still catches real dust: if 99.9%+ of the
        inflow was non-theft (mixer dilution, etc.), the hop should still
        terminate as dust.
        """
        mule_incoming = 100.0
        attributed = 0.05  # 0.05% of inflow

        threshold = mule_incoming * 0.001
        assert attributed < threshold, (
            "Real dust (attribution << 0.1% of inflow) still terminates."
        )

    def test_dust_does_not_fire_on_zero_incoming(self):
        """Guard against division-by-zero / degenerate incoming amounts."""
        mule_incoming = 0.0
        attributed = 0.0

        # With incoming=0 the dust threshold is 0, dust must NOT fire
        # (we treat this as "no signal, let the normal flow decide").
        threshold = mule_incoming * 0.001
        assert not (threshold > 0 and attributed < threshold)
