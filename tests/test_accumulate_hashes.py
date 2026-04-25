"""Tests for BaseTracer._accumulate_hashes chronological selection.

Regression: the previous implementation exited the accumulation loop as
soon as the remaining gap to ``incoming_amount`` dropped below 1.5%.  In
practice the perpetrator often splits funds as one large CEX-bound cluster
plus a smaller tail transfer to a secondary mule (e.g. 346,150 USDT main +
4,000 USDT side → 350,150 USDT total against a 350,000 USDT inflow).  The
early break silently dropped that 4,000 USDT transaction and cascaded into
missing downstream hops.  These tests lock in the new behaviour: we keep
accumulating until the total covers ``incoming_amount`` or the hard
``max_select`` cap is reached.
"""
from agent.base_tracer import BaseTracer


class _StubTracer(BaseTracer):
    """Minimal concrete subclass so we can instantiate and call instance
    methods under test without pulling in httpx/OpenAI clients."""

    async def execute_tool(self, tool_name, arguments):  # pragma: no cover
        raise NotImplementedError

    def _get_client(self):  # pragma: no cover
        raise NotImplementedError


def _tracer() -> BaseTracer:
    # Skip BaseTracer.__init__ (wires httpx + OpenAI clients we don't need).
    return _StubTracer.__new__(_StubTracer)


def _tx(h: str, amount: float) -> dict:
    return {"hash": h, "amount_coerced": amount}


class TestAccumulateHashes:
    def test_returns_empty_for_empty_txs(self):
        tracer = _tracer()
        assert tracer._accumulate_hashes([], 1000.0, "trx", "USDT") == []

    def test_returns_first_when_no_incoming_amount(self):
        tracer = _tracer()
        txs = [_tx("h1", 100.0), _tx("h2", 200.0)]
        assert tracer._accumulate_hashes(txs, None, "trx", "USDT") == ["h1"]
        assert tracer._accumulate_hashes(txs, 0, "trx", "USDT") == ["h1"]

    def test_covers_incoming_amount_exactly(self):
        tracer = _tracer()
        txs = [_tx("h1", 300.0), _tx("h2", 700.0)]
        assert tracer._accumulate_hashes(txs, 1000.0, "trx", "USDT") == ["h1", "h2"]

    def test_stops_once_accumulated_crosses_incoming(self):
        tracer = _tracer()
        txs = [_tx("h1", 400.0), _tx("h2", 700.0), _tx("h3", 500.0)]
        # 400 + 700 = 1100 ≥ 1000, so h3 should be skipped.
        assert tracer._accumulate_hashes(txs, 1000.0, "trx", "USDT") == ["h1", "h2"]

    def test_includes_tail_transfer_near_coverage(self):
        """Regression: old gap<=1.5% heuristic would stop here at 346,150."""
        tracer = _tracer()
        txs = [
            _tx("main_1", 100_000.0),
            _tx("main_2", 100_000.0),
            _tx("main_3", 100_000.0),
            _tx("main_4", 46_150.0),  # running total: 346,150 (1.1% short)
            _tx("tail",   4_000.0),   # pushes total to 350,150, covering 350k
        ]
        selected = tracer._accumulate_hashes(txs, 350_000.0, "trx", "USDT")
        assert "tail" in selected, (
            "accumulator must keep selecting until coverage is reached; "
            "dropping the 4k tail cascades into missing downstream hops"
        )
        assert selected == ["main_1", "main_2", "main_3", "main_4", "tail"]

    def test_respects_max_select_cap(self):
        tracer = _tracer()
        txs = [_tx(f"h{i}", 1.0) for i in range(50)]
        selected = tracer._accumulate_hashes(
            txs, incoming_amount=1_000_000.0, chain="trx", asset="USDT", max_select=25
        )
        assert len(selected) == 25
        assert selected[0] == "h0"
        assert selected[-1] == "h24"

    def test_skips_items_without_hash(self):
        tracer = _tracer()
        txs = [
            {"amount_coerced": 500.0},  # no hash → skipped
            _tx("h1", 600.0),
            _tx("h2", 500.0),
        ]
        assert tracer._accumulate_hashes(txs, 1000.0, "trx", "USDT") == ["h1", "h2"]

    def test_handles_alternate_hash_and_amount_keys(self):
        tracer = _tracer()
        txs = [
            {"tx_hash": "h1", "amount": 600.0},
            {"tx_hash": "h2", "amount": 500.0},
        ]
        assert tracer._accumulate_hashes(txs, 1000.0, "trx", "USDT") == ["h1", "h2"]

    def test_mixed_funds_disables_early_stop(self):
        """Regression: long-lived mule aggregates pre-existing balance with
        the theft inflow and wires one oversized tx out before staging the
        actual theft across many smaller sends.

        Reproduced from
        ``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_5a.jsonl``:
        TLHPDaLrq3SX7uMoyDYL4KJeW3fdiZz231 received 105 103 USDT then wired
        220 100 USDT out in one shot. The human analyst picked ~9 outflows
        spanning the next few days — greedy "stop at first coverage" misses
        all of them. When any single outflow is ≥ 1.2× incoming, widen the
        selection to ``max_select``.
        """
        tracer = _tracer()
        txs = [
            _tx("dust", 10.0),
            _tx("oversized", 220_100.0),     # alone > 1.2× incoming
            _tx("theft_leg_1", 13_000.0),
            _tx("theft_leg_2", 3_002.0),
            _tx("theft_leg_3", 9_995.0),
            _tx("theft_leg_4", 5_503.0),
        ]
        selected = tracer._accumulate_hashes(txs, 105_103.25, "trx", "USDT")
        assert selected == [
            "dust", "oversized",
            "theft_leg_1", "theft_leg_2", "theft_leg_3", "theft_leg_4",
        ], (
            "after an oversized outflow the accumulator must keep going "
            "to capture the staged theft legs the human analyst picks"
        )

    def test_oversized_first_outflow_still_capped_by_max_select(self):
        """Mixed-funds mode widens the selection but ``max_select`` remains
        the hard ceiling — otherwise a chatty mule could spawn hundreds of
        downstream hops."""
        tracer = _tracer()
        txs = [_tx("oversized", 500_000.0)] + [
            _tx(f"tail_{i}", 100.0) for i in range(40)
        ]
        selected = tracer._accumulate_hashes(
            txs, incoming_amount=100_000.0, chain="trx", asset="USDT", max_select=25
        )
        assert len(selected) == 25
        assert selected[0] == "oversized"
        assert selected[-1] == "tail_23"

    def test_mixed_funds_disabled_beyond_hop_1(self):
        """Hop-2+ must NOT cascade into the mixed-funds fan-out AND
        must skip individual oversized outflows.

        Production case from
        ``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_c2.jsonl``:
        a 60.60k USDT inflow to THWY8p produced downstream siblings of
        121.55k + 300k + 58k at hop 3 because the accumulator re-triggered
        mixed-funds mode. Those large outflows almost certainly carry the
        recipient's OWN balance, not propagated theft — and the analyst
        view doesn't show them.

        Updated behavior: at hop_index>=2 we both (a) disable the
        mixed-funds fan-out and (b) skip individual outflows whose
        amount alone exceeds ``1.2× incoming``, since any tx that big
        structurally cannot carry only the narrow theft share.
        """
        tracer = _tracer()
        txs = [
            _tx("big_first", 121_550.0),    # 2.00× incoming — oversized, skip
            _tx("even_bigger", 300_000.0),  # 4.95× — skip
            _tx("similar", 58_000.0),       # 0.96× — include, covers alone
        ]
        selected = tracer._accumulate_hashes(
            txs, 60_600.0, "trx", "USDT", hop_index=2,
        )
        # Both oversized siblings are skipped; the smaller one covers.
        assert selected == ["similar"]

    def test_mixed_funds_still_active_at_hop_1(self):
        """Guard against an over-eager "disable mixed-funds everywhere"
        regression. The TLHPDaL original case (hop_index=1) REQUIRES the
        fan-out to capture the staged theft legs."""
        tracer = _tracer()
        txs = [
            _tx("dust", 10.0),
            _tx("oversized", 220_100.0),    # > 1.2 × 105_103
            _tx("theft_leg", 13_000.0),
        ]
        selected = tracer._accumulate_hashes(
            txs, 105_103.25, "trx", "USDT", hop_index=1,
        )
        assert selected == ["dust", "oversized", "theft_leg"]

    def test_no_hop_index_keeps_legacy_mixed_funds_behavior(self):
        """Callers that don't know their hop depth (e.g. address-mode
        seed) pass ``hop_index=None``. The mixed-funds path must remain
        active so legacy call sites don't silently regress."""
        tracer = _tracer()
        txs = [
            _tx("dust", 10.0),
            _tx("oversized", 220_100.0),
            _tx("tail", 13_000.0),
        ]
        selected = tracer._accumulate_hashes(
            txs, 105_103.25, "trx", "USDT", hop_index=None,
        )
        assert selected == ["dust", "oversized", "tail"]

    def test_hop_2_skips_oversized_single_outflow(self):
        """Regression: at hop_index>=2 a tx whose amount alone is
        >1.2× incoming almost certainly carries the recipient's OWN
        balance, not propagated theft. Strict accumulation would still
        include it as "the one that pushed us past coverage" when the
        preceding txs were small (e.g. a 50k/300k/60k sequence for a
        60.6k inflow includes both the 50k AND the 300k).

        Reproduced from
        ``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_3a.jsonl``:
        THWY8p got 60.6k USDT from TN6c (hop 3). First outgoing 50k
        (c36ba8d0) didn't cover, then 300k (75e43932) did — the graph
        then showed an absurd 300k branch following 60k. The fix skips
        the 300k and keeps accumulating smaller siblings until coverage.
        """
        tracer = _tracer()
        txs = [
            _tx("small_1", 50_000.0),         # 0.83× — fine
            _tx("oversized", 300_000.0),      # 4.95× — must be skipped
            _tx("oversized2", 123_402.0),     # 2.04× — also skipped
            _tx("oversized3", 200_000.0),     # 3.30× — also skipped
            _tx("small_2", 29_500.0),         # 0.49× — fine; fills coverage
        ]
        selected = tracer._accumulate_hashes(
            txs, 60_600.0, "trx", "USDT", hop_index=3,
        )
        assert "oversized" not in selected
        assert "oversized2" not in selected
        assert "oversized3" not in selected
        assert selected == ["small_1", "small_2"]
        # accumulated 50k + 29.5k = 79.5k > 60.6k — good.

    def test_hop_2_falls_back_to_oversized_when_everything_is_oversized(self):
        """Safety net: if EVERY candidate is oversized (a long-lived
        whale mule with only huge outflows), we must still surface
        something — a noisy branch beats a missing one."""
        tracer = _tracer()
        txs = [
            _tx("big_1", 500_000.0),
            _tx("big_2", 900_000.0),
        ]
        selected = tracer._accumulate_hashes(
            txs, 60_600.0, "trx", "USDT", hop_index=3,
        )
        assert selected == ["big_1"], (
            "with no reasonable-size candidate we fall back to the "
            "earliest oversized tx as a single safety-net pick"
        )

    def test_hop_1_still_keeps_oversized_siblings(self):
        """Hop-1 mixed-funds mode must NOT be affected by the new
        oversized-skip rule — we still need to capture the staged theft
        fan-out past the initial big mule outflow."""
        tracer = _tracer()
        txs = [
            _tx("dust", 10.0),
            _tx("oversized", 220_100.0),  # 2.1× of 105k — mixed-funds signal
            _tx("theft_leg_1", 13_000.0),
            _tx("theft_leg_2", 3_000.0),
        ]
        selected = tracer._accumulate_hashes(
            txs, 105_103.25, "trx", "USDT", hop_index=1,
        )
        # Oversized tx stays in; siblings continue to be collected.
        assert selected == ["dust", "oversized", "theft_leg_1", "theft_leg_2"]
