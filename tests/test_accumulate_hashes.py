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
