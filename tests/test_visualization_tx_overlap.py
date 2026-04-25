"""Regression: tx nodes sharing the same (source, target) pair must not
land on identical (mid_x, mid_y) coordinates.

When a mule address sends multiple distinct txs to the same recipient
(dust + real, or a staged series of equal-sized payouts), the old
visualization payload placed every tx node on ``(src+tgt)/2`` — the
frontend then stacked all amount labels on top of each other, producing
the "10k USDT)SDT" garbled labels and the spaghetti of overlapping
red edges the user screenshotted from
``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_c5.jsonl``.

Fix: in ``visualization.py``, offset successive txs sharing the same
``(src_desc, tgt_desc)`` by ``±_TX_PAIR_STEP_PX`` along Y, alternating
above/below the midpoint.
"""
from __future__ import annotations

from agent.models import (
    CaseMeta,
    Path,
    Step,
    TraceResult,
    TraceStats,
)
from agent.visualization import generate_visualization_payload


def _two_tx_same_pair_trace() -> TraceResult:
    victim = "TBwn2GfWZhMfo4kY9CARE7oMEeKsVzua8t"
    mule = "TLHPDaLrq3SX7uMoyDYL4KJeW3fdiZz231"
    exit_ = "TN6cEuxVQNdMNnw4RC2Y8gJnTwwBkJ6DXF"
    return TraceResult(
        case_meta=CaseMeta(
            case_id="case-usdt-trx",
            victim_address=victim,
            blockchain_name="trx",
            chains=["trx"],
            asset_symbol="USDT",
        ),
        paths=[
            Path(
                path_id="1",
                description="USDT theft",
                steps=[
                    Step(
                        step_index=0,
                        from_address=victim,
                        to_address=mule,
                        tx_hash="e37253294b38d18d98981cb12576078991e10e56ca3b9dc65bb09c8c19998c07",
                        chain="trx",
                        asset="USDT",
                        amount_estimate=105103.25,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                    Step(
                        step_index=1,
                        from_address=mule,
                        to_address=exit_,
                        tx_hash="b36ae81c828f414ce93d4d8c28c7c0fd1d558878ece6d5cb130b6cb368c1ab51",
                        chain="trx",
                        asset="USDT",
                        amount_estimate=10.0,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                    Step(
                        step_index=2,
                        from_address=mule,
                        to_address=exit_,
                        tx_hash="376c26c8159eea323447ea34c2d1e22abd0526ec460bcd882dfb8a462b1ddb04",
                        chain="trx",
                        asset="USDT",
                        amount_estimate=220100.0,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                ],
            ),
        ],
        entities=[],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=105103.25, explored_paths=1),
    )


class TestTxOverlapAvoidance:
    def test_same_pair_txs_get_distinct_midpoints(self):
        result = _two_tx_same_pair_trace()
        payload = generate_visualization_payload(result)
        txs = payload["payload"]["txs"]

        same_pair = [
            t for t in txs
            if t["hash"] in (
                "b36ae81c828f414ce93d4d8c28c7c0fd1d558878ece6d5cb130b6cb368c1ab51",
                "376c26c8159eea323447ea34c2d1e22abd0526ec460bcd882dfb8a462b1ddb04",
            )
        ]
        assert len(same_pair) == 2, "both same-pair txs must be emitted"
        coords = [(t["x"], t["y"]) for t in same_pair]
        assert coords[0] != coords[1], (
            f"duplicate tx midpoints {coords[0]} — labels would stack "
            "and the frontend renders the amount as garbled text"
        )

    def test_first_tx_of_pair_keeps_midpoint(self):
        """The first tx in a pair stays on the natural midpoint so
        single-tx pairs (the common case) render unchanged."""
        result = _two_tx_same_pair_trace()
        payload = generate_visualization_payload(result)
        txs = payload["payload"]["txs"]
        # The seed tx (victim→mule) is the only tx between that pair, so
        # its midpoint must match ``(src+tgt)/2`` exactly.
        seed = next(t for t in txs if t["hash"].startswith("e3725329"))
        items = {i["address"]: i for i in payload["payload"]["items"]}
        src_pos = items["TBwn2GfWZhMfo4kY9CARE7oMEeKsVzua8t"]
        tgt_pos = items["TLHPDaLrq3SX7uMoyDYL4KJeW3fdiZz231"]
        assert seed["x"] == (src_pos["x"] + tgt_pos["x"]) / 2
        assert seed["y"] == (src_pos["y"] + tgt_pos["y"]) / 2

    def test_sibling_offsets_straddle_midpoint(self):
        """The 2nd tx of a pair should be offset ABOVE the midpoint
        (``+_TX_PAIR_STEP_PX``), not stacked on it. Guards against a
        future "always offset downward" regression that would pile the
        spread onto one side of the tx row."""
        result = _two_tx_same_pair_trace()
        payload = generate_visualization_payload(result)
        txs = {t["hash"]: t for t in payload["payload"]["txs"]}
        items = {i["address"]: i for i in payload["payload"]["items"]}
        src_pos = items["TLHPDaLrq3SX7uMoyDYL4KJeW3fdiZz231"]
        tgt_pos = items["TN6cEuxVQNdMNnw4RC2Y8gJnTwwBkJ6DXF"]
        baseline_y = (src_pos["y"] + tgt_pos["y"]) / 2
        first = txs["b36ae81c828f414ce93d4d8c28c7c0fd1d558878ece6d5cb130b6cb368c1ab51"]
        second = txs["376c26c8159eea323447ea34c2d1e22abd0526ec460bcd882dfb8a462b1ddb04"]
        assert first["y"] == baseline_y
        assert second["y"] != baseline_y
        assert abs(second["y"] - baseline_y) >= 20.0, (
            "sibling must be visibly separated from the midpoint"
        )
