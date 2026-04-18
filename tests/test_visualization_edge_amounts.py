"""Regression tests for the "NaN" bug on edges in the visualization payload.

Every ``connects[].data`` entry must carry a numeric ``amount`` so the
frontend never renders "NaN" on an edge, even when its own
hash-to-txList fallback fails to resolve a raw base-unit amount.

Historical failures:
  * USDT-on-ETH token transfer where ``tokenId=94252`` but the edge
    ``currency`` was ``"eth"`` — frontend couldn't join to the txList
    and printed ``NaN ETH``.
  * Bridge/swap destination steps whose tx hash never appeared in the
    source-chain ``all_txs`` feed and therefore had no txList row.
"""

from __future__ import annotations

from agent.models import (
    CaseMeta,
    Entity,
    Path,
    Step,
    TraceResult,
    TraceStats,
)
from agent.visualization import generate_visualization_payload


def _trace_usdt_on_eth() -> TraceResult:
    victim = "0xvictim"
    deposit = "0x147ac0b39675769e55a0f0e7fdd3641b47963661"
    recipient = "0xe3a03c2b941e71991560e0e408e0c2b39877e4a6"
    return TraceResult(
        case_meta=CaseMeta(
            case_id="case-usdt",
            victim_address=victim,
            blockchain_name="eth",
            chains=["eth"],
            asset_symbol="ETH",
        ),
        paths=[
            Path(
                path_id="1",
                description="USDT theft",
                steps=[
                    Step(
                        step_index=0,
                        from_address=victim,
                        to_address=deposit,
                        tx_hash="0xa" * 64,
                        chain="eth",
                        asset="ETH",
                        amount_estimate=1.5,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                    Step(
                        step_index=1,
                        from_address=deposit,
                        to_address=recipient,
                        tx_hash="0x5438d5cb244afe6031e5caed15cceda905d0af819d9628219982b75d48f93f46",
                        chain="eth",
                        asset="USDT",
                        token_id=94252,
                        amount_estimate=413.759798,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                ],
                stop_reason="Dead end - no outgoing transactions",
            ),
        ],
        entities=[
            Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
            Entity(address=deposit, chain="eth", role="intermediate", risk_score=0.5),
            Entity(address=recipient, chain="eth", role="intermediate", risk_score=0.2),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=1.5, explored_paths=1),
    )


class TestEdgeAmountsAreResolved:
    def test_connects_amount_never_null_with_provided_tx_list(self):
        """When the upstream tx_list already carries raw amounts, every
        connect for a step with that hash must echo the raw amount on
        ``data.amount`` (so the frontend renders it directly)."""
        trace = _trace_usdt_on_eth()
        tx_list = [
            {
                "hash": "0x5438d5cb244afe6031e5caed15cceda905d0af819d9628219982b75d48f93f46",
                "amount": 413759798,
                "currency": "eth",
                "tokenId": 94252,
                "fiatRate": 1.0,
            },
        ]
        payload = generate_visualization_payload(
            trace, tx_list=tx_list
        ).get("payload", {})

        connects_for_hash = [
            c for c in payload.get("connects", [])
            if "0x5438d5cb" in c.get("source", "")
            or "0x5438d5cb" in c.get("target", "")
        ]
        assert connects_for_hash, "expected connects referencing the USDT tx"
        for conn in connects_for_hash:
            amt = conn.get("data", {}).get("amount")
            assert amt is not None, (
                f"edge amount must not be null (frontend renders NaN); "
                f"connect={conn}"
            )
            assert amt == 413759798, (
                f"edge amount must match provided tx_list raw amount; "
                f"got {amt}"
            )

    def test_connects_amount_falls_back_to_step_estimate(self):
        """No upstream tx_list row for the hash (bridge/swap dst case): we
        must still emit a numeric ``data.amount`` derived from
        ``step.amount_estimate`` and the token's decimals."""
        trace = _trace_usdt_on_eth()
        payload = generate_visualization_payload(trace).get("payload", {})

        connects_for_hash = [
            c for c in payload.get("connects", [])
            if "0x5438d5cb" in c.get("source", "")
            or "0x5438d5cb" in c.get("target", "")
        ]
        assert connects_for_hash
        for conn in connects_for_hash:
            amt = conn.get("data", {}).get("amount")
            assert amt is not None, f"edge amount must not be null; connect={conn}"
            assert amt > 0, f"edge amount must be positive; got {amt}"
            assert amt == int(413.759798 * 10**6), (
                f"USDT decimals must scale step.amount_estimate by 10^6; "
                f"got {amt}"
            )

    def test_stale_step_asset_does_not_break_edge_token_id(self):
        """THE real prod bug: base_tracer propagates the trace-level asset
        ("ETH") to every step, so even a USDT hop records ``step.asset =
        "ETH"``. Without per-hash token_id lookup, the edge's ``data.token_id``
        is 0 and ``data.currency`` is "eth", which breaks the frontend's
        join to the USDT ``txList`` row (``tokenId=94252``) and renders NaN.

        After the fix, the edge must carry ``token_id=94252`` even though
        step.asset is the stale "ETH".
        """
        victim = "0xvictim"
        deposit = "0x147ac0b39675769e55a0f0e7fdd3641b47963661"
        recipient = "0xe3a03c2b941e71991560e0e408e0c2b39877e4a6"
        trace = TraceResult(
            case_meta=CaseMeta(
                case_id="case-stale-asset",
                victim_address=victim,
                blockchain_name="eth",
                chains=["eth"],
                asset_symbol="ETH",
            ),
            paths=[
                Path(
                    path_id="1",
                    description="USDT hop with stale ETH asset",
                    steps=[
                        Step(
                            step_index=0,
                            from_address=deposit,
                            to_address=recipient,
                            tx_hash="0x5438d5cb244afe6031e5caed15cceda905d0af819d9628219982b75d48f93f46",
                            chain="eth",
                            asset="ETH",
                            amount_estimate=413.759798,
                            direction="out",
                            step_type="direct_transfer",
                        ),
                    ],
                    stop_reason="Dead end - no outgoing transactions",
                ),
            ],
            entities=[
                Entity(address=deposit, chain="eth", role="intermediate", risk_score=0.5),
                Entity(address=recipient, chain="eth", role="intermediate", risk_score=0.2),
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=413.759798, explored_paths=1),
        )
        tx_list = [
            {
                "hash": "0x5438d5cb244afe6031e5caed15cceda905d0af819d9628219982b75d48f93f46",
                "amount": 413759798,
                "currency": "eth",
                "tokenId": 94252,
                "fiatRate": 1.0,
            },
        ]
        payload = generate_visualization_payload(trace, tx_list=tx_list).get("payload", {})

        connects_for_hash = [
            c for c in payload.get("connects", [])
            if "0x5438d5cb" in c.get("source", "") or "0x5438d5cb" in c.get("target", "")
        ]
        assert connects_for_hash, "expected connects referencing the USDT tx"
        for conn in connects_for_hash:
            data = conn.get("data", {})
            assert data.get("token_id") == 94252, (
                f"edge must advertise USDT token_id even when step.asset is "
                f"stale 'ETH'; got token_id={data.get('token_id')}"
            )
            assert data.get("amount") == 413759798, (
                f"edge amount must echo tx_list raw amount; got {data.get('amount')}"
            )
            assert data.get("currency") == "eth"

        txs_for_hash = [
            t for t in payload.get("txs", [])
            if t.get("hash", "").startswith("0x5438d5cb")
        ]
        assert txs_for_hash, "expected at least one txs entry for the USDT hash"
        for tx in txs_for_hash:
            assert tx.get("token_id") == 94252, (
                f"txs entry must carry the tx's real token_id; got {tx}"
            )

    def test_native_eth_step_uses_native_unit_for_fallback(self):
        victim = "0xvictim_eth"
        middle = "0xmid_eth"
        trace = TraceResult(
            case_meta=CaseMeta(
                case_id="case-eth-native",
                victim_address=victim,
                blockchain_name="eth",
                chains=["eth"],
                asset_symbol="ETH",
            ),
            paths=[
                Path(
                    path_id="1",
                    description="Native ETH",
                    steps=[
                        Step(
                            step_index=0,
                            from_address=victim,
                            to_address=middle,
                            tx_hash="0xdeadbeef" * 8,
                            chain="eth",
                            asset="ETH",
                            token_id=0,
                            amount_estimate=2.5,
                            direction="out",
                            step_type="direct_transfer",
                        ),
                    ],
                    stop_reason="Dead end - no outgoing transactions",
                ),
            ],
            entities=[
                Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
                Entity(address=middle, chain="eth", role="intermediate", risk_score=0.2),
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=2.5, explored_paths=1),
        )
        payload = generate_visualization_payload(trace).get("payload", {})
        amounts = [
            c.get("data", {}).get("amount")
            for c in payload.get("connects", [])
            if "0xdeadbeef" in c.get("source", "") or "0xdeadbeef" in c.get("target", "")
        ]
        assert amounts, "expected at least one native-ETH connect"
        for amt in amounts:
            assert amt is not None and amt > 0
