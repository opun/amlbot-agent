"""Visualization hides the terminal edge of dust-trimmed paths.

Operator feedback: a branch trimmed at <1% of stolen funds shouldn't
clutter the graph — the tracer already recorded it and added a "Dust
Trimmed" annotation. The data stays in ``TraceResult`` for API use; we
just don't render the last step/node/comment.
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


def _mk_step(
    step_index: int,
    from_addr: str,
    to_addr: str,
    amount: float,
    tx_hash: str,
) -> Step:
    return Step(
        step_index=step_index,
        **{"from": from_addr, "to": to_addr},
        tx_hash=tx_hash,
        chain="trx",
        asset="USDT",
        amount_estimate=amount,
        time=1776000000,
        direction="out",
        step_type="direct_transfer",
    )


def _build_trace(paths: list[Path], dust_entity_addr: str) -> TraceResult:
    entities = [Entity(address="Tvictim", chain="trx", role="victim")]
    seen = {"Tvictim"}
    for p in paths:
        for s in p.steps:
            for addr in (s.from_address, s.to_address):
                if addr not in seen:
                    seen.add(addr)
                    entities.append(Entity(address=addr, chain="trx", role="intermediate"))
    return TraceResult(
        case_meta=CaseMeta(
            case_id="c1", victim_address="Tvictim",
            blockchain_name="trx", chains=["trx"], asset_symbol="USDT",
            token_id=9,
        ),
        paths=paths,
        entities=entities,
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=60000.0, explored_paths=len(paths)),
    )


class TestDustBranchHidden:
    """Tronify-style case: a 3.64-USDT edge out of 60 000 USDT stolen."""

    def test_dust_recipient_not_in_graph(self):
        # Normal hop: perpetrator receives from victim.
        # Dust hop: hub → Tronify (0.006% of stolen).
        normal_path = Path(
            path_id="1",
            description="Main flow",
            stop_reason=None,
            steps=[
                _mk_step(0, "Tvictim", "Tperp", 60000.0, "0xnormal"),
            ],
        )
        dust_path = Path(
            path_id="2",
            description="Tronify dust",
            stop_reason="Below dust threshold (0.01% of stolen amount)",
            steps=[
                _mk_step(0, "Tvictim", "Tperp", 60000.0, "0xnormal"),  # shared prefix
                _mk_step(1, "Tperp", "Ttronify", 3.64, "0xdust"),
            ],
        )
        tr = _build_trace([normal_path, dust_path], dust_entity_addr="Ttronify")

        payload = generate_visualization_payload(tr, tx_list=None, txs=None, address_info=None)["payload"]

        # The dust recipient address must not appear as a node in the graph.
        addresses = {it["address"] for it in payload["items"] if it.get("type") == "address"}
        assert "Ttronify" not in addresses, (
            f"dust recipient rendered as a node: {sorted(addresses)}"
        )
        # Nor as an edge target.
        edge_targets = [c["target"] for c in payload["connects"]]
        assert not any("Ttronify" in str(t) for t in edge_targets), (
            "edge points at dust recipient"
        )
        # And no "Trace endpoint (dust amount)" comment stays orphan:
        dust_comments = [c for c in payload["comments"]
                         if "dust amount" in c.get("text", "").lower()]
        assert not dust_comments, "dust comment still rendered"

    def test_dust_hidden_even_when_only_path(self):
        """Edge case: the whole trace is one dust branch. Node still hidden."""
        only_path = Path(
            path_id="1",
            description="Tiny",
            stop_reason="Below dust threshold (0.17% of stolen amount)",
            steps=[_mk_step(0, "Tvictim", "Tdust", 100.0, "0xdust")],
        )
        tr = _build_trace([only_path], dust_entity_addr="Tdust")

        payload = generate_visualization_payload(tr, tx_list=None, txs=None, address_info=None)["payload"]
        addresses = {it["address"] for it in payload["items"] if it.get("type") == "address"}
        assert "Tdust" not in addresses

    def test_non_dust_path_still_rendered(self):
        """A normal-stop path must not be affected by the dust filter."""
        path = Path(
            path_id="1",
            description="Normal",
            stop_reason="Reached CEX deposit",
            steps=[_mk_step(0, "Tvictim", "Tcex", 60000.0, "0xnormal")],
        )
        tr = _build_trace([path], dust_entity_addr="n/a")

        payload = generate_visualization_payload(tr, tx_list=None, txs=None, address_info=None)["payload"]
        addresses = {it["address"] for it in payload["items"] if it.get("type") == "address"}
        assert "Tcex" in addresses, "normal path lost its terminal"


class TestDustLabelOnlyOnServices:
    """Operator feedback: a bare intermediate address shouldn't carry a
    "Trace endpoint (dust amount)" comment. The label only makes sense
    when the dust landed on an identified service (CEX / bridge / DEX /
    OTC / unidentified-but-known-brand)."""

    def _make_trace_with_dust_leaf(self, role: str, labels: list[str] | None = None) -> TraceResult:
        """Dust path terminated at a custom-role leaf. The path's
        stop_reason contains the word 'dust' but is NOT the
        "Below dust threshold" form (which would be hidden entirely).
        """
        step = _mk_step(0, "Tvictim", "Tleaf", 100.0, "0xdust")
        path = Path(
            path_id="1",
            description="some dust-ish stop",
            # Something that mentions dust but isn't the "Below dust
            # threshold" sentinel the visualization filters out.
            stop_reason="trace ended with dust tail",
            steps=[step],
        )
        entities = [
            Entity(address="Tvictim", chain="trx", role="victim"),
            Entity(
                address="Tleaf", chain="trx",
                role=role,  # type: ignore[arg-type]
                labels=labels or [],
            ),
        ]
        return TraceResult(
            case_meta=CaseMeta(
                case_id="c", victim_address="Tvictim",
                blockchain_name="trx", chains=["trx"], asset_symbol="USDT",
                token_id=9,
            ),
            paths=[path],
            entities=entities,
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=60000.0, explored_paths=1),
        )

    def _comments_for(self, tr: TraceResult) -> list[str]:
        payload = generate_visualization_payload(tr, tx_list=None, txs=None, address_info=None)["payload"]
        return [c.get("text", "") for c in payload["comments"]]

    def test_plain_intermediate_gets_no_dust_label(self):
        tr = self._make_trace_with_dust_leaf(role="intermediate")
        comments = self._comments_for(tr)
        assert not any("dust amount" in t.lower() for t in comments), (
            "plain intermediate leaf should not get a dust-endpoint label"
        )

    def test_cex_leaf_gets_dust_label(self):
        tr = self._make_trace_with_dust_leaf(role="cex_deposit", labels=["Binance"])
        comments = self._comments_for(tr)
        assert any("dust amount" in t.lower() for t in comments), (
            "identified CEX leaf on dust stop should keep the endpoint label"
        )

    def test_bridge_leaf_gets_dust_label(self):
        tr = self._make_trace_with_dust_leaf(role="bridge_service", labels=["LayerZero"])
        comments = self._comments_for(tr)
        assert any("dust amount" in t.lower() for t in comments)

    def test_unidentified_service_with_brand_gets_dust_label(self):
        """Identified brand (Tronify etc.) — role=unidentified_service,
        owner label present."""
        tr = self._make_trace_with_dust_leaf(
            role="unidentified_service", labels=["Tronify"],
        )
        comments = self._comments_for(tr)
        assert any("dust amount" in t.lower() for t in comments)
