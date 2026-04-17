"""Tests for the visualization label logic.

Regression coverage for the bug where every terminal ``intermediate``
address was forcibly relabelled as a CEX deposit ("Exchange deposit
address"), even for plain mule/unclassified addresses.
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


def _case_meta(victim: str = "0xvictim") -> CaseMeta:
    return CaseMeta(
        case_id="case-1",
        victim_address=victim,
        blockchain_name="eth",
        chains=["eth"],
        asset_symbol="ETH",
    )


def _step(idx: int, frm: str, to: str, tx: str = "0xhash", amount: float = 10.0) -> Step:
    return Step(
        step_index=idx,
        from_address=frm,
        to_address=to,
        tx_hash=tx,
        chain="eth",
        asset="ETH",
        amount_estimate=amount,
        direction="out",
        step_type="direct_transfer",
    )


def _get_comment(payload: dict, descriptor: str) -> dict | None:
    for c in payload.get("comments", []):
        if c.get("descriptor") == descriptor:
            return c
    return None


def _descriptor_for_address_comment(payload: dict, address: str) -> str | None:
    """Find the comment descriptor (e.g. "«ren»3") that points at the
    given address via its address-node descriptor."""
    address_desc = f"{address}-eth-0"
    for conn in payload.get("connects", []):
        src = conn.get("source", "")
        if conn.get("target") == address_desc and src.startswith("«ren»"):
            return src
    return None


class TestTerminalLabelForIntermediates:
    def test_dead_end_intermediate_is_labeled_trace_endpoint_not_exchange(self):
        """Historical bug: a terminal intermediate with no classification
        was relabelled "Exchange deposit address". We now label it as a
        Trace endpoint with the dead-end stop reason."""
        victim = "0xvictim"
        perp = "0xperp"
        mule = "0xmule"
        trace = TraceResult(
            case_meta=_case_meta(victim),
            paths=[
                Path(
                    path_id="1",
                    description="Primary theft flow",
                    steps=[
                        _step(0, victim, perp),
                        _step(1, perp, mule),
                    ],
                    stop_reason="Dead end - no outgoing transactions",
                ),
            ],
            entities=[
                Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
                Entity(address=perp, chain="eth", role="perpetrator", risk_score=0.75),
                Entity(address=mule, chain="eth", role="intermediate", risk_score=0.2),
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=10.0, explored_paths=1),
        )

        payload = generate_visualization_payload(trace).get("payload", {})
        comment_desc = _descriptor_for_address_comment(payload, mule)
        assert comment_desc is not None, "terminal intermediate must get a comment"
        comment = _get_comment(payload, comment_desc)
        assert comment is not None

        assert "Exchange deposit address" not in comment["text"], (
            f"Terminal intermediate must NOT be labelled as an exchange deposit: "
            f"got {comment['text']!r}"
        )
        assert "Trace endpoint" in comment["text"]
        assert "no outflows" in comment["text"].lower()

    def test_real_cex_deposit_keeps_exchange_label(self):
        """Addresses actually classified as cex_deposit should still be
        labeled "Exchange deposit address" (regression guard so we don't
        regress the real case)."""
        victim = "0xvictim2"
        perp = "0xperp2"
        cex = "0xcex"
        trace = TraceResult(
            case_meta=_case_meta(victim),
            paths=[
                Path(
                    path_id="1",
                    description="Theft to CEX",
                    steps=[_step(0, victim, perp), _step(1, perp, cex)],
                    stop_reason="Reached terminal entity - CEX deposit",
                ),
            ],
            entities=[
                Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
                Entity(address=perp, chain="eth", role="perpetrator", risk_score=0.75),
                Entity(address=cex, chain="eth", role="cex_deposit", risk_score=0.5,
                       labels=["ChangeNOW"]),
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=10.0, explored_paths=1),
        )

        payload = generate_visualization_payload(trace).get("payload", {})
        comment_desc = _descriptor_for_address_comment(payload, cex)
        assert comment_desc is not None
        comment = _get_comment(payload, comment_desc)
        assert comment is not None
        assert "Exchange deposit address" in comment["text"]
        assert "ChangeNOW" in comment["text"]

    def test_max_hop_terminal_gets_hop_limit_label(self):
        victim = "0xv3"
        perp = "0xp3"
        last = "0xlast"
        trace = TraceResult(
            case_meta=_case_meta(victim),
            paths=[
                Path(
                    path_id="1",
                    description="Long chain",
                    steps=[_step(0, victim, perp), _step(1, perp, last)],
                    stop_reason="Max hop limit reached",
                ),
            ],
            entities=[
                Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
                Entity(address=perp, chain="eth", role="perpetrator", risk_score=0.75),
                Entity(address=last, chain="eth", role="intermediate", risk_score=0.2),
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=10.0, explored_paths=1),
        )
        payload = generate_visualization_payload(trace).get("payload", {})
        desc = _descriptor_for_address_comment(payload, last)
        comment = _get_comment(payload, desc)
        assert comment is not None
        assert "Exchange deposit address" not in comment["text"]
        assert "hop limit" in comment["text"].lower()

    def test_cap_reached_terminal_gets_cap_label(self):
        victim = "0xv4"
        perp = "0xp4"
        last = "0xlast4"
        trace = TraceResult(
            case_meta=_case_meta(victim),
            paths=[
                Path(
                    path_id="1",
                    description="Capped",
                    steps=[_step(0, victim, perp), _step(1, perp, last)],
                    stop_reason="Global traced amount cap reached",
                ),
            ],
            entities=[
                Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
                Entity(address=perp, chain="eth", role="perpetrator", risk_score=0.75),
                Entity(address=last, chain="eth", role="intermediate", risk_score=0.2),
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=10.0, explored_paths=1),
        )
        payload = generate_visualization_payload(trace).get("payload", {})
        desc = _descriptor_for_address_comment(payload, last)
        comment = _get_comment(payload, desc)
        assert comment is not None
        assert "Exchange deposit address" not in comment["text"]
        assert "cap reached" in comment["text"].lower()

    def test_entity_role_is_not_mutated(self):
        """Regression: previously the visualization mutated
        ``entity.role`` from ``intermediate`` to ``cex_deposit`` in-place.
        That corrupts the trace result returned to downstream consumers
        (API callers, tests, exports). The role must remain authoritative."""
        victim = "0xv5"
        perp = "0xp5"
        mule = "0xmule5"

        mule_entity = Entity(address=mule, chain="eth", role="intermediate", risk_score=0.2)
        trace = TraceResult(
            case_meta=_case_meta(victim),
            paths=[
                Path(
                    path_id="1",
                    description="Mule",
                    steps=[_step(0, victim, perp), _step(1, perp, mule)],
                    stop_reason="Dead end - no outgoing transactions",
                ),
            ],
            entities=[
                Entity(address=victim, chain="eth", role="victim", risk_score=0.8),
                Entity(address=perp, chain="eth", role="perpetrator", risk_score=0.75),
                mule_entity,
            ],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=10.0, explored_paths=1),
        )

        generate_visualization_payload(trace)
        assert mule_entity.role == "intermediate", (
            f"visualization must not mutate entity.role; got {mule_entity.role!r}"
        )
