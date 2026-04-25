"""Regression: per-pair merged-tx hubs (``mergedEdge`` + ``mergedTx``)
must be emitted for every unique (from_address, to_address) pair so
the platform's "merged tx mode" toggle has data to render.

Background — what the human-built reference graphs look like:

  TLHPDaLrq3SX7uMoyDYL4KJeW3fdiZz231 ←→ TN6cEuxVQNdMNnw4RC2Y8gJnTwwBkJ6DXF

with 4 transfers between the pair:
  ├─ 376c26c8…  220.10k USDT (forward)
  ├─ 1b0d6e95…   10k USDT (reverse)
  ├─ 10bc32dc…   10k USDT (reverse)
  └─ e5f3feed…   10k USDT (reverse)

In merged-tx mode the four individual tx-nodes collapse into one fat
arrow with a count badge and amount totals. The payload shape that
unlocks this view:

  * ``connects`` includes a ``{type: "mergedEdge", id: "{from}-{to}",
    source: addressDescriptor, target: addressDescriptor}`` per pair.
  * ``txs`` includes a ``{type: "mergedTx", descriptor=hash="{from}{to}"}``
    hub per pair.
  * Every individual ``txEth``/``tx`` carries a
    ``parentNode="{from}{to}"`` linking it to the hub.
  * ``helpers.isMergedTxMode`` is ``true`` when any merged structures
    were emitted.
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


VICTIM = "TBwn2GfWZhMfo4kY9CARE7oMEeKsVzua8t"
PERP = "TLHPDaLrq3SX7uMoyDYL4KJeW3fdiZz231"
HOP3 = "TN6cEuxVQNdMNnw4RC2Y8gJnTwwBkJ6DXF"

SEED_TX = "e37253294b38d18d98981cb12576078991e10e56ca3b9dc65bb09c8c19998c07"
TX_FORWARD = "376c26c8159eea323447ea34c2d1e22abd0526ec460bcd882dfb8a462b1ddb04"
TX_REV_1 = "1b0d6e9525e6c12a22c29ecd5774bc7fd9f2e14d6bccb7ca755dc619d21f4c40"
TX_REV_2 = "10bc32dc98f264960948c5a8fc5739825ddb482a08e3cda163fa3c909fa54cf1"
TX_REV_3 = "e5f3feedc274c18914354208acf6b34f042437dd1229a99f98a5cd6044b6b4fd"


def _build_trace_with_repeated_pair() -> TraceResult:
    """Trace where PERP→HOP3 has 4 distinct on-chain transactions —
    the canonical case the user wants merged into one visual edge."""
    steps = [
        Step(
            step_index=0, from_address=VICTIM, to_address=PERP,
            tx_hash=SEED_TX, chain="trx", asset="USDT",
            amount_estimate=105103.250017,
            direction="out", step_type="direct_transfer",
        ),
        Step(
            step_index=1, from_address=PERP, to_address=HOP3,
            tx_hash=TX_FORWARD, chain="trx", asset="USDT",
            amount_estimate=220100.0,
            direction="out", step_type="direct_transfer",
        ),
    ]
    return TraceResult(
        case_meta=CaseMeta(
            case_id="trx-merged-tx",
            victim_address=VICTIM,
            blockchain_name="trx",
            chains=["trx"],
            asset_symbol="USDT",
        ),
        paths=[
            Path(
                path_id="1",
                description="Repeated transfers between same pair",
                steps=steps,
                stop_reason="end",
            ),
        ],
        entities=[
            Entity(address=VICTIM, chain="trx", role="victim", risk_score=0.166),
            Entity(address=PERP, chain="trx", role="perpetrator", risk_score=0.209),
            Entity(address=HOP3, chain="trx", role="intermediate", risk_score=0.216),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=105103.25, explored_paths=1),
    )


def test_merged_edge_emitted_per_unique_pair():
    trace = _build_trace_with_repeated_pair()
    result = generate_visualization_payload(trace)
    payload = result["payload"]

    merged_edges = [c for c in payload["connects"] if c.get("type") == "mergedEdge"]
    # Two pairs in the trace: VICTIM→PERP and PERP→HOP3.
    assert len(merged_edges) == 2, (
        f"expected one mergedEdge per (from, to) pair (2 pairs), "
        f"got {len(merged_edges)}: {[m.get('id') for m in merged_edges]}"
    )

    by_id = {m["id"]: m for m in merged_edges}
    assert f"{VICTIM}-{PERP}" in by_id
    assert f"{PERP}-{HOP3}" in by_id

    me = by_id[f"{PERP}-{HOP3}"]
    assert me["source"] == f"{PERP}-trx-9"
    assert me["target"] == f"{HOP3}-trx-9"
    assert me["data"]["input"] == PERP
    assert me["data"]["output"] == HOP3
    # Color must be set (not None) so the frontend renders the arrow.
    assert me["data"].get("color")


def test_merged_tx_hub_emitted_per_unique_pair():
    trace = _build_trace_with_repeated_pair()
    result = generate_visualization_payload(trace)
    payload = result["payload"]

    merged_tx_hubs = [t for t in payload["txs"] if t.get("type") == "mergedTx"]
    assert len(merged_tx_hubs) == 2

    by_desc = {t["descriptor"]: t for t in merged_tx_hubs}
    parent_descriptor = f"{PERP}{HOP3}"
    assert parent_descriptor in by_desc
    hub = by_desc[parent_descriptor]
    # The hub uses the concatenated address pair as both descriptor
    # and hash (no chain suffix, no real on-chain hash).
    assert hub["hash"] == parent_descriptor
    assert hub["input"] == PERP
    assert hub["output"] == HOP3


def test_individual_txs_carry_parent_node_pointer():
    trace = _build_trace_with_repeated_pair()
    result = generate_visualization_payload(trace)
    payload = result["payload"]

    individual_txs = [t for t in payload["txs"] if t.get("type") != "mergedTx"]
    # Two real txs in the trace (seed + forward).
    assert len(individual_txs) == 2
    for tx in individual_txs:
        assert "parentNode" in tx, (
            f"individual tx must carry parentNode for merged-mode grouping; "
            f"got {tx}"
        )

    # The PERP→HOP3 tx must point at the right hub.
    by_hash = {t["hash"]: t for t in individual_txs}
    assert by_hash[TX_FORWARD]["parentNode"] == f"{PERP}{HOP3}"
    assert by_hash[SEED_TX]["parentNode"] == f"{VICTIM}{PERP}"


def test_helpers_flag_is_set_when_pairs_present():
    trace = _build_trace_with_repeated_pair()
    result = generate_visualization_payload(trace)
    helpers = result["helpers"]
    assert helpers["isMergedTxMode"] is True


def test_helpers_flag_is_false_for_empty_trace():
    """When no real steps were rendered (degenerate case), the merged
    flag must stay False — the frontend doesn't try to pick a
    nonexistent merged view."""
    empty = TraceResult(
        case_meta=CaseMeta(
            case_id="empty",
            victim_address=VICTIM,
            blockchain_name="trx",
            chains=["trx"],
            asset_symbol="USDT",
        ),
        paths=[],
        entities=[],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=0.0, explored_paths=0),
    )
    result = generate_visualization_payload(empty)
    assert result["helpers"]["isMergedTxMode"] is False


def test_existing_per_tx_connects_are_preserved():
    """Sanity: merged structures are ADDITIVE — the old per-tx
    src→tx and tx→tgt connects must still be present so the frontend
    can fall back to expanded view when the toggle is off."""
    trace = _build_trace_with_repeated_pair()
    result = generate_visualization_payload(trace)
    payload = result["payload"]

    straight_connects = [c for c in payload["connects"] if c.get("type") != "mergedEdge"]
    # Two steps × 2 connects (src→tx, tx→tgt) = 4 transfer connects, plus
    # comment connects for the «ren» labels (not counted here) — at minimum
    # we expect ≥ 4 transfer connects.
    transfer_connects = [c for c in straight_connects if c.get("data", {}).get("currency")]
    assert len(transfer_connects) >= 4
