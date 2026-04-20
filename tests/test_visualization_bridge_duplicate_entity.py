"""Cross-chain bridge shouldn't produce a dangling «ren» comment.

Regression driver: ``trace_postprocess.ensure_entity`` rekeys entities by
``(address, chain)`` and auto-adds an ``intermediate`` Entity for every
chain an address touches. On a Bridgers-style ETH→TRON handoff that
produces TWO Entity rows for the bridge contract: the authoritative
``(Bridgers, eth, bridge_service)`` from the tracer and a second
``(Bridgers, trx, intermediate)`` auto-generated during postprocess.

The old visualization keyed ``service_comment_map`` by address alone and
emitted one comment *per entity row* — so the bridge got two comments
sharing the same ``«ren»2`` descriptor, the second one pointing at a
``-trx-0`` descriptor that didn't exist among the items (because the
TRX-side Bridgers node lived at ``-trx-9``). Operators saw a dangling
"Destination address" label floating at (-68, -60) and the bridge link
appeared broken.
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


def _build_cross_chain_bridge_trace() -> TraceResult:
    victim = "0xvictim"
    perp = "0xperp"
    bridge = "0xbridge"
    dst = "Trecipient"

    # ETH side: victim → perp → bridge (USDT ERC20).
    step0 = Step(
        step_index=0,
        **{"from": victim, "to": perp},
        tx_hash="0xtx0", chain="eth", asset="USDT",
        amount_estimate=60000.0, time=1776000000,
        direction="out", step_type="direct_transfer",
    )
    step1 = Step(
        step_index=1,
        **{"from": perp, "to": bridge},
        tx_hash="0xtx1", chain="eth", asset="USDT",
        amount_estimate=60000.0, time=1776000100,
        direction="out", step_type="direct_transfer",
    )
    # Bridge transfer: chain switches to TRX, asset to USDT(TRON).
    step2 = Step(
        step_index=2,
        **{"from": bridge, "to": dst},
        tx_hash="0xbridge_dst_tx", chain="trx", asset="USDT",
        amount_estimate=60000.0, time=1776000200,
        direction="out", step_type="bridge_transfer",
    )

    path = Path(path_id="1", description="bridge", stop_reason=None,
                steps=[step0, step1, step2])

    # Two entities for the bridge address, mirroring what the postprocess
    # produces: the authoritative bridge_service entity plus the
    # auto-added intermediate twin on the destination chain.
    entities = [
        Entity(address=victim, chain="eth", role="victim"),
        Entity(address=perp, chain="eth", role="perpetrator"),
        Entity(address=bridge, chain="eth", role="bridge_service",
               labels=["Bridgers"]),
        Entity(address=bridge, chain="trx", role="intermediate",
               notes="Auto-added during postprocess"),
        Entity(address=dst, chain="trx", role="intermediate"),
    ]

    return TraceResult(
        case_meta=CaseMeta(
            case_id="c", victim_address=victim,
            blockchain_name="eth", chains=["eth", "trx"],
            asset_symbol="USDT", token_id=94252,
        ),
        paths=[path],
        entities=entities,
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=60000.0, explored_paths=1),
    )


class TestBridgeCommentDescriptorIntegrity:
    def test_single_comment_per_bridge_address(self):
        tr = _build_cross_chain_bridge_trace()
        payload = generate_visualization_payload(
            tr, tx_list=None, txs=None, address_info=None,
        )["payload"]
        bridge_comments = [
            c for c in payload["comments"]
            if "0xbridge" in c.get("text", "").lower()
            or "Bridgers" in c.get("text", "")
            or "Bridge service" in c.get("text", "")
            or "Destination address" in c.get("text", "")
        ]
        # Filter to ones whose «ren» connector targets the bridge address.
        # The cross-chain handoff connector uses bridge-* as *both* source
        # and target, so isolate real «ren» → bridge links.
        ren_targets = {
            c["target"]
            for c in payload["connects"]
            if isinstance(c.get("source"), str)
            and c["source"].startswith("«ren»")
            and isinstance(c.get("target"), str)
            and c["target"].startswith("0xbridge-")
        }
        # Exactly one «ren» connector should point at any bridge-* node.
        assert len(ren_targets) == 1, (
            f"expected one «ren» connector to bridge, got: {ren_targets}"
        )
        # And no two comments should share the same descriptor.
        descs = [c["descriptor"] for c in payload["comments"]]
        assert len(descs) == len(set(descs)), (
            f"duplicate «ren» descriptors in comments: {descs}"
        )

    def test_comment_targets_existing_item(self):
        """The comment→node connector must land on a real item; otherwise
        the UI floats the label at (-68, -60) (our sentinel for the
        default ``pos={0,0}`` fallback)."""
        tr = _build_cross_chain_bridge_trace()
        payload = generate_visualization_payload(
            tr, tx_list=None, txs=None, address_info=None,
        )["payload"]
        item_descriptors = {it["descriptor"] for it in payload["items"]}
        ren_connector_targets = [
            c["target"] for c in payload["connects"]
            if isinstance(c.get("source"), str) and c["source"].startswith("«ren»")
        ]
        for target in ren_connector_targets:
            assert target in item_descriptors, (
                f"«ren» connector points at non-existent item {target!r}; "
                f"items: {sorted(item_descriptors)}"
            )

    def test_no_dangling_comment_at_default_origin(self):
        """No comment should end up at the (-68, -60) default-origin
        fallback — that's the visible symptom of a missing address_desc."""
        tr = _build_cross_chain_bridge_trace()
        payload = generate_visualization_payload(
            tr, tx_list=None, txs=None, address_info=None,
        )["payload"]
        dangling = [
            c for c in payload["comments"]
            if c.get("x") == -68 and c.get("y") == -60
        ]
        assert not dangling, f"dangling default-origin comments: {dangling}"


class TestThroughNodeNotMislabeled:
    """An intermediate node with outgoing edges in any other path must
    NOT be labeled "Destination address" just because some path stops
    at it. Operator feedback: the label appearing mid-graph (with the
    trace clearly continuing from the node) is confusing."""

    def _build_branching_trace(self) -> TraceResult:
        """victim → perp → hub; hub then branches to (a) a dead-end
        intermediate, and (b) continues to a further recipient. The hub
        is a through-node on branch (b) but the leaf of branch (a).
        """
        victim, perp, hub, dead, farther = (
            "Tvictim", "Tperp", "Thub", "Tdead", "Tfarther",
        )
        s0 = Step(step_index=0, **{"from": victim, "to": perp}, tx_hash="t0",
                  chain="trx", asset="USDT", amount_estimate=100.0,
                  time=1776000000, direction="out", step_type="direct_transfer")
        s1 = Step(step_index=1, **{"from": perp, "to": hub}, tx_hash="t1",
                  chain="trx", asset="USDT", amount_estimate=100.0,
                  time=1776000100, direction="out", step_type="direct_transfer")
        s2 = Step(step_index=2, **{"from": hub, "to": dead}, tx_hash="t2",
                  chain="trx", asset="USDT", amount_estimate=40.0,
                  time=1776000200, direction="out", step_type="direct_transfer")
        s3 = Step(step_index=2, **{"from": hub, "to": farther}, tx_hash="t3",
                  chain="trx", asset="USDT", amount_estimate=60.0,
                  time=1776000300, direction="out", step_type="direct_transfer")
        branch_a = Path(path_id="1", description="dead", steps=[s0, s1, s2],
                        stop_reason="Dead end - no outgoing transactions")
        branch_b = Path(path_id="2", description="onward", steps=[s0, s1, s3],
                        stop_reason=None)
        entities = [
            Entity(address=victim, chain="trx", role="victim"),
            Entity(address=perp, chain="trx", role="perpetrator"),
            Entity(address=hub, chain="trx", role="intermediate"),
            Entity(address=dead, chain="trx", role="intermediate"),
            Entity(address=farther, chain="trx", role="intermediate"),
        ]
        return TraceResult(
            case_meta=CaseMeta(
                case_id="c", victim_address=victim,
                blockchain_name="trx", chains=["trx"],
                asset_symbol="USDT", token_id=9,
            ),
            paths=[branch_a, branch_b],
            entities=entities,
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=100.0, explored_paths=2),
        )

    def test_hub_is_not_labeled_destination(self):
        tr = self._build_branching_trace()
        payload = generate_visualization_payload(
            tr, tx_list=None, txs=None, address_info=None,
        )["payload"]
        # Find the comment (if any) whose connector lands on the hub.
        hub_comments = []
        for conn in payload["connects"]:
            src = conn.get("source", "")
            tgt = conn.get("target", "")
            if isinstance(src, str) and src.startswith("«ren»") and "Thub-" in tgt:
                desc = src
                for c in payload["comments"]:
                    if c.get("descriptor") == desc:
                        hub_comments.append(c.get("text", ""))
        assert not any("Destination address" in t for t in hub_comments), (
            f"through-node hub labeled as destination: {hub_comments}"
        )


class TestBlockchainNameOnTxList:
    """Frontend sidebar renders 'Blockchain: MOCK DATA' when it can't
    resolve the chain for a tx. Our synthesized tx_list entries must
    carry an explicit human-readable chain name so the card shows
    'TRON' / 'Ethereum' instead of the mock sentinel."""

    def test_synthesized_tx_has_blockchain_name(self):
        tr = _build_cross_chain_bridge_trace()
        payload = generate_visualization_payload(
            tr, tx_list=None, txs=None, address_info=None,
        )["payload"]
        trx_txs = [t for t in payload["helpers"].get("txList", []) if t.get("currency") == "trx"] \
            if isinstance(payload.get("helpers"), dict) else []
        # Fallback: the full envelope puts txList under helpers at the
        # outer level; generate_visualization_payload returns the inner
        # payload in these tests. Look in txList wherever it is.
        if not trx_txs:
            # payload here is the inner "payload" dict; helpers is on
            # the outer envelope. The test case doesn't expose helpers
            # easily, so reach through the envelope by re-invoking.
            envelope = generate_visualization_payload(
                tr, tx_list=None, txs=None, address_info=None,
            )
            helpers = envelope.get("helpers", {})
            trx_txs = [t for t in (helpers.get("txList") or []) if t.get("currency") == "trx"]
        assert trx_txs, "no TRX tx_list entries emitted at all"
        for tx in trx_txs:
            assert tx.get("blockchain"), (
                f"TRX tx missing blockchain name: {tx}"
            )
            assert tx["blockchain"] != "MOCK DATA"
            assert tx["blockchain"].upper() == "TRON" or tx["blockchain"] == "trx"


class TestBridgeHandoffConnector:
    """The ETH-side and TRX-side Bridgers nodes must be visually linked,
    otherwise the TRX component of the trace looks disconnected."""

    def test_cross_chain_bridge_connector_emitted(self):
        tr = _build_cross_chain_bridge_trace()
        payload = generate_visualization_payload(
            tr, tx_list=None, txs=None, address_info=None,
        )["payload"]
        connector_endpoints = {
            (c["source"], c["target"]) for c in payload["connects"]
            if isinstance(c.get("source"), str)
            and c["source"].startswith("0xbridge-")
            and isinstance(c.get("target"), str)
            and c["target"].startswith("0xbridge-")
            and c["source"] != c["target"]
        }
        assert connector_endpoints, (
            "no cross-chain bridge connector between ETH-side and TRX-side "
            "bridge nodes; connects were: "
            + str([(c.get("source"), c.get("target")) for c in payload["connects"]])
        )
