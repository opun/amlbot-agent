"""UTXO seed visualization: per-output amounts + deduped src→tx edge.

Real-world bug this guards: a Bitcoin theft tx with one input and seven
outputs produces seven ``Step`` records that all share the same
``tx_hash``. Before the fix, every edge emitted by
``generate_visualization_payload`` used the tx-total amount
(``amount_by_hash[tx_hash]``) — so all seven outputs rendered as
"36.50M sat" when in reality each output was a different per-output
share (17.94M / 2.07M / 0.66M / …). Additionally, the ``src→tx`` edge
was duplicated seven times (once per step) because we didn't track
"this src→tx pair was already emitted".

Both behaviors are asserted here.
"""
from __future__ import annotations

from agent.models import (
    Annotation,
    CaseMeta,
    Entity,
    Path,
    Step,
    TraceResult,
    TraceStats,
)
from agent.visualization import generate_visualization_payload


def _make_step(
    step_index: int,
    from_addr: str,
    to_addr: str,
    amount_btc: float,
    tx_hash: str = "e9cbc26e" + "0" * 56,
    chain: str = "btc",
) -> Step:
    return Step(
        step_index=step_index,
        **{"from": from_addr, "to": to_addr},
        tx_hash=tx_hash,
        chain=chain,
        asset="BTC",
        amount_estimate=amount_btc,
        time=1700000000,
        direction="out",
        step_type="direct_transfer",
        reasoning=f"UTXO output {step_index}",
    )


def _trace_result_from_steps(steps: list[Step]) -> TraceResult:
    entities = [
        Entity(address="bc1qvictim", chain="btc", role="victim"),
    ]
    seen = {"bc1qvictim"}
    for st in steps:
        for addr in (st.from_address, st.to_address):
            if addr not in seen:
                entities.append(Entity(address=addr, chain="btc", role="intermediate"))
                seen.add(addr)
    return TraceResult(
        case_meta=CaseMeta(
            case_id="c1",
            victim_address="bc1qvictim",
            blockchain_name="btc",
            chains=["btc"],
            asset_symbol="BTC",
            token_id=0,
        ),
        paths=[Path(path_id=str(i + 1), description=f"branch {i}", steps=[s])
               for i, s in enumerate(steps)],
        entities=entities,
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=0.365, explored_paths=len(steps)),
    )


class TestUtxoSeedEdgeAmounts:
    """A one-input, three-output UTXO seed. The three output amounts are
    distinct — the visualization must preserve them."""

    TX_HASH = "e9cbc26edbd53c08e26a3a65fad670ab89ef218dff75930e856bb1aaf9450b9d"

    def _setup(self):
        src = "bc1qperpetrator"
        # 0.365 BTC total split 0.179 / 0.021 / 0.165 (values chosen so
        # each is visibly different post-scale).
        steps = [
            _make_step(0, src, "bc1qrecv_big", 0.1794, tx_hash=self.TX_HASH),
            _make_step(0, src, "bc1qrecv_mid", 0.0208, tx_hash=self.TX_HASH),
            _make_step(0, src, "bc1qrecv_small", 0.1648, tx_hash=self.TX_HASH),
        ]
        tr = _trace_result_from_steps(steps)
        # Supply a tx_list entry so amount_by_hash sees the tx-total.
        tx_list = [{
            "hash": self.TX_HASH,
            "currency": "btc",
            "tokenId": 0,
            "fiatRate": 29230.0,
            "amount": 36500000,  # sat (0.365 BTC) ← total input
            "inputs": [{"address": src, "riskscore": 0.5}],
            "outputs": [
                {"address": "bc1qrecv_big", "riskscore": 0.2},
                {"address": "bc1qrecv_mid", "riskscore": 0.2},
                {"address": "bc1qrecv_small", "riskscore": 0.2},
            ],
        }]
        payload = generate_visualization_payload(
            tr, tx_list=tx_list, txs=None, address_info=None,
        )
        return payload["payload"]

    def test_src_to_tx_edge_is_not_duplicated(self):
        """One on-chain tx → one src→tx edge, not three."""
        payload = self._setup()
        src_tx_edges = [
            c for c in payload["connects"]
            if str(c["target"]).startswith(self.TX_HASH)
            and not str(c["source"]).startswith(self.TX_HASH)
            and not str(c["source"]).startswith("«ren»")
        ]
        assert len(src_tx_edges) == 1, (
            f"expected 1 src→tx edge, got {len(src_tx_edges)}: "
            f"{[e['source'] + '→' + e['target'] for e in src_tx_edges]}"
        )

    def test_src_to_tx_edge_carries_tx_total(self):
        """The single src→tx edge should carry the aggregate tx amount
        (36 500 000 sat), not a per-output share."""
        payload = self._setup()
        src_tx_edges = [
            c for c in payload["connects"]
            if str(c["target"]).startswith(self.TX_HASH)
            and not str(c["source"]).startswith(self.TX_HASH)
            and not str(c["source"]).startswith("«ren»")
        ]
        assert src_tx_edges, "no src→tx edge found"
        assert src_tx_edges[0]["data"]["amount"] == 36_500_000

    def test_tx_to_tgt_edges_carry_per_output_amounts(self):
        """Each tx→tgt edge uses the step's own amount_estimate, not
        the aggregate. The three outputs must be distinct."""
        payload = self._setup()
        tx_tgt_edges = [
            c for c in payload["connects"]
            if str(c["source"]).startswith(self.TX_HASH)
            and not str(c["target"]).startswith("«ren»")
        ]
        # Three outputs → three tx→tgt edges, with three distinct amounts.
        amounts = sorted(e["data"]["amount"] for e in tx_tgt_edges)
        assert len(tx_tgt_edges) == 3, (
            f"expected 3 tx→tgt edges, got {len(tx_tgt_edges)}"
        )
        # 0.0208, 0.1648, 0.1794 BTC × 10^8 sat/BTC
        assert amounts == [2_080_000, 16_480_000, 17_940_000], (
            f"per-output amounts wrong: {amounts}"
        )
        # Crucially: none of them should equal the total (36.5M).
        assert all(a != 36_500_000 for a in amounts), (
            "tx→tgt edge is still using tx total, not per-output share"
        )


class TestAccountModelStillWorks:
    """Single-output account-model txs (ETH/TRX) should be unaffected —
    src→tx and tx→tgt amounts coincide."""

    def test_eth_single_output_unchanged(self):
        tx_hash = "0x" + "a" * 64
        step = Step(
            step_index=0,
            **{"from": "0xsrc", "to": "0xdst"},
            tx_hash=tx_hash,
            chain="eth",
            asset="ETH",
            amount_estimate=140_570_190,  # stored in gwei (API base-unit)
            time=1700000000,
            direction="out",
            step_type="direct_transfer",
        )
        tr = _trace_result_from_steps([step])
        tr.case_meta.blockchain_name = "eth"
        tr.case_meta.chains = ["eth"]
        tr.case_meta.asset_symbol = "ETH"
        tx_list = [{
            "hash": tx_hash, "currency": "eth", "tokenId": 0,
            "fiatRate": 1.0, "amount": 140_570_190,  # gwei — matches estimate
            "inputs": [{"address": "0xsrc"}], "outputs": [{"address": "0xdst"}],
        }]
        payload = generate_visualization_payload(
            tr, tx_list=tx_list, txs=None, address_info=None,
        )["payload"]

        src_tx = [c for c in payload["connects"]
                  if c["source"] == "0xsrc-eth-0" and str(c["target"]).startswith(tx_hash)]
        tx_tgt = [c for c in payload["connects"]
                  if str(c["source"]).startswith(tx_hash) and c["target"] == "0xdst-eth-0"]
        assert len(src_tx) == 1
        assert len(tx_tgt) == 1
        # Critical: both edges carry the same API base-unit value
        # (140_570_190 gwei), NOT 140_570_190 × 10^9 wei. This is the
        # regression that produced "0.14 ETH" on src→tx and "140M ETH"
        # on tx→tgt in the e0c92b55 case.
        assert src_tx[0]["data"]["amount"] == 140_570_190
        assert tx_tgt[0]["data"]["amount"] == 140_570_190
