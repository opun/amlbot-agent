"""Regression tests for visualization tx list capping."""
from agent.base_tracer import MAX_TX_LIST, BaseTracer


def test_cap_visualization_tx_lists_noop_when_small():
    txs = [{"hash": "a", "x": 1}]
    tx_list = [{"hash": "a", "date": 1}]
    out_list, out_txs = BaseTracer._cap_visualization_tx_lists(tx_list, txs)
    assert out_list == tx_list
    assert out_txs == txs


def test_cap_visualization_tx_lists_truncates():
    tx_list = [{"hash": f"h{i}", "date": i} for i in range(MAX_TX_LIST + 50)]
    txs = [{"hash": f"h{i}"} for i in range(MAX_TX_LIST + 50)]
    out_list, out_txs = BaseTracer._cap_visualization_tx_lists(tx_list, txs)
    assert len(out_list) == MAX_TX_LIST
    assert len(out_txs) <= MAX_TX_LIST
    hashes = {e["hash"] for e in out_list}
    assert all(t.get("hash") in hashes for t in (out_txs or []))
