from agent.models import CaseMeta, Entity, Path, Step, TraceResult, TraceStats
from agent.visualization import generate_visualization_payload


def _trace_for_same_hash_two_legs() -> TraceResult:
    same_hash = "0x1c9a"
    return TraceResult(
        case_meta=CaseMeta(
            case_id="case-hash-path-alignment",
            victim_address="0xvictim",
            blockchain_name="bsc",
            chains=["bsc"],
            asset_symbol="USDT",
        ),
        paths=[
            Path(
                path_id="p1",
                description="inflow leg",
                steps=[
                    Step(
                        step_index=0,
                        from_address="0x1d11",
                        to_address="0xb7cb",
                        tx_hash=same_hash,
                        chain="bsc",
                        asset="USDT",
                        amount_estimate=199500.0,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                ],
                stop_reason="",
            ),
            Path(
                path_id="p2",
                description="outflow leg",
                steps=[
                    Step(
                        step_index=0,
                        from_address="0xb7cb",
                        to_address="0x172f",
                        tx_hash=same_hash,
                        chain="bsc",
                        asset="USDT",
                        amount_estimate=69374.03,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                ],
                stop_reason="",
            ),
        ],
        entities=[
            Entity(address="0x1d11", chain="bsc", role="intermediate", risk_score=0.25),
            Entity(address="0xb7cb", chain="bsc", role="intermediate", risk_score=0.60),
            Entity(address="0x172f", chain="bsc", role="intermediate", risk_score=0.50),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=199500.0, explored_paths=2),
    )


def test_tx_node_path_uses_matching_txlist_leg_not_default_zero():
    same_hash = "0x1c9a"
    trace = _trace_for_same_hash_two_legs()
    tx_list = [
        {
            "inputs": [{"address": "0x1d11", "riskscore": 0.25}],
            "outputs": [{"address": "0xb7cb", "riskscore": 0.60}],
            "hash": same_hash,
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 199500.0,
            "currency": "bsc",
            "tokenId": 9,
            "poolTime": 1,
            "date": 1,
            "path": "0",
            "type": "txEth",
        },
        {
            "inputs": [{"address": "0xb7cb", "riskscore": 0.60}],
            "outputs": [{"address": "0x172f", "riskscore": 0.50}],
            "hash": same_hash,
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 69374.03,
            "currency": "bsc",
            "tokenId": 9,
            "poolTime": 1,
            "date": 1,
            "path": "10",
            "type": "txEth",
        },
    ]

    payload = generate_visualization_payload(trace, tx_list=tx_list, txs=None)
    tx_nodes = [t for t in payload["payload"]["txs"] if t.get("hash") == same_hash]
    assert tx_nodes, "expected tx node for shared hash"

    # The rendered tx nodes must preserve per-leg paths for same-hash
    # multi-transfer txs (not collapse everything to "0").
    paths = {str(t.get("path")) for t in tx_nodes}
    assert "10" in paths
    assert "0" in paths


def test_token_id_prefers_real_tx_row_over_synthetic_zero_row():
    bridge_hash = "0x1e53"
    trace = TraceResult(
        case_meta=CaseMeta(
            case_id="case-bridge-token-preference",
            victim_address="0xbridge-src",
            blockchain_name="eth",
            chains=["eth"],
            asset_symbol="ETH",
        ),
        paths=[
            Path(
                path_id="p1",
                description="bridge continuation",
                steps=[
                    Step(
                        step_index=0,
                        from_address="0xbridge-src",
                        to_address="0xbridge-dst",
                        tx_hash=bridge_hash,
                        chain="eth",
                        asset="ETH",
                        amount_estimate=577723.426715,
                        direction="out",
                        step_type="bridge_transfer",
                    ),
                ],
                stop_reason="",
            ),
        ],
        entities=[
            Entity(address="0xbridge-src", chain="eth", role="intermediate", risk_score=0.25),
            Entity(address="0xbridge-dst", chain="eth", role="intermediate", risk_score=0.25),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=577723.426715, explored_paths=1),
    )

    tx_list = [
        # Real on-chain destination row (USDT on Ethereum).
        {
            "inputs": [{"address": "0xnear-treasury", "riskscore": 0.25}],
            "outputs": [{"address": "0xbridge-dst", "riskscore": 0.25}],
            "hash": bridge_hash,
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 577723426715,
            "currency": "eth",
            "tokenId": 94252,
            "poolTime": 1,
            "date": 1,
            "path": "0",
            "type": "txEth",
        },
        # Synthetic fallback row with tokenId=0 should not override the real one.
        {
            "inputs": [{"address": "0xbridge-src", "riskscore": 0.25}],
            "outputs": [{"address": "0xbridge-dst", "riskscore": 0.25}],
            "hash": bridge_hash,
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 577723426715000,
            "currency": "eth",
            "tokenId": 0,
            "poolTime": 1,
            "date": 1,
            "path": "0",
            "type": "txEth",
        },
    ]

    payload = generate_visualization_payload(trace, tx_list=tx_list, txs=None)
    tx_nodes = [t for t in payload["payload"]["txs"] if t.get("hash") == bridge_hash]
    assert tx_nodes, "expected tx node for bridge hash"
    assert tx_nodes[0].get("token_id") == 94252


def test_same_address_token_lanes_are_linked_with_dashed_edge():
    trace = TraceResult(
        case_meta=CaseMeta(
            case_id="case-token-lane-link",
            victim_address="0xsrc",
            blockchain_name="eth",
            chains=["eth"],
            asset_symbol="ETH",
        ),
        paths=[
            Path(
                path_id="p1",
                description="native lane",
                steps=[
                    Step(
                        step_index=0,
                        from_address="0xsrc",
                        to_address="0xhub",
                        tx_hash="0xhash-native",
                        chain="eth",
                        asset="ETH",
                        amount_estimate=1.0,
                        direction="out",
                        step_type="direct_transfer",
                    )
                ],
                stop_reason="",
            ),
            Path(
                path_id="p2",
                description="token lane",
                steps=[
                    Step(
                        step_index=0,
                        from_address="0xhub",
                        to_address="0xdst",
                        tx_hash="0xhash-usdt",
                        chain="eth",
                        asset="USDT",
                        amount_estimate=2.0,
                        direction="out",
                        step_type="direct_transfer",
                    )
                ],
                stop_reason="",
            ),
        ],
        entities=[
            Entity(address="0xsrc", chain="eth", role="victim", risk_score=0.1),
            Entity(address="0xhub", chain="eth", role="intermediate", risk_score=0.2),
            Entity(address="0xdst", chain="eth", role="intermediate", risk_score=0.3),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=1.0, explored_paths=2),
    )

    tx_list = [
        {
            "inputs": [{"address": "0xsrc", "riskscore": 0.1}],
            "outputs": [{"address": "0xhub", "riskscore": 0.2}],
            "hash": "0xhash-native",
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 1000000000,
            "currency": "eth",
            "tokenId": 0,
            "poolTime": 1,
            "date": 1,
            "path": "0",
            "type": "txEth",
        },
        {
            "inputs": [{"address": "0xhub", "riskscore": 0.2}],
            "outputs": [{"address": "0xdst", "riskscore": 0.3}],
            "hash": "0xhash-usdt",
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 2000000,
            "currency": "eth",
            "tokenId": 94252,
            "poolTime": 1,
            "date": 1,
            "path": "0",
            "type": "txEth",
        },
    ]

    payload = generate_visualization_payload(trace, tx_list=tx_list, txs=None)
    connects = payload["payload"]["connects"]
    same_addr_edges = [
        c for c in connects
        if c.get("data", {}).get("label") == "Same address"
    ]
    assert same_addr_edges, "expected dashed connector between token lanes"
    assert any(
        {edge.get("source"), edge.get("target")} == {"0xhub-eth-0", "0xhub-eth-94252"}
        for edge in same_addr_edges
    )
