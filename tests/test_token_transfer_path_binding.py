import json

from agent.base_tracer import BaseTracer


class _StubTracer(BaseTracer):
    async def execute_tool(self, tool_name, arguments):  # pragma: no cover
        raise NotImplementedError

    def _get_client(self):  # pragma: no cover
        raise NotImplementedError


def _tracer() -> BaseTracer:
    return _StubTracer.__new__(_StubTracer)


def _tool_result(transfers: list[dict]) -> str:
    return json.dumps({"data": transfers})


def test_collect_token_transfer_data_prefers_matching_path_from_all_txs():
    tracer = _tracer()
    txs_collected: list[dict] = []
    tx_list_collected: list[dict] = []
    txs_seen: set = set()
    risk_map: dict = {}

    tx_hash = "0x1c9a"
    tool_result = _tool_result([
        {
            "hash": tx_hash,
            "path": "0",
            "token_id": 9,
            "amount": 199500.0,
            "block_time": 1770199738,
            "input": {"address": "0xfrom_path0", "riskscore": 0.25},
            "output": {"address": "0xto_path0", "riskscore": 0.60},
        },
        {
            "hash": tx_hash,
            "path": "2",
            "token_id": 9,
            "amount": 69374.03,
            "block_time": 1770199738,
            "input": {"address": "0xfrom_path2", "riskscore": 0.60},
            "output": {"address": "0xto_path2", "riskscore": 0.50},
        },
    ])
    all_txs_map = {
        tx_hash: {"token_id": 9, "amount": 69374.03, "block_time": 1770199738, "path": "2"}
    }

    tracer._collect_token_transfer_data(
        tool_result=tool_result,
        arguments={"tx_hash": tx_hash, "blockchain_name": "bsc"},
        all_txs_map=all_txs_map,
        risk_map=risk_map,
        txs_collected=txs_collected,
        tx_list_collected=tx_list_collected,
        txs_seen=txs_seen,
    )

    assert len(txs_collected) == 1
    assert txs_collected[0]["path"] == "2"
    assert txs_collected[0]["hash"] == tx_hash

    assert len(tx_list_collected) == 1
    assert tx_list_collected[0]["path"] == "2"
    assert tx_list_collected[0]["inputs"][0]["address"] == "0xfrom_path2"
    assert tx_list_collected[0]["outputs"][0]["address"] == "0xto_path2"
    assert tx_list_collected[0]["amount"] == 69374.03


def test_collect_token_transfer_data_dedupes_on_hash_path_token():
    tracer = _tracer()
    txs_collected: list[dict] = []
    tx_list_collected: list[dict] = []
    txs_seen: set = set()
    risk_map: dict = {}

    tx_hash = "0xabc"
    path0_result = _tool_result([{
        "hash": tx_hash,
        "path": "0",
        "token_id": 9,
        "amount": 100.0,
        "block_time": 1,
        "input": {"address": "0xfrom0", "riskscore": 0.1},
        "output": {"address": "0xto0", "riskscore": 0.2},
    }])
    path2_result = _tool_result([{
        "hash": tx_hash,
        "path": "2",
        "token_id": 9,
        "amount": 50.0,
        "block_time": 2,
        "input": {"address": "0xfrom2", "riskscore": 0.3},
        "output": {"address": "0xto2", "riskscore": 0.4},
    }])

    tracer._collect_token_transfer_data(
        path0_result,
        {"tx_hash": tx_hash, "blockchain_name": "bsc"},
        {tx_hash: {"token_id": 9, "amount": 100.0, "block_time": 1, "path": "0"}},
        risk_map,
        txs_collected,
        tx_list_collected,
        txs_seen,
    )
    tracer._collect_token_transfer_data(
        path0_result,
        {"tx_hash": tx_hash, "blockchain_name": "bsc"},
        {tx_hash: {"token_id": 9, "amount": 100.0, "block_time": 1, "path": "0"}},
        risk_map,
        txs_collected,
        tx_list_collected,
        txs_seen,
    )
    tracer._collect_token_transfer_data(
        path2_result,
        {"tx_hash": tx_hash, "blockchain_name": "bsc"},
        {tx_hash: {"token_id": 9, "amount": 50.0, "block_time": 2, "path": "2"}},
        risk_map,
        txs_collected,
        tx_list_collected,
        txs_seen,
    )

    assert [tx["path"] for tx in txs_collected] == ["0", "2"]
