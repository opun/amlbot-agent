"""End-to-end regression for the exact prod trace that kept showing NaN.

Reproduces the payload shape reported for
``0xe0c92b55…802ce86`` (stolen-ETH case that hops into a USDT transfer at
``0x5438d5cb…f46``) using the very same ``tx_list``/``txs`` blocks the
tracer emits. Asserts that after the visualization fix:

  * the USDT edge advertises ``token_id=94252`` (not 0)
  * the USDT tx entry advertises ``token_id=94252`` (not 0)
  * the edge amount is the raw USDT base-unit amount (``413759798``)

If this test passes but the live server still emits ``token_id=0``, the
running process is stale — re-run `python -m agent.api`.
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
from agent.visualization import (
    _token_id_from_descriptor,
    generate_visualization_payload,
)


def test_token_id_from_descriptor_parses_usdt_descriptor():
    assert _token_id_from_descriptor(
        "0x5438d5cb244afe6031e5caed15cceda905d0af819d9628219982b75d48f93f46-eth-94252-7"
    ) == 94252


def test_token_id_from_descriptor_parses_native_descriptor():
    assert _token_id_from_descriptor(
        "0xe0c92b5519912358a0cb8c95dfc4831b083596db9ef81590463500480802ce86-eth-0-0"
    ) == 0


def test_token_id_from_descriptor_parses_synthetic_tron_descriptor():
    # Synthetic bridge txs use ``tx-{uuid}`` as the hash, so descriptor has
    # an extra dash segment at the front; the second-to-last segment is
    # still the token_id.
    assert _token_id_from_descriptor(
        "tx-ff5312349af5426ba45284326b7a6466-trx-0-7"
    ) == 0


def test_token_id_from_descriptor_returns_none_for_garbage():
    assert _token_id_from_descriptor(None) is None
    assert _token_id_from_descriptor("") is None
    assert _token_id_from_descriptor("not-a-descriptor") is None
    assert _token_id_from_descriptor("no-enough-dashes") is None
    assert _token_id_from_descriptor("hash-eth-notanint-7") is None


def test_edge_token_id_falls_back_to_descriptor_when_txlist_is_empty():
    """Belt-and-suspenders: when ``tx_list`` is empty (none of the harvests
    populated ``token_id_by_hash``) but the pre-built tx descriptor still
    encodes the right token_id, the main loop's descriptor-parse fallback
    must recover it for the edge. Previously this path returned 0 and the
    frontend rendered "NaN".
    """
    trace = _build_trace()
    txs = [
        {
            "currency": "eth",
            "descriptor": f"{USDT_TX}-eth-94252-7",
            "hash": USDT_TX,
            "token_id": 94252,
            "type": "txEth",
            "color": "#EC292C",
            "path": "0",
        },
    ]
    result = generate_visualization_payload(trace, tx_list=None, txs=txs)
    usdt_edges = [
        c for c in result["payload"]["connects"]
        if USDT_TX[:12] in c.get("target", "")
    ]
    assert usdt_edges and all(e["data"]["token_id"] == 94252 for e in usdt_edges)


USDT_TX = "0x5438d5cb244afe6031e5caed15cceda905d0af819d9628219982b75d48f93f46"
DEPOSIT = "0x147ac0b39675769e55a0f0e7fdd3641b47963661"
RECIPIENT = "0xe3a03c2b941e71991560e0e408e0c2b39877e4a6"
VICTIM = "0x264bd8291fae1d75db2c5f573b07faa6715997b5"


def _build_trace() -> TraceResult:
    return TraceResult(
        case_meta=CaseMeta(
            case_id="case-usdt-hop",
            victim_address=VICTIM,
            blockchain_name="eth",
            chains=["eth"],
            asset_symbol="ETH",
        ),
        paths=[
            Path(
                path_id="1",
                description="Stolen ETH that ends in a USDT deposit to Binance",
                steps=[
                    Step(
                        step_index=0,
                        from_address=DEPOSIT,
                        to_address=RECIPIENT,
                        tx_hash=USDT_TX,
                        chain="eth",
                        asset="ETH",  # stale, trace-level asset (the prod bug)
                        amount_estimate=413.759798,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                ],
                stop_reason="Dust",
            ),
        ],
        entities=[
            Entity(address=DEPOSIT, chain="eth", role="intermediate", risk_score=0.498),
            Entity(address=RECIPIENT, chain="eth", role="intermediate", risk_score=0.141),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=413.759798, explored_paths=1),
    )


def test_live_payload_shape_is_usdt_not_eth():
    trace = _build_trace()
    tx_list = [
        {
            "inputs": [{"address": DEPOSIT, "riskscore": 0.498}],
            "outputs": [{"address": RECIPIENT, "riskscore": 0.141}],
            "hash": USDT_TX,
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": 413759798,
            "currency": "eth",
            "tokenId": 94252,
            "poolTime": 1760399819,
            "date": 1760399819,
            "path": "0",
            "type": "txEth",
        },
    ]
    txs = [
        {
            "currency": "eth",
            "descriptor": f"{USDT_TX}-eth-94252-7",
            "hash": USDT_TX,
            "token_id": 94252,
            "x": 2352,
            "y": 311.25,
            "color": "#EC292C",
            "path": "0",
            "type": "txEth",
        },
    ]

    result = generate_visualization_payload(trace, tx_list=tx_list, txs=txs)
    payload = result.get("payload", {})
    helpers = result.get("helpers", {})

    usdt_edges = [
        c for c in payload.get("connects", [])
        if USDT_TX[:12] in c.get("source", "") or USDT_TX[:12] in c.get("target", "")
    ]
    assert usdt_edges, "expected edges wired to the USDT tx descriptor"
    for edge in usdt_edges:
        data = edge["data"]
        assert data["token_id"] == 94252, (
            f"USDT edge must advertise token_id=94252 (got {data['token_id']}); "
            f"stale viz code is still running if this is 0"
        )
        assert data["amount"] == 413759798
        assert data["currency"] == "eth"

    usdt_txs = [t for t in payload.get("txs", []) if t.get("hash") == USDT_TX]
    assert usdt_txs, "expected a txs entry for the USDT tx"
    for tx in usdt_txs:
        assert tx["token_id"] == 94252, (
            f"USDT txs entry must advertise token_id=94252 (got {tx['token_id']})"
        )

    currency_info_tids = {ci["token_id"] for ci in helpers.get("currencyInfo", [])}
    # The fix must also ensure the currencyInfo entry for USDT is present,
    # otherwise the frontend can't decimal-scale even with a correct token_id.
    assert 94252 in currency_info_tids, (
        f"currencyInfo must include USDT (token_id=94252); got tids={currency_info_tids}"
    )
