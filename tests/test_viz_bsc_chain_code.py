"""Regression: BSC traces must emit ``currency: "bsc"`` (not ``"bnb"``).

The whole codebase canonicalizes BSC's chain code to ``"bsc"`` —
``models.py``, ``base_tracer._normalize_chain``, ``currency_registry``,
and the AMLBot platform's frontend all key BSC tokens under that code.
``visualization._normalize_chain`` previously returned ``"bnb"`` for any
BSC input, which broke the (chain, token_id) lookup against
``currencies.json`` (where BSC USDT lives under ``("bsc", 9)``):

  * ``currencyInfo`` for native BSC dropped to the fallback path and
    rendered ``unit=9, symbol="bnb", name="BNB Chain"`` instead of the
    registry's ``unit=9, symbol="bnb", name="Binance Smart Chain"``.
  * ``currencyInfo`` for USDT-on-BSC fell through entirely and rendered
    ``unit=6, symbol="bnb", name="bnb"`` — the frontend then displayed
    "NaN" for every USDT amount.
  * Address descriptors used ``-bnb-`` (e.g. ``…-bnb-9``) which did not
    line up with the ``-bsc-`` lanes the platform expects.

This test pins the fix.
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
    _normalize_chain,
    generate_visualization_payload,
)


def test_normalize_chain_maps_bsc_aliases_to_bsc():
    for alias in ("bsc", "bnb", "bep20", "binance", "BSC", "BNB"):
        assert _normalize_chain(alias) == "bsc", (
            f"BSC alias '{alias}' must normalize to 'bsc' (chain code) — "
            "the rest of the codebase keys BSC under 'bsc', not 'bnb'."
        )


VICTIM = "0x54764a6b7b06f27d95159b12ffc3ce020947eaac"
PERP = "0xf23e10a82817607f3283255be9f150253e7019ea"
USDT_TX = "0xcdabe9f7f614fe2720693c03643d1a6f8d68cd0a3fa1bfeaf77996ca999718eb"


def _build_bsc_trace() -> TraceResult:
    return TraceResult(
        case_meta=CaseMeta(
            case_id="bsc-usdt",
            victim_address=VICTIM,
            blockchain_name="bsc",
            chains=["bsc"],
            asset_symbol="USDT",
        ),
        paths=[
            Path(
                path_id="1",
                description="USDT theft on BSC",
                steps=[
                    Step(
                        step_index=0,
                        from_address=VICTIM,
                        to_address=PERP,
                        tx_hash=USDT_TX,
                        chain="bsc",
                        asset="USDT",
                        amount_estimate=199500.0,
                        direction="out",
                        step_type="direct_transfer",
                    ),
                ],
                stop_reason="end",
            ),
        ],
        entities=[
            Entity(address=VICTIM, chain="bsc", role="victim", risk_score=0.25),
            Entity(address=PERP, chain="bsc", role="perpetrator", risk_score=0.25),
        ],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=199500.0, explored_paths=1),
    )


def test_bsc_payload_emits_bsc_chain_code_and_canonical_currency_info():
    """Top-level smoke for BSC USDT — descriptors, edges, currencyInfo."""
    trace = _build_bsc_trace()
    tx_list = [
        {
            "inputs": [{"address": VICTIM, "riskscore": 0.25}],
            "outputs": [{"address": PERP, "riskscore": 0.25}],
            "hash": USDT_TX,
            "fiatRate": 1,
            "addressesCount": 2,
            "amount": int(1.995e23),
            "currency": "bsc",
            "tokenId": 9,
            "poolTime": 1770199406,
            "date": 1770199406,
            "path": "0",
            "type": "txEth",
        },
    ]

    result = generate_visualization_payload(trace, tx_list=tx_list)
    payload = result["payload"]
    helpers = result["helpers"]

    # Items must use the BSC chain code in their descriptors.
    descriptors = {item["descriptor"] for item in payload["items"]}
    assert any(d.endswith("-bsc-9") for d in descriptors), (
        f"BSC items must use -bsc- lane, got {descriptors}"
    )
    assert not any("-bnb-" in d for d in descriptors), (
        f"No item descriptor may carry -bnb- (BSC chain code is 'bsc'); got {descriptors}"
    )

    # Edges advertise currency='bsc' so the frontend joins to the right txList row.
    transfer_edges = [c for c in payload["connects"] if c["data"].get("currency")]
    assert transfer_edges, "expected at least one transfer edge"
    for edge in transfer_edges:
        assert edge["data"]["currency"] == "bsc", (
            f"edge currency must be 'bsc'; got {edge['data']['currency']}"
        )
        assert edge["data"]["token_id"] == 9

    # currencyInfo: native BSC and USDT-on-BSC, both keyed under 'bsc'.
    by_tid = {(ci["currency"], ci["token_id"]): ci for ci in helpers["currencyInfo"]}
    native = by_tid.get(("bsc", 0))
    assert native is not None, f"missing native BSC entry; got {list(by_tid)}"
    assert native["symbol"] == "bnb", (
        f"native BSC symbol must be 'bnb' (asset symbol on BSC); got {native['symbol']}"
    )

    usdt = by_tid.get(("bsc", 9))
    assert usdt is not None, f"missing BSC-USDT entry; got {list(by_tid)}"
    assert usdt["symbol"] == "USDT"
    assert usdt["issuer"] == "0x55d398326f99059ff775485246999027b3197955"
    assert usdt["unit"] == 18, (
        "BSC USDT unit must come from the registry (18), not the fallback (6)"
    )


def test_get_token_id_uses_registry_not_python_hash():
    """The non-deterministic ``hash()`` fallback handed out IDs like 431 for
    BSC USDT and broke the (chain, token_id) lookup. The fix must use the
    currency registry; missing tokens fall back to 0, never a random int."""
    from agent.visualization import _get_token_id

    # Registry-backed: BSC USDT == 9, ETH USDT == 94252, TRX USDT == 9.
    assert _get_token_id("USDT", "bsc") == 9
    assert _get_token_id("USDT", "eth") == 94252
    assert _get_token_id("USDT", "trx") == 9

    # Native shortcuts still return 0.
    assert _get_token_id("BNB", "bsc") == 0
    assert _get_token_id("ETH", "eth") == 0
    assert _get_token_id("TRX", "trx") == 0

    # Unknown asset falls back to 0 (deterministic) — not a random hash.
    assert _get_token_id("DEFINITELY_NOT_A_REAL_TOKEN_X9Z", "eth") == 0
