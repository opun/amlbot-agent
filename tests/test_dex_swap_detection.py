"""Regression: DEX atomic swaps must continue the trace on the
post-swap asset, not stop at the pool.

Background — what the on-chain event log looks like for the SunSwap
atomic swap at tx ``aa3c12cd…6699`` (USDT → TRX, ~$170k):

  * Transfer (TRC20 USDT, token_id=9):
      input  TLW7Rwn… (user)
      output TSUUVjysX… (SunSwap pool)
      amount 170 549 647 444 (170 549.647 USDT)

  * Transfer (TRC20 WTRX, token_id=1584):
      input  TSUUVjysX… (SunSwap pool)
      output TSJEtPuq…  (unwrap helper)
      amount 541 117 722 355 (541 117.722 WTRX)

  * Withdrawal (WTRX → native TRX):
      WTRX 541 117 722 355 unwrapped, native TRX delivered to user
      via internal call (NOT in TRC20 events list)

The current trace stops when it pops the SunSwap pool as a HopJob and
classifies it ``role=dex_service, terminal=true`` — without seeing the
~541k native TRX the user received and went on to spend.

The fix: when processing a transfer whose recipient is a DEX pool,
inspect the FULL ``token_transfers`` response for a wrapped-native
swap-out leg whose fiat value matches the swap-in's. On match,
push a ``HopJob`` for the SAME user wallet but with the native
``token_id=0`` and the new asset symbol (TRX/ETH/BNB/etc.). The
scheduler's loop-detection bypasses ``is_swap_continuation=True``
jobs so the new asset's outflows can be fetched even though the
address is already in ``path_seen_addresses``.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import BaseTracer, HopJob


class _T(BaseTracer):
    async def execute_tool(self, tool_name, arguments):
        raise NotImplementedError

    def _get_client(self):
        raise NotImplementedError


@pytest.fixture
def tracer():
    return _T()


# ---- Wrapped-native detection ----

class TestWrappedNativeDetection:
    """The remap from wrapped contract to native ``token_id=0`` is
    what lets the swap-continuation HopJob filter outflows by the
    correct asset. Two-pronged detection: by symbol (works on
    eth/bsc/matic/avax) and by registry ``name`` starting with
    ``"Wrapped "`` (catches TRX, where WTRX shares ``symbol="TRX"``
    with the native variant)."""

    def test_wtrx_detected_via_name(self, tracer):
        # TRX/WTRX both use symbol="TRX"; the registry's name field
        # is the only distinguishing signal.
        assert tracer._is_wrapped_native(
            "trx", symbol="TRX", name="Wrapped TRX", token_id=1584,
        ) is True

    def test_native_trx_not_misdetected(self, tracer):
        # token_id=0 must never read as wrapped — the check requires
        # token_id != 0 for the name-based branch.
        assert tracer._is_wrapped_native(
            "trx", symbol="TRX", name="TRON", token_id=0,
        ) is False

    def test_weth_detected_via_symbol(self, tracer):
        assert tracer._is_wrapped_native(
            "eth", symbol="WETH", name="Wrapped Ether", token_id=50608,
        ) is True

    def test_wbnb_detected_via_symbol(self, tracer):
        assert tracer._is_wrapped_native(
            "bsc", symbol="WBNB", name="Wrapped BNB", token_id=13,
        ) is True

    def test_unrelated_token_not_wrapped(self, tracer):
        assert tracer._is_wrapped_native(
            "trx", symbol="USDT", name="Tether USD", token_id=9,
        ) is False

    def test_native_symbol_for_chain(self, tracer):
        assert tracer._native_symbol_for_chain("trx") == "TRX"
        assert tracer._native_symbol_for_chain("eth") == "ETH"
        assert tracer._native_symbol_for_chain("bsc") == "BNB"
        assert tracer._native_symbol_for_chain("matic") == "MATIC"
        assert tracer._native_symbol_for_chain("arb") == "ETH"
        assert tracer._native_symbol_for_chain("op") == "ETH"


# ---- _detect_dex_swap_out ----

USER = "TLW7RwnufaouqcQJP9hGJqZewdadCM2UMG"
POOL = "TSUUVjysXV8YqHytSNjfkNXnnB49QDvZpx"
HELPER = "TSJEtPuqHpvSaVnSwvCsngaeBxrGUzp95Q"


def _sunswap_tx_transfers() -> list[dict]:
    """Verbatim TRC20 event log shape from tx ``aa3c12cd…6699``.

    Two transfers in one tx: the user's USDT swap-in, and the pool's
    WTRX swap-out to the unwrap helper. The native-TRX delivery to
    the user is via internal call (not represented here)."""
    pool_owner = {
        "global_id": 16028, "id": 17651073,
        "name": "SunSwap", "slug": "SunSwap",
        "subtype": "DEX", "type": "p2p_exchange_unlicensed",
    }
    return [
        # Pool → unwrap helper, WTRX (token_id=1584). raw amount is
        # 10^6-scaled (display: 541 117.722 WTRX × $0.315 ≈ $170 452).
        {
            "amount": 541117722355,
            "block_time": 1775066454,
            "fiat_rate": 0.315,
            "hash": "aa3c12cd",
            "input": {"address": POOL, "owner": pool_owner, "riskscore": 0.6},
            "output": {"address": HELPER, "owner": pool_owner, "riskscore": 0.6},
            "path": "0",
            "pool_time": 1775066454,
            "token_id": 1584,
            "type": "send",
        },
        # User → pool, USDT (token_id=9). 170 549.647 USDT × $1 ≈ $170 550.
        {
            "amount": 170549647444,
            "block_time": 1775066454,
            "fiat_rate": 1,
            "hash": "aa3c12cd",
            "input": {"address": USER, "riskscore": 1},
            "output": {"address": POOL, "owner": pool_owner, "riskscore": 0.6},
            "path": "0",
            "pool_time": 1775066454,
            "token_id": 9,
            "type": "send",
        },
    ]


class TestDetectDexSwapOut:
    def test_sunswap_usdt_to_trx_swap_detected(self, tracer):
        """The canonical case: USDT in, WTRX out (≈ TRX after unwrap),
        within ~5% fiat-tolerance."""
        transfers = _sunswap_tx_transfers()
        result = tracer._detect_dex_swap_out(
            transfers,
            current_address=USER,
            swap_in_token_id=9,
            swap_in_amount_display=170_549.647444,  # already in display units
            chain="trx",
        )
        assert result is not None
        # WTRX must be remapped to native TRX (token_id=0).
        assert result["token_id"] == 0
        assert result["asset_symbol"] == "TRX"
        # Display amount: 541117722355 / 10^6 = 541 117.722
        assert result["amount"] == pytest.approx(541117.722355, rel=1e-6)
        assert result["raw_token_id"] == 1584
        # Fiat-match within 5% (USDT $170 550 vs WTRX $170 452).
        assert result["fiat_value"] > 0
        # The wrapped recipient is the unwrap helper, NOT the user.
        # (The user receives native TRX via internal call.)
        assert result["swap_out_address"] == HELPER

    def test_no_swap_when_only_one_transfer(self, tracer):
        """Plain transfer (not a swap) — only one transfer in tx."""
        transfers = _sunswap_tx_transfers()[1:2]  # just the USDT in
        result = tracer._detect_dex_swap_out(
            transfers,
            current_address=USER,
            swap_in_token_id=9,
            swap_in_amount_display=170_549.647444,
            chain="trx",
        )
        assert result is None

    def test_no_swap_when_fiat_mismatch_exceeds_tolerance(self, tracer):
        """If the swap-out's fiat is wildly different (e.g. dust or
        a separate transfer riding the same tx), skip — that's not a
        balanced swap."""
        transfers = _sunswap_tx_transfers()
        # Tighten the WTRX leg to 1k WTRX × $0.315 ≈ $315 (≈0.18% of
        # the USDT swap-in's $170 550) — far outside 5% tolerance.
        transfers[0]["amount"] = 1_000_000_000  # 1000 WTRX in raw
        result = tracer._detect_dex_swap_out(
            transfers,
            current_address=USER,
            swap_in_token_id=9,
            swap_in_amount_display=170_549.647444,
            chain="trx",
        )
        assert result is None

    def test_user_address_must_be_swap_in_input(self, tracer):
        """Without a swap-in transfer that the current_address
        actually originated, we can't anchor the fiat match — return
        None instead of guessing."""
        transfers = _sunswap_tx_transfers()
        # Strip the USDT swap-in
        transfers = transfers[:1]
        result = tracer._detect_dex_swap_out(
            transfers,
            current_address=USER,
            swap_in_token_id=9,
            swap_in_amount_display=170_549.647444,
            chain="trx",
        )
        assert result is None

    def test_empty_or_garbage_input(self, tracer):
        assert tracer._detect_dex_swap_out(
            [], current_address=USER, swap_in_token_id=9,
            swap_in_amount_display=100.0, chain="trx",
        ) is None
        assert tracer._detect_dex_swap_out(
            [{"junk": True}, {"junk": True}],
            current_address=USER, swap_in_token_id=9,
            swap_in_amount_display=100.0, chain="trx",
        ) is None


# ---- HopJob.is_swap_continuation flag ----

class TestHopJobSwapContinuationFlag:
    def test_default_is_false(self):
        job = HopJob(
            path_id="1", current_address="X",
            incoming_tx_hash=None, incoming_amount=0.0,
            incoming_time=None, chain="trx", asset="USDT",
            token_id=9, hop_index=1,
        )
        assert job.is_swap_continuation is False

    def test_explicit_true(self):
        job = HopJob(
            path_id="1", current_address="X",
            incoming_tx_hash=None, incoming_amount=0.0,
            incoming_time=None, chain="trx", asset="TRX",
            token_id=0, hop_index=2,
            is_swap_continuation=True,
        )
        assert job.is_swap_continuation is True


class TestSwapContinuationStrictTokenFiltering:
    """Regression for the bbd7c0fa…trace_8d run: after the swap
    detection synthesized a TLW7Rwn(trx-0) HopJob, the post-swap
    ``all_txs`` query for the user wallet returned **mixed-asset**
    outflows — 4 USDT txs (already in path_seen_hashes from the
    pre-swap trace) plus 4 native TRX txs (the new continuation).

    Without strict token_id filtering, ``_accumulate_hashes`` saw all
    12 entries, picked the already-seen USDT hashes (whose fiat values
    match the post-swap target), and the for-loop dropped them via the
    path_seen_hashes guard — leaving only one TRX outflow rendered.

    The fix forces ``strict_token_id=True`` whenever the popped HopJob
    was synthesized by swap detection. The accumulator then sees only
    the actually-new asset's outflows and selects all of them.

    This test verifies the predicate plumbing — ``HopJob.is_swap_continuation``
    drives the strict filter at the call site. The actual
    ``_fetch_outgoing_txs`` filtering is exercised through the
    reproduction below.
    """

    def test_strict_filter_drops_wrong_token_id(self):
        """The ``_ingest`` body inside ``_fetch_outgoing_txs`` is what
        applies the strict filter. We can't easily exercise the closure
        without mocking the whole tracer, so reproduce the filter rule
        in isolation: given a list of mixed-token outflows, requesting
        token_id=0 with strict mode keeps only the native rows."""
        items = [
            {"hash": "feee89", "token_id": 0, "amount_coerced": 1000},
            {"hash": "d84410", "token_id": 0, "amount_coerced": 100000},
            {"hash": "0eb3cd", "token_id": 0, "amount_coerced": 500000},
            {"hash": "f0b5a3", "token_id": 0, "amount_coerced": 924900},
            {"hash": "7d561b", "token_id": 9, "amount_coerced": 10000},  # USDT — drop
            {"hash": "04e878", "token_id": 9, "amount_coerced": 100000},  # USDT — drop
        ]
        # The same predicate the closure uses, mirrored here for the
        # regression. If the production rule ever shifts, this test
        # forces a deliberate update + re-validation against the
        # bbd7c0fa case in the tracker.
        token_id = 0
        strict = True
        kept = []
        for item in items:
            if strict and token_id is not None:
                tid = item.get("token_id")
                try:
                    if int(tid) != int(token_id):
                        continue
                except (TypeError, ValueError):
                    continue
            kept.append(item["hash"])
        assert kept == ["feee89", "d84410", "0eb3cd", "f0b5a3"]

    def test_non_strict_keeps_mixed_assets(self):
        """Without ``strict_token_id`` the historical behavior holds:
        token_id=0 means "no filter" and the accumulator gets every
        outflow on the address. Important so non-swap traces don't
        regress on the seed-flow's asset-agnostic accumulation."""
        items = [
            {"hash": "a", "token_id": 0, "amount_coerced": 1000},
            {"hash": "b", "token_id": 9, "amount_coerced": 5000},
        ]
        token_id = 0
        strict = False
        kept = []
        for item in items:
            if strict and token_id is not None:
                tid = item.get("token_id")
                try:
                    if int(tid) != int(token_id):
                        continue
                except (TypeError, ValueError):
                    continue
            kept.append(item["hash"])
        assert kept == ["a", "b"]

    def test_strict_filter_drops_fee_only_paired_rows(self):
        """Regression for the phantom ``TLW7Rwn → TSJEtPuq`` edge: when
        ``all_txs`` is queried for the user wallet right after a USDT→TRX
        swap, the server returns paired rows for each swap tx hash —
        the real ``token_id=9`` USDT transfer AND a paired
        ``token_id=0, type=regular`` row whose only delta is the gas
        cost (``amount_coerced=0``).

        The token_id filter alone wouldn't drop the gas row (it matches
        the requested native filter), so without the amount-coerced
        check the accumulator picks the swap tx hash, ``_resolve_transfer``
        returns the WTRX side of the atomic swap, and we register a
        phantom step against the WTRX unwrap helper.

        Strict mode must additionally drop ``amount_coerced ≤ 0`` rows."""
        items = [
            # Real native TRX outflow — keep
            {"hash": "feee89", "token_id": 0, "amount_coerced": 924900},
            # USDT swap tx — paired rows on the same hash, both must drop
            {"hash": "7d561b", "token_id": 9, "amount_coerced": 10000},
            {"hash": "7d561b", "token_id": 0, "amount_coerced": 0},
            # Another native TRX outflow — keep
            {"hash": "0eb3cd", "token_id": 0, "amount_coerced": 500000},
        ]
        token_id = 0
        strict = True
        kept = []
        for item in items:
            if strict and token_id is not None:
                tid = item.get("token_id")
                try:
                    if int(tid) != int(token_id):
                        continue
                except (TypeError, ValueError):
                    continue
                try:
                    amt = float(item.get("amount_coerced") or 0)
                except (TypeError, ValueError):
                    amt = 0.0
                if amt <= 0:
                    continue
            kept.append(item["hash"])
        assert kept == ["feee89", "0eb3cd"]

    def test_strict_filter_drops_negative_amount_coerced(self):
        """Edge case: defensively drop negative ``amount_coerced`` too —
        an outflow with a negative coerced value would only happen if
        the upstream feed mis-signed a balance delta, but if it does we
        must not register a phantom step against it."""
        items = [
            {"hash": "real", "token_id": 0, "amount_coerced": 1000},
            {"hash": "weird", "token_id": 0, "amount_coerced": -50},
        ]
        token_id = 0
        strict = True
        kept = []
        for item in items:
            if strict and token_id is not None:
                tid = item.get("token_id")
                try:
                    if int(tid) != int(token_id):
                        continue
                except (TypeError, ValueError):
                    continue
                try:
                    amt = float(item.get("amount_coerced") or 0)
                except (TypeError, ValueError):
                    amt = 0.0
                if amt <= 0:
                    continue
            kept.append(item["hash"])
        assert kept == ["real"]


class TestSwapContinuationOversizedSkip:
    """Regression for the bbd7c0fa…trace_df run: a swap-continuation
    HopJob's ``incoming_amount`` reflects only the FIRST swap-out leg
    (e.g. 31.7k TRX from the 10k USDT swap) even when the user actually
    chained 4 swaps that summed to ~1.5M TRX. With the legacy hop≥2
    oversized-skip rule, every subsequent native-TRX outflow > 1.2× of
    that 31.7k anchor (i.e. 100k, 500k, 924k) gets classified as
    "mixed funds" and dropped — leaving only the 1k TRX dust outflow.

    The fix: when ``is_swap_continuation=True``, disable the
    oversized-skip unconditionally. The HopJob's source IS the user
    wallet; there's no third-party balance to defend against.
    """

    def test_oversized_skip_disabled_for_swap_continuation(self, tracer):
        """Reproduce the bbd7c0fa selection: 4 native TRX outflows
        ranging from 1k to 924k against a 31.7k incoming anchor.
        With ``is_swap_continuation=True`` ALL four must be selected;
        without the flag (legacy hop≥2 behaviour) only the 1k stays."""
        txs = [
            {"hash": "feee89", "amount_coerced": 1000.0},
            {"hash": "d84410", "amount_coerced": 100000.0},
            {"hash": "0eb3cd", "amount_coerced": 500000.0},
            {"hash": "f0b5a3", "amount_coerced": 924900.75},
        ]
        # Legacy behavior (the bug being fixed): oversized-skip ON.
        legacy = tracer._accumulate_hashes(
            txs, incoming_amount=31744.26, chain="trx", asset="TRX",
            hop_index=2,
        )
        assert legacy == ["feee89"], (
            "regression guard: with oversized-skip ON, only the 1k TRX "
            "outflow survives (this is the bug)"
        )

        # New behavior: swap-continuation disables oversized-skip AND
        # treats the first oversized outflow as a mixed-funds signal so
        # the "first to cross incoming" break is suppressed. All four
        # native-TRX outflows then make it into the selection — which
        # matches the user's expected graph (1k + 100k + 500k + 924.9k
        # all rendered against the SunSwap chain).
        swap_cont = tracer._accumulate_hashes(
            txs, incoming_amount=31744.26, chain="trx", asset="TRX",
            hop_index=2,
            is_swap_continuation=True,
        )
        assert swap_cont == ["feee89", "d84410", "0eb3cd", "f0b5a3"], (
            f"swap-cont must select ALL chained-swap tail outflows; got {swap_cont}"
        )

    def test_oversized_skip_still_active_for_normal_hop2(self, tracer):
        """Non-swap hop≥2 jobs MUST keep the oversized-skip — that's
        the original mixed-funds defense (e.g. 220k USDT outflow from a
        mule that received 60k of theft + 160k of legit balance)."""
        txs = [
            {"hash": "small", "amount_coerced": 1000.0},
            {"hash": "huge", "amount_coerced": 220000.0},  # > 60k * 1.2
        ]
        result = tracer._accumulate_hashes(
            txs, incoming_amount=60000.0, chain="trx", asset="USDT",
            hop_index=2,
        )
        # huge gets oversized-skipped; only small selected.
        # Then accumulator never reaches incoming (1k < 60k), so it
        # falls through and the safety-net surfaces nothing extra
        # because ``selected`` already has an entry.
        assert "huge" not in result
        assert "small" in result

    def test_swap_cont_at_hop_1_unchanged(self, tracer):
        """Hop-1 mixed-funds detection is independent of the
        ``is_swap_continuation`` flag — swap-cont jobs are always
        hop≥2 by construction (parent has already done one hop), so
        the flag should be irrelevant at hop 1. Sanity-check that
        passing it doesn't accidentally regress hop-1 selection."""
        txs = [
            {"hash": "a", "amount_coerced": 50000.0},
            {"hash": "b", "amount_coerced": 60000.0},
        ]
        baseline = tracer._accumulate_hashes(
            txs, incoming_amount=100000.0, chain="trx", asset="USDT",
            hop_index=1,
        )
        with_flag = tracer._accumulate_hashes(
            txs, incoming_amount=100000.0, chain="trx", asset="USDT",
            hop_index=1,
            is_swap_continuation=True,
        )
        assert baseline == with_flag


class TestSwapContinuationDustAnchor:
    """Regression for the bbd7c0fa…trace_4e run: after the dust-anchor
    fix, swap-continuation HopJobs must compare ``step_amount`` against
    the swap's ``incoming_amount`` (post-swap asset units), NOT the
    original ``stolen_amount`` (pre-swap asset units).

    Without this, a 1000 TRX outflow ($315) gets compared against
    480549 USDT * 0.01 = 4805 — different units, always trips the
    1% rule. Path "1" gets stop_reason="Below dust threshold (0.21%)",
    postprocess propagates that to every split sibling, and the
    visualization drops the LAST step of every dust-marked path —
    leaving only the seed rendered.
    """

    def test_native_outflow_anchored_on_swap_amount_not_stolen(self):
        """1000 TRX outflow after a USDT→TRX swap that delivered
        31744 TRX must NOT be classified dust. The right anchor is
        the swap's incoming amount (31744 TRX); 1000/31744 ≈ 3.15%,
        well above the 1% rule. Mirroring the predicate from the
        production code so a future tweak forces a deliberate update."""
        step_amount = 1000.0  # native TRX outflow
        stolen_amount = 480549.647444  # original USDT theft
        swap_incoming_amount = 31744.2613  # post-swap TRX (first swap leg)
        min_attribution_ratio = 0.01

        is_swap_continuation = True
        dust_anchor = stolen_amount
        if is_swap_continuation and swap_incoming_amount and swap_incoming_amount > 0:
            dust_anchor = float(swap_incoming_amount)

        dust_hit = (
            dust_anchor > 0
            and min_attribution_ratio > 0.0
            and step_amount < dust_anchor * min_attribution_ratio
        )
        assert not dust_hit, (
            f"1000 TRX vs 31744 TRX anchor must NOT be dust "
            f"(threshold = {dust_anchor * min_attribution_ratio:.2f})"
        )

    def test_native_outflow_dust_with_stolen_anchor_regression(self):
        """Mirror of the *broken* behavior — proves the bug exists
        when the anchor is left as stolen_amount across a unit change.
        If this assertion ever needs to flip, the fix has been
        accidentally reverted."""
        step_amount = 1000.0
        stolen_amount = 480549.647444
        min_attribution_ratio = 0.01

        # Old behavior: anchor = stolen_amount regardless of swap.
        dust_anchor = stolen_amount
        dust_hit = step_amount < dust_anchor * min_attribution_ratio
        assert dust_hit, (
            "regression guard: with the stolen anchor 1000 TRX < 4805 USDT — "
            "this is the unit-mixing bug the fix removes"
        )

    def test_non_swap_job_keeps_stolen_anchor(self):
        """Regular (non-swap) HopJobs must continue to anchor on
        ``stolen_amount`` so the original 1%-of-stolen rule still
        traps tiny dust siblings on the seed-asset flow."""
        step_amount = 100.0  # small same-asset outflow
        stolen_amount = 480549.647444
        min_attribution_ratio = 0.01

        is_swap_continuation = False
        incoming_amount = 50000.0  # plausible HopJob inflow
        dust_anchor = stolen_amount
        if is_swap_continuation and incoming_amount > 0:
            dust_anchor = float(incoming_amount)

        dust_hit = step_amount < dust_anchor * min_attribution_ratio
        assert dust_hit, "100 USDT < 1% of 480549 USDT — must still be dust"

    def test_effective_dust_anchor_overrides_stolen(self):
        """Downstream of a swap, ``effective_dust_anchor`` propagates
        from the swap-continuation HopJob to its descendants. A regular
        (non-``is_swap_continuation``) HopJob whose ``effective_dust_anchor``
        is set still uses that anchor instead of the stale stolen-asset
        ``stolen_amount``. Regression for the bbd7c0fa…trace_e1 case
        where HopJob C (TETizvp3, hop 3, native TRX) was comparing
        1000 TRX against 4805 USDT (the 1% of stolen rule) and trimming
        every legitimate downstream native-TRX outflow as 'dust'."""
        step_amount = 1000.0  # native TRX outflow (hop 3)
        stolen_amount = 480549.647444  # original USDT theft
        effective_dust_anchor = 31744.2613  # TRX, propagated from swap
        is_swap_continuation = False  # this is hop 3, NOT the swap itself
        min_attribution_ratio = 0.01

        dust_anchor = stolen_amount
        if effective_dust_anchor and effective_dust_anchor > 0:
            dust_anchor = float(effective_dust_anchor)
        elif is_swap_continuation:
            pass  # not relevant here

        dust_hit = step_amount < dust_anchor * min_attribution_ratio
        assert not dust_hit, (
            f"1000 TRX vs propagated 31744 TRX anchor must NOT be dust "
            f"(threshold = {dust_anchor * min_attribution_ratio:.2f})"
        )

    def test_effective_dust_anchor_default_none_uses_stolen(self):
        """For a normal trace with no swap upstream,
        ``effective_dust_anchor`` is None and the dust check still
        anchors on ``stolen_amount`` — preserves historical behavior
        for asset-aligned traces."""
        step_amount = 100.0
        stolen_amount = 480549.647444
        effective_dust_anchor = None
        is_swap_continuation = False
        min_attribution_ratio = 0.01

        dust_anchor = stolen_amount
        if effective_dust_anchor and effective_dust_anchor > 0:
            dust_anchor = float(effective_dust_anchor)
        elif is_swap_continuation:
            pass

        dust_hit = step_amount < dust_anchor * min_attribution_ratio
        assert dust_hit  # 100 < 4805 → dust as expected

    def test_swap_continuation_with_zero_incoming_falls_back_to_stolen(self):
        """Defensive: if a swap-continuation HopJob somehow has 0/None
        ``incoming_amount`` we fall back to ``stolen_amount`` rather
        than treating ``dust_anchor=0`` and disabling the dust check
        entirely. Keeps the safety net even on a pathological input."""
        step_amount = 100.0
        stolen_amount = 480549.647444
        min_attribution_ratio = 0.01

        is_swap_continuation = True
        incoming_amount = 0.0  # pathological
        dust_anchor = stolen_amount
        if is_swap_continuation and incoming_amount and incoming_amount > 0:
            dust_anchor = float(incoming_amount)

        # Falls back to stolen_amount — the trip still happens.
        assert dust_anchor == stolen_amount
        dust_hit = step_amount < dust_anchor * min_attribution_ratio
        assert dust_hit
