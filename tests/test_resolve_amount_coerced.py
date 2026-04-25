"""Regression: ``_resolve_amount`` must NOT re-normalize ``amount_coerced``.

The TRX-USDT replay trace ``20260425T060645Z_trx_USDT_2e2e0822d7…`` ended
abruptly at recording event 8 — the same place every time, across two
different runs. Root cause:

  1. Seed (``2e2e08…``) extracted theft_amount = 1.08M USDT correctly via
     ``token_transfers`` → ``stolen_amount = 1.08e6``.
  2. Hop 1 (TEwB298) classifier said "intermediate, continue tracing".
  3. ``_fetch_outgoing_txs`` populated ``all_txs_map[92ad781e]`` with
     ``amount_coerced = 1080200`` — already in display USDT.
  4. ``_resolve_amount`` for hop 2 swapped to ``amount_coerced`` and
     pushed it through ``_normalize_amount``.
  5. ``_normalize_amount`` saw ``val = 1.08e6 >= safeguard = 10^6``
     for USDT (unit=6) and *divided again* by 10^6, returning **1.0802**.
  6. The per-step dust check ``step_amount < stolen_amount * 0.01`` fired
     (``1.0802 < 10802``) → path stopped with
     ``stop_reason = "Below dust threshold"``, no HopJob for TXag1jh
     pushed, scheduler heap empty → trace ended.

The fix: treat ``amount_coerced`` as display (per the API contract,
which the ``_normalize_amount`` docstring already documents) and skip
the re-normalization step. Only the raw ``amount`` field gets
normalized.

This regression matters for any 6-decimal token (USDT, USDC, USDP, etc.)
moving ≥ $1M — it's a hard scaling cliff at exactly the values operators
care about most.
"""
from __future__ import annotations

from agent.base_tracer import BaseTracer


class _Stub(BaseTracer):
    async def execute_tool(self, tool_name, arguments):
        raise NotImplementedError

    def _get_client(self):
        raise NotImplementedError


def _tracer() -> BaseTracer:
    """Concrete tracer instance with no MCP client — ``_resolve_amount``
    is a pure method, so the abstract MCP hooks are irrelevant."""
    return _Stub()


def test_resolve_amount_treats_amount_coerced_as_display_for_trx_usdt():
    """1.08M USDT on TRX must not be wrongly divided to 1.08.

    Reproduces the exact scenario that stopped the
    ``2e2e0822…7c8fc03`` trace at hop 2.
    """
    tracer = _tracer()
    all_txs_map = {
        "92ad781e": {
            "hash": "92ad781e",
            # ``amount_coerced`` is already in display USDT — 1080200 USDT.
            "amount_coerced": 1080200,
            "token_id": 9,
            "type": "token",
        }
    }
    # The raw amount we'd get from token_transfers (1.08M USDT × 10^6 = 1.08e12).
    raw_amount = 1080200000000

    resolved = tracer._resolve_amount(
        tx_hash="92ad781e",
        amount=raw_amount,
        chain="trx",
        all_txs_map=all_txs_map,
        asset="USDT",
    )

    assert resolved == 1080200.0, (
        f"Expected 1.08M USDT (1080200) — got {resolved}. "
        "Likely re-normalized via _normalize_amount and divided by 10^6 again."
    )


def test_resolve_amount_treats_low_value_amount_coerced_as_display():
    """0.14 ETH (display) was already the only case the docstring
    promised, and it must keep working — make sure the fix didn't
    accidentally bypass it the wrong way."""
    tracer = _tracer()
    all_txs_map = {
        "abc": {
            "hash": "abc",
            "amount_coerced": 0.14,  # 0.14 ETH, display
            "token_id": 0,
        }
    }
    resolved = tracer._resolve_amount(
        tx_hash="abc",
        amount=140000000000000000,  # 0.14 ETH raw (10^17)
        chain="eth",
        all_txs_map=all_txs_map,
        asset="ETH",
    )
    assert resolved == 0.14


def test_resolve_amount_falls_back_to_raw_amount_when_no_coerced():
    """When ``amount_coerced`` is missing, the raw ``amount`` is still
    normalized through ``_normalize_amount`` (the original code path)."""
    tracer = _tracer()
    all_txs_map = {
        "raw-only": {
            "hash": "raw-only",
            # No amount_coerced.
            "amount": 1080200000000,  # raw USDT × 10^6
            "token_id": 9,
        }
    }
    resolved = tracer._resolve_amount(
        tx_hash="raw-only",
        amount=999,  # ignored when tx_hash is in the map
        chain="trx",
        all_txs_map=all_txs_map,
        asset="USDT",
    )
    assert resolved == 1080200.0


def test_resolve_amount_falls_back_to_arg_when_tx_hash_unknown():
    """When the tx isn't in all_txs_map, normalize the function arg."""
    tracer = _tracer()
    resolved = tracer._resolve_amount(
        tx_hash="not-in-map",
        amount=1080200000000,
        chain="trx",
        all_txs_map={},
        asset="USDT",
    )
    assert resolved == 1080200.0


def test_resolve_amount_handles_garbage_amount_coerced_gracefully():
    """A non-numeric ``amount_coerced`` shouldn't crash — fall back to
    raw amount, then to the function arg."""
    tracer = _tracer()
    all_txs_map = {
        "broken": {
            "hash": "broken",
            "amount_coerced": "not-a-number",
            "amount": 1080200000000,
            "token_id": 9,
        }
    }
    resolved = tracer._resolve_amount(
        tx_hash="broken",
        amount=0,
        chain="trx",
        all_txs_map=all_txs_map,
        asset="USDT",
    )
    assert resolved == 1080200.0
