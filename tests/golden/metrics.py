"""Similarity metrics for golden TraceResult evaluation.

Each metric is opt-in on a per-case basis (the case's ``expected.json``
names which ones apply). That way a case can assert the one or two
things that matter to it — ``terminal_addresses`` for a CEX-ending case,
``chains`` for a cross-chain bridge case — without forcing every case to
supply every field.

All metrics return ``(ok: bool, detail: str)`` tuples like the per-prompt
metrics, for consistent failure messages.
"""
from __future__ import annotations

from typing import Any


def terminal_addresses_match(result: dict, expected: dict) -> tuple[bool, str]:
    """Every ``expected.terminal_addresses`` must be a terminal in the trace.

    Terminals are the last step's ``to`` on each path. We check set
    containment: the case lists what *must* appear; the trace may have
    more paths/terminals, which is fine.
    """
    want = expected.get("terminal_addresses")
    if not want:
        return True, "terminal_addresses: skipped"
    got = set()
    for path in result.get("paths") or []:
        steps = path.get("steps") or []
        if steps:
            got.add(steps[-1].get("to") or steps[-1].get("to_address"))
    want_set = set(want)
    missing = want_set - got
    ok = not missing
    return ok, f"terminal_addresses: missing={sorted(missing)} got={sorted(got)}"


def path_count_within(result: dict, expected: dict) -> tuple[bool, str]:
    """``expected.paths_min`` / ``expected.paths_max`` bracket the path count."""
    lo = expected.get("paths_min")
    hi = expected.get("paths_max")
    if lo is None and hi is None:
        return True, "path_count: skipped"
    actual = len(result.get("paths") or [])
    if lo is not None and actual < lo:
        return False, f"path_count: {actual} < min {lo}"
    if hi is not None and actual > hi:
        return False, f"path_count: {actual} > max {hi}"
    return True, f"path_count: {actual} within [{lo}, {hi}]"


def traced_amount_within(result: dict, expected: dict) -> tuple[bool, str]:
    """Check ``trace_stats.total_traced_amount`` against an expected value.

    ``expected.traced_amount`` is the target; ``expected.traced_tolerance``
    is the allowed relative deviation (default 5%).
    """
    target = expected.get("traced_amount")
    if target is None:
        return True, "traced_amount: skipped"
    tol = float(expected.get("traced_tolerance", 0.05))
    stats = result.get("trace_stats") or {}
    actual = stats.get("total_traced_amount")
    if actual is None:
        return False, "traced_amount: trace_stats.total_traced_amount is missing"
    deviation = abs(float(actual) - float(target)) / max(float(target), 1.0)
    ok = deviation <= tol
    return ok, f"traced_amount: actual={actual} target={target} deviation={deviation:.3f} (tol={tol})"


def chain_coverage(result: dict, expected: dict) -> tuple[bool, str]:
    """``expected.chains`` must all appear in ``case_meta.chains``."""
    want = expected.get("chains")
    if not want:
        return True, "chains: skipped"
    got = set(result.get("case_meta", {}).get("chains") or [])
    missing = set(want) - got
    ok = not missing
    return ok, f"chains: missing={sorted(missing)} got={sorted(got)}"


def entity_role_contains(result: dict, expected: dict) -> tuple[bool, str]:
    """For each (address, role) pair in ``expected.entity_roles``, the
    trace must contain an entity at that address with that role."""
    want = expected.get("entity_roles")
    if not want:
        return True, "entity_roles: skipped"
    by_addr = {e.get("address"): e for e in result.get("entities") or []}
    mismatches = []
    for addr, role in want.items():
        actual = by_addr.get(addr)
        if actual is None:
            mismatches.append(f"{addr[:10]}…: absent")
        elif actual.get("role") != role:
            mismatches.append(f"{addr[:10]}…: want={role} got={actual.get('role')}")
    ok = not mismatches
    return ok, f"entity_roles: mismatches={mismatches}"


ALL_METRICS = [
    terminal_addresses_match,
    path_count_within,
    traced_amount_within,
    chain_coverage,
    entity_role_contains,
]


def evaluate(result: Any, expected: dict) -> list[tuple[bool, str]]:
    """Run every metric; return all ``(ok, detail)`` results. Metrics that
    don't apply to the case (expected field missing) short-circuit to ok."""
    # Accept either a pydantic TraceResult or a plain dict.
    payload = result.model_dump(by_alias=True) if hasattr(result, "model_dump") else result
    return [m(payload, expected) for m in ALL_METRICS]
