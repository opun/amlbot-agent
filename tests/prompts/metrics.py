"""Pure functions comparing a parsed LLM output to the case's ``expected`` block.

Each function returns an ``(ok: bool, detail: str)`` tuple. Test code
collects these, asserts, and includes ``detail`` in the failure message so
a failing eval tells you *which field* drifted.

Metrics are deliberately small and composable so cases can opt-in to
whichever subset they care about. New metrics go here, not in the runner,
so they stay testable in isolation.
"""
from __future__ import annotations

from typing import Any


def role_match(actual: dict, expected: dict) -> tuple[bool, str]:
    exp = expected.get("role")
    if exp is None:
        return True, "role: skipped"
    got = (actual or {}).get("role")
    return got == exp, f"role: expected={exp!r} got={got!r}"


def terminal_match(actual: dict, expected: dict) -> tuple[bool, str]:
    if "terminal" not in expected:
        return True, "terminal: skipped"
    exp = bool(expected["terminal"])
    got = bool((actual or {}).get("terminal"))
    return got == exp, f"terminal: expected={exp} got={got}"


def labels_jaccard(actual: dict, expected: dict, threshold: float = 0.5) -> tuple[bool, str]:
    """Jaccard similarity on labels; case-insensitive, order-insensitive.

    Expected labels are a *minimum* overlap — the model can add more,
    but the ones the case cares about must be present. Threshold of 1.0
    means strict superset; lower values allow some freedom.
    """
    exp = expected.get("labels")
    if not exp:
        return True, "labels: skipped"
    exp_set = {str(x).strip().lower() for x in exp}
    got_set = {str(x).strip().lower() for x in (actual or {}).get("labels", [])}
    if not exp_set and not got_set:
        return True, "labels: both empty"
    union = exp_set | got_set
    inter = exp_set & got_set
    score = len(inter) / len(union) if union else 0.0
    ok = score >= threshold
    return ok, f"labels jaccard={score:.2f} (threshold={threshold}) exp={sorted(exp_set)} got={sorted(got_set)}"


def selected_hashes_set_match(actual: dict, expected: dict) -> tuple[bool, str]:
    """Selector output: selected_hashes must match exactly, as a set.

    Order is ignored because selector output is logically a set. If a
    case wants to assert ordering, it should add a dedicated metric.
    """
    exp = expected.get("selected_hashes")
    if exp is None:
        return True, "selected_hashes: skipped"
    got = (actual or {}).get("selected_hashes") or []
    ok = set(exp) == set(got)
    return ok, f"selected_hashes: expected={sorted(exp)} got={sorted(got)}"


def json_shape_valid(actual: Any, expected: dict) -> tuple[bool, str]:
    """Validator cases assert the output parses as a dict with the keys
    the expected block names. Keeps contract-level regression visible
    without over-specifying content."""
    req_keys = expected.get("required_keys")
    if not req_keys:
        return True, "required_keys: skipped"
    if not isinstance(actual, dict):
        return False, f"required_keys: actual is not a dict (type={type(actual).__name__})"
    missing = [k for k in req_keys if k not in actual]
    ok = not missing
    return ok, f"required_keys: missing={missing} present={sorted(set(req_keys) & set(actual))}"


# ─── Dispatcher ────────────────────────────────────────────────────────────

# Map prompt_name → list of metric functions applied by default. Cases
# can override/opt out via ``metrics`` in the JSON (not implemented for
# the initial scaffolding — extend here when needed).
DEFAULT_METRICS_BY_PROMPT: dict[str, list] = {
    "hop_classifier": [role_match, terminal_match, labels_jaccard],
    "hop_selector": [selected_hashes_set_match],
    "validator": [json_shape_valid],
}


def evaluate_case(prompt_name: str, actual: Any, expected: dict) -> list[tuple[bool, str]]:
    """Apply the default metric suite for ``prompt_name`` and return the
    full list of ``(ok, detail)`` results."""
    metrics = DEFAULT_METRICS_BY_PROMPT.get(prompt_name, [])
    return [m(actual, expected) for m in metrics]
