"""Parametrized per-case tests for the validator prompt."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.prompts.conftest import skip_if_case_not_runnable
from tests.prompts.harness import Case, discover_cases, run_case
from tests.prompts.metrics import evaluate_case

PROMPT_DIR = Path(__file__).parent
CASES = discover_cases(PROMPT_DIR)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[c.name for c in CASES] if CASES else [],
)
def test_validator_case(case: Case):
    skip_if_case_not_runnable(case)

    result = asyncio.run(run_case(case))
    parsed = result.parsed or {}
    outcomes = evaluate_case("validator", parsed, case.expected)

    failures = [detail for ok, detail in outcomes if not ok]
    assert not failures, (
        f"validator eval failed for {case.name}:\n  " + "\n  ".join(failures)
        + f"\nParsed output type: {type(parsed).__name__}"
    )
