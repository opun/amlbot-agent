"""Parametrized per-case tests for the hop_classifier prompt.

Each JSON file under ``cases/`` becomes one test. Live mode (OPENAI_API_KEY
set + AGENT_EVAL_LIVE=1) hits the real API; default mode replays from
``<case>.recording.jsonl`` if present, else pytest-skip.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.recorder import MissingReplayEvent
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
def test_hop_classifier_case(case: Case):
    skip_if_case_not_runnable(case)

    try:
        result = asyncio.run(run_case(case))
    except MissingReplayEvent as miss:
        # Recording was captured against an older prompt version; in
        # offline mode we can't re-hit the LLM to regenerate. Skip
        # rather than fail so a prompt iteration doesn't red-ify CI.
        pytest.skip(
            f"recording stale for {case.name} "
            f"(prompt hash changed): {miss}. Re-capture with AGENT_EVAL_LIVE=1."
        )
    parsed = result.parsed or {}
    outcomes = evaluate_case("hop_classifier", parsed, case.expected)

    failures = [detail for ok, detail in outcomes if not ok]
    assert not failures, (
        f"hop_classifier eval failed for {case.name}:\n  " + "\n  ".join(failures)
        + f"\nParsed output: {parsed}"
    )
