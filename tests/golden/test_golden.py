"""Parametrized golden end-to-end tests.

Each case directory under ``tests/golden/`` becomes one test case. The
case's ``recording.jsonl`` drives a replay-only tracer; the resulting
``TraceResult`` is diff'd against ``expected.json`` via
:mod:`tests.golden.metrics`.

Cases without a recording (e.g. freshly-seeded directories) are
pytest-skip'd rather than erroring, so the test file stays green while
recordings are captured from live runs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.golden.harness import GoldenCase, discover_cases, run_golden_case
from tests.golden.metrics import evaluate

GOLDEN_ROOT = Path(__file__).parent
CASES = discover_cases(GOLDEN_ROOT)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[c.name for c in CASES] if CASES else [],
)
def test_golden_case(case: GoldenCase):
    if not case.is_runnable:
        pytest.skip(
            f"No recording at {case.recording_path.name}. "
            f"Capture one with AGENT_RECORD=1 and drop it in {case.dir}."
        )
    result = asyncio.run(run_golden_case(case))
    outcomes = evaluate(result, case.expected)
    failures = [detail for ok, detail in outcomes if not ok]
    assert not failures, (
        f"golden eval failed for {case.name}:\n  " + "\n  ".join(failures)
    )
