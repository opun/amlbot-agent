"""CLI to run the per-prompt eval suite and emit a machine-readable report.

Usage::

    .venv/bin/python -m agent.eval_prompts                     # all prompts, recorded
    .venv/bin/python -m agent.eval_prompts --prompt hop_classifier
    AGENT_EVAL_LIVE=1 .venv/bin/python -m agent.eval_prompts --live
    .venv/bin/python -m agent.eval_prompts --override-model gpt-5
    .venv/bin/python -m agent.eval_prompts --override-effort high

Output is a single JSON object on stdout summarizing each case and
aggregate pass rate. For A/B comparisons, run this twice with different
overrides and diff the outputs externally (``jq`` or
``tests/prompts/metrics.py``).

Kept as a separate tool rather than a pytest hook because operators want
to invoke it in CI dashboards and compare builds — pytest output is not
the right shape for that.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from tests.prompts.harness import Case, discover_cases, is_live_mode, run_case
from tests.prompts.metrics import evaluate_case

PROMPT_ROOT = Path(__file__).resolve().parents[2] / "tests" / "prompts"


def _discover_all(prompt_filter: str | None) -> list[tuple[str, Case]]:
    out: list[tuple[str, Case]] = []
    for child in sorted(PROMPT_ROOT.iterdir()) if PROMPT_ROOT.exists() else []:
        if not child.is_dir() or child.name in {"__pycache__"} or child.name.startswith("_"):
            continue
        if (child / "__init__.py").is_file() is False and not (child / "cases").exists():
            continue
        if prompt_filter and child.name != prompt_filter:
            continue
        for case in discover_cases(child):
            out.append((child.name, case))
    return out


async def _run_one(case: Case) -> dict:
    result = await run_case(case)
    parsed = result.parsed or {}
    outcomes = evaluate_case(case.prompt_name, parsed, case.expected)
    failures = [detail for ok, detail in outcomes if not ok]
    return {
        "name": case.name,
        "prompt": case.prompt_name,
        "version": case.prompt_version,
        "model": result.model,
        "family": result.family,
        "reasoning_effort": result.reasoning_effort,
        "from_replay": result.from_replay,
        "ok": not failures,
        "metric_results": [{"ok": ok, "detail": detail} for ok, detail in outcomes],
        "usage": result.usage,
        "latency_ms": result.latency_ms,
    }


async def _run_all(cases: list[tuple[str, Case]]) -> list[dict]:
    rows = []
    for _, case in cases:
        try:
            rows.append(await _run_one(case))
        except Exception as exc:
            rows.append({
                "name": case.name,
                "prompt": case.prompt_name,
                "ok": False,
                "error": str(exc),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", help="Run only cases for this prompt directory name")
    ap.add_argument("--live", action="store_true", help="Force live mode (equivalent to AGENT_EVAL_LIVE=1)")
    ap.add_argument("--override-model", help="Override the model_default in frontmatter for every case")
    ap.add_argument("--override-effort", help="Override reasoning_effort for every case (reasoning-family only)")
    args = ap.parse_args(argv)

    if args.live:
        os.environ["AGENT_EVAL_LIVE"] = "1"

    cases = _discover_all(args.prompt)
    if not cases:
        print(json.dumps({"error": "no cases discovered", "prompt_filter": args.prompt}), file=sys.stderr)
        return 2

    # Apply overrides by mutating the in-memory Case dataclass. The files
    # on disk aren't modified — overrides are a runtime experiment knob.
    for _, case in cases:
        if args.override_model:
            case.model = args.override_model
        if args.override_effort:
            case.reasoning_effort = args.override_effort

    rows = asyncio.run(_run_all(cases))
    total = len(rows)
    passed = sum(1 for r in rows if r.get("ok"))
    report = {
        "live": is_live_mode(),
        "overrides": {
            "model": args.override_model,
            "reasoning_effort": args.override_effort,
        },
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "cases": rows,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if total == passed else 1


if __name__ == "__main__":
    sys.exit(main())
