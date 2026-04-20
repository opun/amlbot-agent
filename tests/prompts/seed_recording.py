"""Generate a ``<case>.recording.jsonl`` file from a case's ``synthetic_response``.

Usage (from repo root):

    .venv/bin/python -m tests.prompts.seed_recording \
        tests/prompts/hop_classifier/cases/001_binance_deposit_eth.json

The case JSON must contain a ``synthetic_response`` key (dict or string).
We compute the exact ``input_hash`` the harness will compute at replay
time, write one ``llm_call`` event to the sibling recording file, and
stop. Re-running is idempotent — the file is overwritten.

This is seeding logic, not test code. The harness in ``harness.py`` is
what actually consumes the recording.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from agent.recorder import messages_hash
from tests.prompts.harness import _load_system, load_case


def _build_event(case_path: Path) -> dict:
    raw = json.loads(case_path.read_text(encoding="utf-8"))
    if "synthetic_response" not in raw:
        raise SystemExit(f"{case_path} has no 'synthetic_response' — cannot seed recording")

    case = load_case(case_path)
    user_content = (
        case.user if isinstance(case.user, str)
        else json.dumps(case.user, indent=2, ensure_ascii=False, default=str)
    )
    messages = [
        {"role": "system", "content": case.system},
        {"role": "user", "content": user_content},
    ]
    input_hash = messages_hash(messages)

    synthetic = raw["synthetic_response"]
    if isinstance(synthetic, dict):
        content = json.dumps(synthetic, ensure_ascii=False)
        parsed = synthetic
    else:
        content = str(synthetic)
        parsed = None

    return {
        "event_id": 1,
        "ts_ns": time.time_ns(),
        "event_type": "llm_call",
        "decision_id": f"{case.prompt_name}_seed_{uuid.uuid4().hex[:8]}",
        "prompt_name": case.prompt_name,
        "prompt_version": case.prompt_version,
        "model": case.model,
        "family": "reasoning",  # harness fills from model_spec if missing on replay
        "reasoning_effort": case.reasoning_effort,
        "input_hash": input_hash,
        "content": content,
        "parsed": parsed,
        "usage": {
            "input_tokens": len(user_content) // 4,
            "output_tokens": len(content) // 4,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        },
        "latency_ms": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", type=Path)
    args = ap.parse_args()
    for case_path in args.cases:
        event = _build_event(case_path)
        rec_path = case_path.with_suffix(".recording.jsonl")
        rec_path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {rec_path}  (input_hash={event['input_hash'][:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
