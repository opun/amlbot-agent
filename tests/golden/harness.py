"""End-to-end (golden) eval harness.

A golden case is a directory like::

    tests/golden/case_xxx/
        input.json        # TracerConfig fields
        recording.jsonl   # all tool + LLM calls recorded from a real run
        expected.json     # subset of TraceResult to compare against
        metadata.yaml     # optional free-form notes (not parsed)

Execution is always replay-based: we never call the real MCP server or
OpenAI from a golden test. The recording supplies every tool + LLM
result, and a :class:`_ReplayOnlyTracer` subclass fails loudly if the
runtime ever tries to execute a tool that wasn't recorded.

Recordings are **produced** by running the live tracer with
``AGENT_RECORD=1`` on a real case, then copying the ``.jsonl`` into the
case directory. Any new golden case should go through that loop rather
than being hand-written — maintaining synthetic recordings is a tax we
don't need to pay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.base_tracer import BaseTracer
from agent.models import TracerConfig, TraceResult
from agent.recorder import TraceRecorder


class _ReplayOnlyTracer(BaseTracer):
    """Tracer that refuses to reach out to a live MCP backend.

    Every tool call must come from the recording. If anything asks for a
    tool that isn't in the replay index, we raise — that surfaces gaps
    in the recording immediately instead of making a silent live call.
    """

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError(
            f"replay-only tracer: attempted to execute tool {tool_name!r} — "
            "recording is missing an event or the code path diverged from the recording"
        )


@dataclass
class GoldenCase:
    dir: Path
    name: str
    config: TracerConfig
    recording_path: Path
    expected: dict[str, Any]

    @property
    def is_runnable(self) -> bool:
        return self.recording_path.exists()


def load_case(case_dir: Path) -> GoldenCase:
    input_path = case_dir / "input.json"
    expected_path = case_dir / "expected.json"
    recording_path = case_dir / "recording.jsonl"

    import json
    config_raw = json.loads(input_path.read_text(encoding="utf-8"))
    config = TracerConfig(**config_raw)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    return GoldenCase(
        dir=case_dir,
        name=case_dir.name,
        config=config,
        recording_path=recording_path,
        expected=expected,
    )


def discover_cases(root: Path) -> list[GoldenCase]:
    if not root.exists():
        return []
    cases = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        # A directory is a case iff it has both input.json + expected.json.
        if (child / "input.json").exists() and (child / "expected.json").exists():
            cases.append(load_case(child))
    return cases


async def run_golden_case(case: GoldenCase) -> TraceResult:
    """Replay a golden case and return the resulting ``TraceResult``."""
    recorder = TraceRecorder.for_replay(case.recording_path)
    tracer = _ReplayOnlyTracer()
    tracer.recorder = recorder
    try:
        return await tracer.trace(case.config)
    finally:
        recorder.close()
