"""Per-prompt eval harness.

A case is a JSON file that describes:
  * the prompt to run (`prompt_name`),
  * the model + optional reasoning effort,
  * a ``system`` prompt (either inline, via ``system_file``, or looked up
    on disk under ``src/agent/prompts/``),
  * the user input to feed in, and
  * the expected output fields (a *subset* — we only assert what the case
    cares about, so future prompt evolution doesn't break unrelated
    cases).

Two execution modes
-------------------
* **Live** (``AGENT_EVAL_LIVE=1`` in env): calls the real OpenAI API via
  ``call_llm``. Requires ``OPENAI_API_KEY``. Use this for the actual
  regression gate.
* **Recorded** (default): reads a sibling ``<case>.recording.jsonl`` if
  present and replays it via ``TraceRecorder.for_replay``. Lets CI / dev
  laptops validate the harness + parsing code without LLM access.
* **Skip**: no recording and no live credentials → pytest skip.

Keeping the harness here (rather than in ``src/``) so tests-only
dependencies stay out of the runtime package.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import ModelConfig
from agent.llm_client import LLMResult, call_llm
from agent.model_registry import resolve_model
from agent.prompt_loader import load_prompt
from agent.recorder import TraceRecorder

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "src" / "agent" / "prompts"

# Default model per prompt family. Keeps cases terse — most don't need to
# pick a model explicitly.
DEFAULT_MODEL_FOR_PROMPT = {
    "hop_classifier": lambda: ModelConfig.SELECTOR_MODEL,
    "hop_selector": lambda: ModelConfig.SELECTOR_MODEL,
    "validator": lambda: ModelConfig.VALIDATOR_MODEL,
}

# Default system-prompt file per prompt name. Keeping the mapping in one
# place means adding a new prompt is: drop a file in prompts/, add a line
# here, add a subdir in tests/prompts/.
DEFAULT_SYSTEM_FILE = {
    "hop_classifier": "trace_hop_classifier.md",
    "hop_selector": "trace_hop_selector.md",
    "validator": "trace_validator.md",
}


@dataclass
class Case:
    path: Path
    name: str
    prompt_name: str
    prompt_version: str
    model: str
    reasoning_effort: str | None
    system: str
    user: Any
    expected: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    response_format: str = "json"
    max_output_tokens: int | None = None

    @property
    def recording_path(self) -> Path:
        return self.path.with_suffix(".recording.jsonl")


def _load_system(case_dir: Path, data: dict[str, Any], prompt_name: str) -> str:
    """Return the system-prompt body the case should run against.

    Frontmatter is stripped (via ``load_prompt``) so the hash used here
    matches exactly what ``BaseTracer`` sends at runtime — this keeps the
    seeded recording valid across frontmatter edits.
    """
    if "system" in data:
        return data["system"]
    system_file = data.get("system_file") or DEFAULT_SYSTEM_FILE.get(prompt_name)
    if not system_file:
        raise ValueError(f"No system prompt for {prompt_name}; set 'system' or 'system_file'")
    # Case-relative system_file takes precedence; fall back to runtime prompts dir.
    candidates = [case_dir / system_file, PROMPTS_DIR / system_file]
    for cand in candidates:
        if cand.exists():
            return load_prompt(cand, name_default=prompt_name).body
    raise FileNotFoundError(f"system prompt {system_file!r} not found in {candidates}")


def load_case(path: Path) -> Case:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompt_name = data["prompt_name"]
    model = data.get("model") or DEFAULT_MODEL_FOR_PROMPT[prompt_name]()
    return Case(
        path=path,
        name=data.get("name") or path.stem,
        prompt_name=prompt_name,
        prompt_version=data.get("prompt_version", "v1"),
        model=model,
        reasoning_effort=data.get("reasoning_effort"),
        system=_load_system(path.parent, data, prompt_name),
        user=data["input"],
        expected=data["expected"],
        metrics=data.get("metrics", {}),
        response_format=data.get("response_format", "json"),
        max_output_tokens=data.get("max_output_tokens"),
    )


def discover_cases(prompt_dir: Path) -> list[Case]:
    cases_dir = prompt_dir / "cases"
    if not cases_dir.exists():
        return []
    return [load_case(p) for p in sorted(cases_dir.glob("*.json")) if not p.name.endswith(".recording.jsonl")]


def is_live_mode() -> bool:
    return os.environ.get("AGENT_EVAL_LIVE", "").lower() in ("1", "true", "yes")


async def run_case(case: Case) -> LLMResult:
    """Run a case via ``call_llm`` in whichever mode is configured.

    * Live: real API.
    * Recorded: read the sibling ``<case>.recording.jsonl``.

    Raises ``RuntimeError`` if neither is possible so pytest can decide
    whether to fail or skip.
    """
    model_spec = resolve_model(
        case.model,
        reasoning_effort=case.reasoning_effort if case.reasoning_effort and resolve_model(case.model).is_reasoning else None,
    )

    if is_live_mode():
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("AGENT_EVAL_LIVE=1 but OPENAI_API_KEY is not set")
        # Lazy import to avoid hard-requiring a real openai client at
        # harness import time (tests without OPENAI_API_KEY should still
        # be importable).
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        return await call_llm(
            openai_client=client,
            model_spec=model_spec,
            prompt_name=case.prompt_name,
            prompt_version=case.prompt_version,
            system=case.system,
            user=case.user,
            response_format=case.response_format,  # type: ignore[arg-type]
            max_output_tokens=case.max_output_tokens,
        )

    # Recorded mode
    if not case.recording_path.exists():
        raise RuntimeError(
            f"No recording at {case.recording_path.name} and AGENT_EVAL_LIVE is off"
        )
    recorder = TraceRecorder.for_replay(case.recording_path)
    try:
        # ``openai_client=None`` is fine — replay mode skips the network
        # and we never dereference the client. Pass a typed placeholder
        # so mypy doesn't complain in stricter configurations.
        return await call_llm(
            openai_client=None,  # type: ignore[arg-type]
            model_spec=model_spec,
            prompt_name=case.prompt_name,
            prompt_version=case.prompt_version,
            system=case.system,
            user=case.user,
            response_format=case.response_format,  # type: ignore[arg-type]
            max_output_tokens=case.max_output_tokens,
            recorder=recorder,
        )
    finally:
        recorder.close()
