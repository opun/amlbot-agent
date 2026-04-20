"""Load prompt files with YAML frontmatter.

Each runtime prompt under ``src/agent/prompts/`` may carry a YAML header::

    ---
    name: hop_classifier
    version: v2
    model_default: gpt-5-mini
    reasoning_effort: medium
    max_output_tokens: 250
    ---

    # Hop Classifier ...
    <body>

``load_prompt`` returns a :class:`PromptSpec` with the parsed metadata
and the body text (without the frontmatter). A file with no frontmatter
still loads — ``version`` falls back to ``"v1"`` and everything else is
``None``. That lets us adopt frontmatter gradually.

Why its own module? The tracer, the eval harness, and any future A/B
tooling all need the same parsing — one place means one bug to fix when
the format evolves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    body: str
    model_default: str | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    extra: dict = field(default_factory=dict)


_FRONTMATTER_MARKER = "---"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return ``(frontmatter_dict, body)``. Missing header returns ``({}, text)``."""
    if not text.startswith(_FRONTMATTER_MARKER + "\n") and not text.startswith(_FRONTMATTER_MARKER + "\r\n"):
        return {}, text
    # Find the closing marker on its own line.
    lines = text.splitlines(keepends=True)
    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].rstrip() == _FRONTMATTER_MARKER:
            close_idx = i
            break
    if close_idx is None:
        # Open frontmatter with no close — treat as body (don't silently drop content).
        return {}, text
    header_block = "".join(lines[1:close_idx])
    body = "".join(lines[close_idx + 1:]).lstrip("\n")
    parsed = yaml.safe_load(header_block) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


def load_prompt(path: Path, *, name_default: str | None = None) -> PromptSpec:
    """Read and parse a prompt file. Raises ``FileNotFoundError`` if missing."""
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(raw)

    known_keys = {"name", "version", "model_default", "reasoning_effort", "max_output_tokens"}
    extra = {k: v for k, v in meta.items() if k not in known_keys}

    return PromptSpec(
        name=str(meta.get("name") or name_default or path.stem),
        version=str(meta.get("version") or "v1"),
        body=body,
        model_default=(str(meta["model_default"]) if meta.get("model_default") else None),
        reasoning_effort=(str(meta["reasoning_effort"]) if meta.get("reasoning_effort") else None),
        max_output_tokens=(int(meta["max_output_tokens"]) if meta.get("max_output_tokens") is not None else None),
        extra=extra,
    )
