"""Unit tests for the YAML frontmatter prompt loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.prompt_loader import PromptSpec, load_prompt


def _write(tmp_path: Path, content: str, name: str = "p.md") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_full_frontmatter(tmp_path):
    p = _write(tmp_path, """---
name: hop_classifier
version: v3
model_default: gpt-5-mini
reasoning_effort: high
max_output_tokens: 400
---

# body line 1
body line 2
""")
    spec = load_prompt(p)
    assert spec.name == "hop_classifier"
    assert spec.version == "v3"
    assert spec.model_default == "gpt-5-mini"
    assert spec.reasoning_effort == "high"
    assert spec.max_output_tokens == 400
    assert spec.body.startswith("# body line 1")
    assert "body line 2" in spec.body


def test_no_frontmatter_uses_defaults(tmp_path):
    p = _write(tmp_path, "# just a body\n", name="plain.md")
    spec = load_prompt(p)
    assert spec.version == "v1"
    assert spec.model_default is None
    assert spec.reasoning_effort is None
    assert spec.max_output_tokens is None
    assert spec.body.startswith("# just a body")
    assert spec.name == "plain"  # derived from filename


def test_open_frontmatter_treated_as_body(tmp_path):
    # A stray ``---`` at the top with no close marker must not eat the body.
    p = _write(tmp_path, "---\nnot really frontmatter\n# heading\n")
    spec = load_prompt(p)
    assert spec.body.startswith("---")  # body preserved
    assert spec.version == "v1"


def test_extra_keys_captured(tmp_path):
    p = _write(tmp_path, """---
name: x
version: v1
custom_field: hello
---

body
""")
    spec = load_prompt(p)
    assert spec.extra == {"custom_field": "hello"}


def test_runtime_prompts_parse(tmp_path):
    """The three live prompt files must parse without error."""
    runtime = Path(__file__).resolve().parents[1] / "src" / "agent" / "prompts"
    for fname in ("trace_hop_classifier.md", "trace_hop_selector.md", "trace_validator.md"):
        spec = load_prompt(runtime / fname)
        assert spec.name, f"{fname} missing name"
        assert spec.version, f"{fname} missing version"
        assert spec.body.strip(), f"{fname} body empty after stripping frontmatter"
