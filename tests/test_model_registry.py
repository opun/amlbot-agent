"""Tests for the model registry: correct family classification,
correct payload shape, correct override behavior."""
from __future__ import annotations

import pytest

from agent.llm_client import _build_create_kwargs
from agent.model_registry import KNOWN_MODELS, ModelSpec, resolve_model


class TestResolveKnownModels:
    def test_gpt5_mini_is_reasoning(self):
        spec = resolve_model("gpt-5-mini")
        assert spec.is_reasoning
        assert spec.family == "reasoning"
        assert spec.reasoning_effort == "medium"
        assert spec.supports_temperature is False

    def test_gpt4o_is_standard(self):
        spec = resolve_model("gpt-4o")
        assert not spec.is_reasoning
        assert spec.family == "standard"
        assert spec.reasoning_effort is None
        assert spec.supports_temperature is True

    def test_o4_mini_is_reasoning(self):
        spec = resolve_model("o4-mini")
        assert spec.is_reasoning
        assert spec.reasoning_effort == "medium"


class TestInferUnknownModels:
    def test_inferred_gpt5_future_variant(self):
        # gpt-5-preview etc. — prefix match makes it reasoning.
        spec = resolve_model("gpt-5-preview-2026")
        assert spec.is_reasoning

    def test_inferred_future_o_series(self):
        spec = resolve_model("o5-mini")
        assert spec.is_reasoning
        spec2 = resolve_model("o7")
        assert spec2.is_reasoning

    def test_gpt4_variant_stays_standard(self):
        spec = resolve_model("gpt-4o-mini-next")
        assert not spec.is_reasoning

    def test_gpt35_stays_standard(self):
        spec = resolve_model("gpt-3.5-turbo")
        assert not spec.is_reasoning


class TestOverrides:
    def test_effort_override_on_reasoning(self):
        spec = resolve_model("gpt-5-mini", reasoning_effort="high")
        assert spec.reasoning_effort == "high"

    def test_effort_override_rejected_on_standard(self):
        with pytest.raises(ValueError, match="non-reasoning"):
            resolve_model("gpt-4o", reasoning_effort="high")

    def test_max_output_tokens_override(self):
        spec = resolve_model("gpt-5-mini", max_output_tokens=1024)
        assert spec.max_output_tokens == 1024


class TestPayloadShape:
    """Reasoning family must never see temperature/top_p/max_tokens. This is
    the invariant that prevents API errors when swapping between families."""

    def _messages(self):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]

    def test_reasoning_payload_excludes_temperature(self):
        spec = resolve_model("gpt-5-mini")
        kw = _build_create_kwargs(
            spec, self._messages(), response_format="text", max_output_tokens=None
        )
        assert "temperature" not in kw
        assert "top_p" not in kw
        assert "max_tokens" not in kw
        assert kw["reasoning_effort"] == "medium"

    def test_reasoning_uses_max_completion_tokens(self):
        spec = resolve_model("gpt-5-mini")
        kw = _build_create_kwargs(
            spec, self._messages(), response_format="text", max_output_tokens=500
        )
        assert kw["max_completion_tokens"] == 500
        assert "max_tokens" not in kw

    def test_standard_no_reasoning_effort(self):
        spec = resolve_model("gpt-4o")
        kw = _build_create_kwargs(
            spec, self._messages(), response_format="text", max_output_tokens=None
        )
        assert "reasoning_effort" not in kw

    def test_json_response_format_applied(self):
        spec = resolve_model("gpt-4o")
        kw = _build_create_kwargs(
            spec, self._messages(), response_format="json", max_output_tokens=None
        )
        assert kw["response_format"] == {"type": "json_object"}

    def test_json_skipped_when_unsupported(self):
        # Hand-build a spec that opts out of structured output. Prevents the
        # wrapper from forcing json mode on a model that would error.
        spec = ModelSpec(name="weird", family="standard", supports_structured_output=False)
        kw = _build_create_kwargs(
            spec, self._messages(), response_format="json", max_output_tokens=None
        )
        assert "response_format" not in kw


class TestKnownModelsCoverage:
    """Every model in KNOWN_MODELS has a sensible spec — catches typos in
    the table."""

    def test_every_reasoning_has_effort(self):
        for name, spec in KNOWN_MODELS.items():
            if spec.is_reasoning:
                assert spec.reasoning_effort is not None, f"{name} lacks reasoning_effort"
                assert spec.supports_temperature is False, f"{name} wrongly advertises temperature"

    def test_every_standard_skips_effort(self):
        for name, spec in KNOWN_MODELS.items():
            if not spec.is_reasoning:
                assert spec.reasoning_effort is None, f"{name} wrongly has reasoning_effort"
