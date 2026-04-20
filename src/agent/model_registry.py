"""Model registry: one place that knows about each OpenAI model's capabilities.

The rest of the codebase never branches on ``model.startswith("gpt-5")`` —
it consults ``resolve_model(name)`` and reads fields off the returned
``ModelSpec``. That keeps reasoning-family quirks (no ``temperature``, use
``reasoning_effort``, ``max_completion_tokens`` instead of ``max_tokens``)
in exactly one place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

Family = Literal["reasoning", "standard"]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class ModelSpec:
    """Everything ``llm_client.call_llm`` needs to shape the API payload."""

    name: str
    family: Family
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int | None = None
    supports_temperature: bool = True
    supports_structured_output: bool = True

    @property
    def is_reasoning(self) -> bool:
        return self.family == "reasoning"

    def with_overrides(
        self,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelSpec:
        """Return a copy with the given fields overridden.

        Raises if ``reasoning_effort`` is set on a non-reasoning model — that
        combination silently degrades (the API ignores it) and the caller
        almost certainly intended to set it on a different model.
        """
        if reasoning_effort is not None and not self.is_reasoning:
            raise ValueError(
                f"reasoning_effort={reasoning_effort!r} set on non-reasoning model {self.name!r}"
            )
        changes: dict[str, object] = {}
        if reasoning_effort is not None:
            changes["reasoning_effort"] = reasoning_effort
        if max_output_tokens is not None:
            changes["max_output_tokens"] = max_output_tokens
        return replace(self, **changes) if changes else self


KNOWN_MODELS: dict[str, ModelSpec] = {
    # Reasoning family: no temperature/top_p, use reasoning_effort
    "gpt-5": ModelSpec(
        name="gpt-5", family="reasoning",
        reasoning_effort="high", supports_temperature=False,
    ),
    "gpt-5-mini": ModelSpec(
        name="gpt-5-mini", family="reasoning",
        reasoning_effort="medium", supports_temperature=False,
    ),
    "gpt-5-nano": ModelSpec(
        name="gpt-5-nano", family="reasoning",
        reasoning_effort="low", supports_temperature=False,
    ),
    "o1": ModelSpec(
        name="o1", family="reasoning",
        reasoning_effort="medium", supports_temperature=False,
    ),
    "o1-mini": ModelSpec(
        name="o1-mini", family="reasoning",
        reasoning_effort="medium", supports_temperature=False,
    ),
    "o3": ModelSpec(
        name="o3", family="reasoning",
        reasoning_effort="medium", supports_temperature=False,
    ),
    "o3-mini": ModelSpec(
        name="o3-mini", family="reasoning",
        reasoning_effort="medium", supports_temperature=False,
    ),
    "o4-mini": ModelSpec(
        name="o4-mini", family="reasoning",
        reasoning_effort="medium", supports_temperature=False,
    ),
    # Standard family: temperature OK, no reasoning_effort
    "gpt-4o": ModelSpec(
        name="gpt-4o", family="standard",
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini", family="standard",
    ),
    "gpt-4.1": ModelSpec(
        name="gpt-4.1", family="standard",
    ),
    "gpt-4.1-mini": ModelSpec(
        name="gpt-4.1-mini", family="standard",
    ),
    "gpt-4.1-nano": ModelSpec(
        name="gpt-4.1-nano", family="standard",
    ),
}


# o-series (o1, o3, o4, o5-...) and gpt-5* are reasoning. Anything else
# falls through to standard; add to KNOWN_MODELS explicitly when that's
# wrong, rather than growing this heuristic.
_REASONING_RE = re.compile(r"^(o\d+|gpt-5)([-.].*)?$", re.IGNORECASE)


def _infer_spec(name: str) -> ModelSpec:
    """Best-effort inference for model names not in ``KNOWN_MODELS``."""
    if _REASONING_RE.match(name):
        return ModelSpec(
            name=name, family="reasoning",
            reasoning_effort="medium", supports_temperature=False,
        )
    return ModelSpec(name=name, family="standard")


def resolve_model(
    name: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    max_output_tokens: int | None = None,
) -> ModelSpec:
    """Look up a model spec by name; apply optional runtime overrides."""
    spec = KNOWN_MODELS.get(name) or _infer_spec(name)
    if reasoning_effort is None and max_output_tokens is None:
        return spec
    return spec.with_overrides(
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
