"""Single place where we call the OpenAI chat API.

Every LLM decision in the tracer (hop_classifier, hop_selector, validator,
future hop_strategist) goes through :func:`call_llm`. The wrapper owns:

* Payload shaping per :class:`ModelSpec` family — reasoning models never
  see ``temperature`` or ``max_tokens``; they get ``reasoning_effort`` and
  ``max_completion_tokens`` instead.
* Recorder integration — records the call on the way out, replays it on
  the way in.
* ``DecisionRef``-ready result: ``content``, ``parsed`` JSON, ``usage``
  (incl. ``reasoning_tokens``), ``latency_ms``, stable ``input_hash``.

The goal is that no caller ever passes ``temperature`` or branches on model
name. If a specific prompt needs a tuning knob, it goes on ``ModelSpec``
first, then flows through this wrapper.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import AsyncOpenAI

from .model_registry import ModelSpec
from .recorder import MissingReplayEvent, TraceRecorder, messages_hash

try:
    from agents import generation_span as _agents_generation_span  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - openai-agents is a project dep today
    _agents_generation_span = None

logger = logging.getLogger(__name__)

ResponseFormat = Literal["text", "json"]


@dataclass
class LLMResult:
    """Everything a caller might want from a single LLM invocation."""

    content: str
    parsed: Any
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_summary: str | None = None
    model: str = ""
    family: str = ""
    reasoning_effort: str | None = None
    prompt_name: str = ""
    prompt_version: str = ""
    input_hash: str = ""
    latency_ms: int = 0
    decision_id: str = ""
    from_replay: bool = False


def strip_code_fences(text: str) -> str:
    r"""Strip the ```json / ``` fences some models add despite being told not to."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _build_create_kwargs(
    model_spec: ModelSpec,
    messages: list[dict[str, Any]],
    *,
    response_format: ResponseFormat,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Assemble the ``chat.completions.create`` kwargs for a given model.

    Reasoning family: add ``reasoning_effort``, skip ``temperature``,
    ``top_p``, ``max_tokens``. Use ``max_completion_tokens`` for output
    caps.

    Standard family: allow ``temperature`` and ``max_completion_tokens``
    (preferred over the deprecated ``max_tokens``).
    """
    kwargs: dict[str, Any] = {
        "model": model_spec.name,
        "messages": messages,
    }

    if model_spec.is_reasoning:
        if model_spec.reasoning_effort:
            kwargs["reasoning_effort"] = model_spec.reasoning_effort
    # Deliberately do *not* set ``temperature`` or ``top_p`` by default — the
    # tracer's decisions want the model's default (typically temperature=1
    # on standard, implicit on reasoning). If a future caller needs a
    # temperature knob, add it to ModelSpec.

    output_cap = max_output_tokens if max_output_tokens is not None else model_spec.max_output_tokens
    if output_cap is not None:
        kwargs["max_completion_tokens"] = output_cap

    if response_format == "json" and model_spec.supports_structured_output:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs


def _normalize_usage(usage_obj: Any) -> dict[str, int]:
    """Flatten the OpenAI ``usage`` object into a dict with reasoning stats."""
    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}

    usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj)
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

    details = usage.get("completion_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)

    input_details = usage.get("prompt_tokens_details") or {}
    if hasattr(input_details, "model_dump"):
        input_details = input_details.model_dump()
    cached_tokens = int(input_details.get("cached_tokens") or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
    }


def _new_decision_id(prompt_name: str) -> str:
    return f"{prompt_name}_{uuid.uuid4().hex[:10]}"


class _NullSpan:
    """No-op context manager used when openai-agents isn't installed."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


async def call_llm(
    *,
    openai_client: AsyncOpenAI,
    model_spec: ModelSpec,
    prompt_name: str,
    prompt_version: str,
    system: str,
    user: str | dict[str, Any] | list[Any],
    recorder: TraceRecorder | None = None,
    response_format: ResponseFormat = "text",
    max_output_tokens: int | None = None,
    timeout: float = 60.0,
    decision_id: str | None = None,
    allow_replay: bool = True,
) -> LLMResult:
    """Invoke an LLM decision point.

    Parameters
    ----------
    prompt_name, prompt_version:
        Identify which prompt + which version. Written to recorder and
        ``DecisionRef``; used by replay to match events.
    system, user:
        Message contents. ``user`` is JSON-serialized if not already a
        string. ``list`` is passed through as-is (for future multi-part
        content support).
    response_format:
        ``"json"`` adds ``response_format={"type": "json_object"}`` and
        parses the response into ``LLMResult.parsed``. Parse failures
        raise ``json.JSONDecodeError`` — the caller decides whether to
        retry or error out.
    allow_replay:
        When ``True`` and ``recorder.is_replay`` is set, results come from
        the recording. Set to ``False`` to force a live call even in
        replay mode (used by ``--override-prompt``).
    """
    # Build messages
    if isinstance(user, str):
        user_content = user
    else:
        user_content = json.dumps(user, indent=2, ensure_ascii=False, default=str)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    input_hash = messages_hash(messages)
    decision_id = decision_id or _new_decision_id(prompt_name)

    # Replay short-circuit
    if allow_replay and recorder is not None and recorder.is_replay:
        try:
            event = recorder.replay_llm_call(prompt_name, input_hash)
        except MissingReplayEvent:
            logger.warning(
                "replay miss for %s (hash=%s...) — falling back to live call",
                prompt_name, input_hash[:12],
            )
            # When there's no live client (typical for offline prompt-eval
            # tests whose recordings were captured against an older prompt
            # version), surface the miss so callers can pytest-skip rather
            # than crash on a ``None.chat`` dereference.
            if openai_client is None:
                raise
        else:
            content = event.get("content", "") or ""
            parsed = event.get("parsed")
            if parsed is None and response_format == "json" and content:
                try:
                    parsed = json.loads(strip_code_fences(content))
                except json.JSONDecodeError:
                    parsed = None
            return LLMResult(
                content=content,
                parsed=parsed,
                usage=event.get("usage") or {},
                reasoning_summary=event.get("reasoning_summary"),
                model=event.get("model", model_spec.name),
                family=event.get("family", model_spec.family),
                reasoning_effort=event.get("reasoning_effort", model_spec.reasoning_effort),
                prompt_name=prompt_name,
                prompt_version=event.get("prompt_version", prompt_version),
                input_hash=input_hash,
                latency_ms=int(event.get("latency_ms") or 0),
                decision_id=event.get("decision_id") or decision_id,
                from_replay=True,
            )

    # Live call
    create_kwargs = _build_create_kwargs(
        model_spec,
        messages,
        response_format=response_format,
        max_output_tokens=max_output_tokens,
    )

    # Wrap the actual API call in openai-agents' generation_span when the
    # SDK is available so traces flow into the same observability pipeline
    # the rest of the tracer uses.
    span_cm = (
        _agents_generation_span(
            input=messages,
            model=model_spec.name,
            model_config={"purpose": prompt_name, "family": model_spec.family},
        )
        if _agents_generation_span is not None
        else _NullSpan()
    )

    started = time.perf_counter()
    with span_cm as gen_span:
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(**create_kwargs),
            timeout=timeout,
        )
        try:
            if hasattr(gen_span, "span_data"):
                gen_span.span_data.output = [response.choices[0].message.model_dump()]
                gen_span.span_data.usage = _normalize_usage(response.usage)
        except Exception:
            pass
    latency_ms = int((time.perf_counter() - started) * 1000)

    content = response.choices[0].message.content or ""
    parsed: Any = None
    if response_format == "json":
        cleaned = strip_code_fences(content)
        if cleaned:
            parsed = json.loads(cleaned)

    usage = _normalize_usage(response.usage)

    # Reasoning-starvation guard. Reasoning models share
    # ``max_completion_tokens`` between internal reasoning and the final
    # content. When the reasoning pass uses the whole budget, content is
    # empty and ``parsed`` stays ``None`` — downstream code then silently
    # falls back to heuristics. Warning here is cheap and makes the
    # symptom diagnosable without a recorder.
    if (
        model_spec.is_reasoning
        and not content
        and usage.get("reasoning_tokens", 0) >= usage.get("output_tokens", 0) > 0
    ):
        logger.warning(
            "call_llm(%s) returned empty content: reasoning used the entire "
            "output budget (reasoning=%d, output=%d, cap=%s). Raise "
            "max_output_tokens in the prompt frontmatter.",
            prompt_name, usage["reasoning_tokens"], usage["output_tokens"],
            max_output_tokens or model_spec.max_output_tokens,
        )

    result = LLMResult(
        content=content,
        parsed=parsed,
        usage=usage,
        reasoning_summary=None,  # chat.completions does not expose it; Responses API would.
        model=model_spec.name,
        family=model_spec.family,
        reasoning_effort=model_spec.reasoning_effort,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        input_hash=input_hash,
        latency_ms=latency_ms,
        decision_id=decision_id,
        from_replay=False,
    )

    if recorder is not None and recorder.is_recording:
        recorder.record_llm_call(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            model=model_spec.name,
            family=model_spec.family,
            reasoning_effort=model_spec.reasoning_effort,
            input_hash=input_hash,
            content=content,
            parsed=parsed,
            usage=usage,
            latency_ms=latency_ms,
            decision_id=decision_id,
            reasoning_summary=result.reasoning_summary,
        )

    return result
