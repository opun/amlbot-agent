"""Tests for call_llm: record, replay, and payload correctness.

We mock the OpenAI async client — no network calls. The important
invariants are: (1) reasoning models don't leak `temperature`, (2) the
recorder captures live calls and replay reconstructs them without
touching the client, (3) usage is normalized with reasoning_tokens.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm_client import LLMResult, _normalize_usage, call_llm, strip_code_fences
from agent.model_registry import resolve_model
from agent.recorder import TraceRecorder


def _build_fake_openai_response(
    *,
    content: str = '{"role": "intermediate"}',
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    reasoning_tokens: int | None = None,
    cached_tokens: int | None = None,
):
    """Build an object that quacks like an openai chat.completions response."""
    message = MagicMock()
    message.content = content
    message.model_dump = lambda: {"role": "assistant", "content": content}

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    usage = MagicMock()
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if reasoning_tokens is not None:
        details = MagicMock()
        details.model_dump = lambda: {"reasoning_tokens": reasoning_tokens}
        usage_dict["completion_tokens_details"] = details
    if cached_tokens is not None:
        input_details = MagicMock()
        input_details.model_dump = lambda: {"cached_tokens": cached_tokens}
        usage_dict["prompt_tokens_details"] = input_details
    usage.model_dump = lambda: usage_dict
    response.usage = usage
    return response


def _fake_openai_client(response) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


# ─── strip_code_fences ─────────────────────────────────────────────────────


def test_strip_code_fences_json_block():
    inp = "```json\n{\"a\": 1}\n```"
    assert strip_code_fences(inp) == '{"a": 1}'


def test_strip_code_fences_plain_block():
    inp = "```\n{\"a\": 1}\n```"
    assert strip_code_fences(inp) == '{"a": 1}'


def test_strip_code_fences_no_fence_passthrough():
    assert strip_code_fences("{\"x\": true}") == '{"x": true}'


# ─── usage normalization ───────────────────────────────────────────────────


def test_normalize_usage_captures_reasoning_tokens():
    usage = MagicMock()
    details = MagicMock()
    details.model_dump = lambda: {"reasoning_tokens": 128}
    usage.model_dump = lambda: {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "completion_tokens_details": details,
    }
    out = _normalize_usage(usage)
    assert out == {
        "input_tokens": 100,
        "output_tokens": 50,
        "reasoning_tokens": 128,
        "cached_tokens": 0,
    }


def test_normalize_usage_handles_none():
    out = _normalize_usage(None)
    assert out["input_tokens"] == 0
    assert out["reasoning_tokens"] == 0


# ─── call_llm: reasoning family payload ────────────────────────────────────


def test_call_llm_reasoning_does_not_pass_temperature():
    """If `temperature` ever leaks into a gpt-5-mini request, the API 400s.
    Catches that before anyone feels it in prod."""
    spec = resolve_model("gpt-5-mini")
    response = _build_fake_openai_response(reasoning_tokens=42)
    client = _fake_openai_client(response)

    result: LLMResult = asyncio.run(call_llm(
        openai_client=client,
        model_spec=spec,
        prompt_name="hop_classifier",
        prompt_version="v1",
        system="sys",
        user={"x": 1},
        response_format="json",
    ))

    client.chat.completions.create.assert_awaited_once()
    call_kwargs = client.chat.completions.create.await_args.kwargs
    assert "temperature" not in call_kwargs
    assert "top_p" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    assert call_kwargs["reasoning_effort"] == "medium"
    assert result.parsed == {"role": "intermediate"}
    assert result.usage["reasoning_tokens"] == 42
    assert result.family == "reasoning"


def test_call_llm_standard_passes_response_format():
    spec = resolve_model("gpt-4o")
    response = _build_fake_openai_response()
    client = _fake_openai_client(response)

    asyncio.run(call_llm(
        openai_client=client,
        model_spec=spec,
        prompt_name="validator",
        prompt_version="v1",
        system="sys",
        user={"x": 1},
        response_format="json",
    ))

    kwargs = client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


# ─── call_llm: replay ──────────────────────────────────────────────────────


def test_call_llm_replay_short_circuits_client(tmp_path):
    """Replay mode must not call the OpenAI client."""
    spec = resolve_model("gpt-5-mini")
    response = _build_fake_openai_response(content='{"live": true}')
    client = _fake_openai_client(response)
    recorder = TraceRecorder("t", tmp_path)

    # First: live call writes the event to the recording
    live_result = asyncio.run(call_llm(
        openai_client=client,
        model_spec=spec,
        prompt_name="hop_classifier",
        prompt_version="v1",
        system="sys",
        user={"x": 1},
        response_format="json",
        recorder=recorder,
    ))
    assert live_result.parsed == {"live": True}
    assert client.chat.completions.create.await_count == 1
    rec_path = recorder.out_path
    recorder.close()

    # Second: replay from the recording — fresh client with a different
    # canned response would win, but replay must not touch the client.
    replay_client_resp = _build_fake_openai_response(content='{"live": false}')
    replay_client = _fake_openai_client(replay_client_resp)
    replay_recorder = TraceRecorder.for_replay(rec_path)

    replayed = asyncio.run(call_llm(
        openai_client=replay_client,
        model_spec=spec,
        prompt_name="hop_classifier",
        prompt_version="v1",
        system="sys",
        user={"x": 1},  # same messages → same input_hash
        response_format="json",
        recorder=replay_recorder,
    ))
    assert replayed.from_replay is True
    assert replayed.parsed == {"live": True}  # original recording wins
    assert replay_client.chat.completions.create.await_count == 0


def test_call_llm_replay_falls_back_on_miss(tmp_path):
    spec = resolve_model("gpt-5-mini")
    response = _build_fake_openai_response(content='{"fallback": true}')
    client = _fake_openai_client(response)
    # Recorder is in replay mode but has no matching event.
    rec_path = tmp_path / "empty.jsonl"
    rec_path.write_text("")
    recorder = TraceRecorder.for_replay(rec_path)

    result = asyncio.run(call_llm(
        openai_client=client,
        model_spec=spec,
        prompt_name="hop_classifier",
        prompt_version="v1",
        system="sys",
        user={"y": 2},
        response_format="json",
        recorder=recorder,
    ))
    assert result.from_replay is False
    assert result.parsed == {"fallback": True}
    assert client.chat.completions.create.await_count == 1


def test_call_llm_same_input_same_hash():
    """Same messages ⇒ same input_hash ⇒ stable replay lookup."""
    spec = resolve_model("gpt-5-mini")
    r1 = _build_fake_openai_response()
    c1 = _fake_openai_client(r1)
    res1 = asyncio.run(call_llm(
        openai_client=c1, model_spec=spec,
        prompt_name="x", prompt_version="v1",
        system="sys", user={"a": 1},
        response_format="json",
    ))
    r2 = _build_fake_openai_response()
    c2 = _fake_openai_client(r2)
    res2 = asyncio.run(call_llm(
        openai_client=c2, model_spec=spec,
        prompt_name="x", prompt_version="v1",
        system="sys", user={"a": 1},
        response_format="json",
    ))
    assert res1.input_hash == res2.input_hash
