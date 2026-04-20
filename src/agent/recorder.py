"""TraceRecorder — append-only JSONL log of every tool call and LLM call a
trace makes, plus a replay mode that feeds recorded results back in order.

Design notes
------------
* Write-through. Each event is ``json.dumps``-ed and flushed to disk as
  it's recorded, so a crash mid-trace still leaves a usable partial log.
* Hash-based replay matching. Tool calls match on ``(tool_name, args_hash)``
  and LLM calls on ``(prompt_name, input_hash)``. When multiple events
  share a key (e.g. two identical ``get_address`` calls), they're popped
  in FIFO order — preserving the original execution order.
* No network. The recorder is a pure filesystem concern; it never talks
  to S3/GCS or any external store. Recordings live under
  ``AGENT_RECORDINGS_DIR`` (default ``./recordings/<yyyy-mm-dd>/``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class MissingReplayEvent(RuntimeError):
    """Replay was asked for an event that isn't in the recording."""


def canonical_json(value: Any) -> str:
    """JSON with stable key ordering, for hashing purposes.

    ``default=str`` catches stray objects (``datetime``, ``Path``) so we
    never crash mid-hash; those types have stable repr so hashes stay
    comparable.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def messages_hash(messages: list[dict[str, Any]]) -> str:
    """Hash the semantic content of an OpenAI messages list.

    Includes only ``role`` and ``content`` — drops transport fields like
    ``name``, ``tool_call_id`` that don't influence LLM output for our
    single-turn calls.
    """
    normalized = [
        {"role": m.get("role", ""), "content": m.get("content", "")}
        for m in messages
    ]
    return stable_hash(normalized)


@dataclass
class _ReplayIndex:
    """In-memory queues of recorded events, keyed for FIFO matching."""

    tool_calls: dict[tuple[str, str], deque[dict[str, Any]]] = field(default_factory=dict)
    llm_calls: dict[tuple[str, str], deque[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "_ReplayIndex":
        idx = cls()
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("recorder: skipping malformed line %d in %s: %s", line_no, path, e)
                    continue
                kind = event.get("event_type")
                if kind == "tool_call":
                    key = (event.get("tool_name", ""), event.get("args_hash", ""))
                    idx.tool_calls.setdefault(key, deque()).append(event)
                elif kind == "llm_call":
                    key = (event.get("prompt_name", ""), event.get("input_hash", ""))
                    idx.llm_calls.setdefault(key, deque()).append(event)
                # other kinds (state_snapshot, llm_reasoning) are informational
                # on replay and don't need lookup queues.
        return idx


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _sanitize_segment(value: str | None, *, fallback: str) -> str:
    """Keep filenames cross-platform-safe.

    Strips everything outside ``[A-Za-z0-9_-]`` and collapses whitespace to
    ``_``. Used for ``chain`` and ``asset`` segments — symbols like
    ``BUSD.E`` or ``USD₮`` would otherwise yield invalid paths on some
    filesystems. Returns ``fallback`` for empty / all-whitespace input so
    downstream joiners always get a non-empty token.
    """
    if not value:
        return fallback
    cleaned = _FILENAME_SAFE_RE.sub("_", str(value).strip())
    return cleaned or fallback


def _tx_prefix(tx_hash: str | None, victim_address: str | None) -> str:
    """Compose the identifier segment of the filename.

    Priority: (1) ``tx_hash`` — first 10 hex chars, ``0x`` prefix kept for
    EVM so the recording is trivially joinable to an explorer URL.
    (2) ``addr<victim_last_8>`` when no tx_hash is available (address-mode
    trace seed). (3) ``noctx`` as last resort so the segment is never
    empty.
    """
    if tx_hash:
        clean = _FILENAME_SAFE_RE.sub("", str(tx_hash).strip())
        if clean:
            if clean.lower().startswith("0x"):
                # Keep 0x + 8 hex = 10 chars total.
                return clean[:10]
            return clean[:10]
    if victim_address:
        clean = _FILENAME_SAFE_RE.sub("", str(victim_address).strip())
        if clean:
            return f"addr{clean[-8:]}"
    return "noctx"


def _trace_suffix(trace_id: str) -> str:
    """Short collision-breaker derived from the outer trace_id.

    Two concurrent runs on the same (ts, chain, asset, tx) tuple would
    otherwise fight over the same path — 8 hex chars is plenty for a
    single operator's recordings dir.
    """
    hex_only = _FILENAME_SAFE_RE.sub("", trace_id or "")
    return (hex_only or "00000000")[:8]


def build_recording_filename(
    *,
    trace_id: str,
    timestamp: datetime | None = None,
    chain: str | None = None,
    asset: str | None = None,
    tx_hash: str | None = None,
    victim_address: str | None = None,
) -> str:
    """Compose a human-scannable filename for a trace recording.

    Format: ``{YYYYMMDDThhmmssZ}_{chain}_{asset}_{tx_prefix}__{trace_suffix}.jsonl``

    Example: ``20260420T143012Z_eth_ETH_0xe0c92b55__50c334fb.jsonl``.

    Missing context is papered over with explicit fallback tokens
    (``unknown`` / ``NA`` / ``noctx``) so the filename remains valid even
    for a replay that never learned its seed.
    """
    ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
    ts_token = ts.strftime("%Y%m%dT%H%M%SZ")
    chain_token = _sanitize_segment(chain, fallback="unknown").lower()
    asset_token = _sanitize_segment(asset, fallback="NA").upper()
    tx_token = _tx_prefix(tx_hash, victim_address)
    suffix = _trace_suffix(trace_id)
    return f"{ts_token}_{chain_token}_{asset_token}_{tx_token}__{suffix}.jsonl"


class TraceRecorder:
    """Append-only event log + optional replay of a prior recording.

    Instantiate one of two modes:

    * Recording:       ``TraceRecorder(trace_id, out_dir)``
    * Replay-only:     ``TraceRecorder.for_replay(path)``
    * Replay + log:    ``TraceRecorder.for_replay(path, out_dir=...)``
                       (useful for ``--replay --override-prompt`` — we
                       re-record the mutated events to a new file.)
    """

    def __init__(
        self,
        trace_id: str,
        out_dir: Path | None,
        *,
        record_reasoning: bool = False,
        _replay_index: _ReplayIndex | None = None,
        _replay_path: Path | None = None,
    ):
        self.trace_id = trace_id
        self.record_reasoning = record_reasoning
        self._event_counter = 0
        self._replay_index = _replay_index
        self._replay_path = _replay_path
        self._out_path: Path | None = None
        self._fh = None
        # Filename generation is deferred: callers typically don't know
        # ``tx_hash`` / ``chain`` / ``asset`` at construction time (CLI
        # parses them from free-text after the recorder is wired). We
        # stash the target directory and materialize the file on first
        # ``record_*`` call, picking up whatever context ``set_context``
        # has supplied by then.
        self._out_dir: Path | None = out_dir
        self._created_at: datetime = datetime.now(timezone.utc)
        self._context: dict[str, str | None] = {
            "chain": None,
            "asset": None,
            "tx_hash": None,
            "victim_address": None,
        }

    def set_context(
        self,
        *,
        chain: str | None = None,
        asset: str | None = None,
        tx_hash: str | None = None,
        victim_address: str | None = None,
    ) -> None:
        """Populate fields used to build the output filename.

        Call once the seed details are known (for CLI, after description
        parsing + explicit-param merge; for API, right after creating the
        recorder from ``TracerConfig``). Any ``None`` argument leaves the
        existing value alone, so partial updates compose. Must be called
        before the first ``record_*()`` call to influence the filename —
        otherwise fallbacks are used.
        """
        for key, value in (
            ("chain", chain),
            ("asset", asset),
            ("tx_hash", tx_hash),
            ("victim_address", victim_address),
        ):
            if value is not None:
                self._context[key] = value

    def _ensure_open(self) -> None:
        """Materialize ``_out_path`` + ``_fh`` on first write.

        Idempotent: subsequent calls are no-ops. Only runs when an
        ``out_dir`` was provided; replay-only recorders stay in-memory.
        """
        if self._fh is not None or self._out_dir is None:
            return
        day = self._created_at.strftime("%Y-%m-%d")
        day_dir = self._out_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = build_recording_filename(
            trace_id=self.trace_id,
            timestamp=self._created_at,
            chain=self._context.get("chain"),
            asset=self._context.get("asset"),
            tx_hash=self._context.get("tx_hash"),
            victim_address=self._context.get("victim_address"),
        )
        self._out_path = day_dir / filename
        self._fh = self._out_path.open("a", encoding="utf-8")
        logger.info("recorder: writing to %s", self._out_path)

    @classmethod
    def for_replay(
        cls,
        replay_path: Path,
        *,
        out_dir: Path | None = None,
        trace_id: str | None = None,
        record_reasoning: bool = False,
    ) -> "TraceRecorder":
        index = _ReplayIndex.load(replay_path)
        if trace_id is None:
            trace_id = f"replay_{replay_path.stem}_{int(time.time())}"
        return cls(
            trace_id=trace_id,
            out_dir=out_dir,
            record_reasoning=record_reasoning,
            _replay_index=index,
            _replay_path=replay_path,
        )

    # ─── Properties ────────────────────────────────────────────────────────

    @property
    def is_replay(self) -> bool:
        return self._replay_index is not None

    @property
    def is_recording(self) -> bool:
        # Truth-value of the "am I going to write?" flag. ``_fh`` is
        # opened lazily on first ``_write`` call, so we can't rely on it
        # alone — check the configured ``out_dir`` too.
        return self._fh is not None or self._out_dir is not None

    @property
    def out_path(self) -> Path | None:
        return self._out_path

    @property
    def replay_path(self) -> Path | None:
        return self._replay_path

    # ─── Writing ───────────────────────────────────────────────────────────

    def _next_event_id(self) -> int:
        self._event_counter += 1
        return self._event_counter

    def _write(self, event: dict[str, Any]) -> None:
        # Lazy materialization: the first write triggers path construction
        # using whatever ``set_context`` has learned by now. No-ops when
        # the recorder has no ``out_dir`` (replay-only mode).
        if self._fh is None:
            self._ensure_open()
        if self._fh is None:
            return
        event = {
            "event_id": self._next_event_id(),
            "ts_ns": time.time_ns(),
            **event,
        }
        self._fh.write(json.dumps(event, ensure_ascii=False, default=str))
        self._fh.write("\n")
        self._fh.flush()

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        duration_ms: int,
        error: str | None = None,
        decision_id: str | None = None,
    ) -> None:
        self._write({
            "event_type": "tool_call",
            "tool_name": tool_name,
            "args_hash": stable_hash({"tool": tool_name, "args": arguments}),
            "arguments": arguments,
            "result": result if error is None else None,
            "error": error,
            "duration_ms": duration_ms,
            "decision_id": decision_id,
        })

    def record_llm_call(
        self,
        *,
        prompt_name: str,
        prompt_version: str,
        model: str,
        family: str,
        reasoning_effort: str | None,
        input_hash: str,
        content: str,
        parsed: Any,
        usage: dict[str, Any],
        latency_ms: int,
        decision_id: str,
        reasoning_summary: str | None = None,
    ) -> None:
        self._write({
            "event_type": "llm_call",
            "decision_id": decision_id,
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "model": model,
            "family": family,
            "reasoning_effort": reasoning_effort,
            "input_hash": input_hash,
            "content": content,
            "parsed": parsed,
            "usage": usage,
            "latency_ms": latency_ms,
        })
        if reasoning_summary and self.record_reasoning:
            self._write({
                "event_type": "llm_reasoning",
                "decision_id": decision_id,
                "reasoning_summary": reasoning_summary,
            })

    def record_state(self, kind: str, payload: dict[str, Any]) -> None:
        self._write({
            "event_type": "state_snapshot",
            "kind": kind,
            "payload": payload,
        })

    # ─── Replay ────────────────────────────────────────────────────────────

    def replay_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self._replay_index is None:
            raise RuntimeError("replay_tool_call called on non-replay recorder")
        args_hash = stable_hash({"tool": tool_name, "args": arguments})
        queue = self._replay_index.tool_calls.get((tool_name, args_hash))
        if not queue:
            raise MissingReplayEvent(
                f"No recorded tool_call for {tool_name} with args_hash={args_hash[:12]}"
            )
        event = queue.popleft()
        if event.get("error"):
            raise RuntimeError(f"Recorded tool error: {event['error']}")
        return event.get("result")

    def replay_llm_call(self, prompt_name: str, input_hash: str) -> dict[str, Any]:
        if self._replay_index is None:
            raise RuntimeError("replay_llm_call called on non-replay recorder")
        queue = self._replay_index.llm_calls.get((prompt_name, input_hash))
        if not queue:
            raise MissingReplayEvent(
                f"No recorded llm_call for {prompt_name} with input_hash={input_hash[:12]}"
            )
        return queue.popleft()

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    def flush(self) -> Path | None:
        if self._fh is not None:
            self._fh.flush()
        return self._out_path

    def close(self) -> Path | None:
        out = self._out_path
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return out

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ─── Helpers ───────────────────────────────────────────────────────────

    def unused_replay_events(self) -> Iterator[dict[str, Any]]:
        """Yield events in the replay index that were never consumed.

        Useful as a sanity check — after ``--replay``, all recorded events
        should have been consumed in the same order. Any leftover means
        the current code path diverged from the recorded one.
        """
        if self._replay_index is None:
            return
        for queue in self._replay_index.tool_calls.values():
            yield from queue
        for queue in self._replay_index.llm_calls.values():
            yield from queue


def default_recordings_dir() -> Path:
    return Path(os.environ.get("AGENT_RECORDINGS_DIR", "recordings"))


def should_record() -> bool:
    return os.environ.get("AGENT_RECORD", "").lower() in ("1", "true", "yes")


def should_record_reasoning() -> bool:
    return os.environ.get("AGENT_RECORD_REASONING", "").lower() in ("1", "true", "yes")
