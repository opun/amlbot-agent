"""Tests for the trace recorder: record/replay round-trip, hash stability,
FIFO order on duplicate keys."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.recorder import (
    MissingReplayEvent,
    TraceRecorder,
    build_recording_filename,
    canonical_json,
    messages_hash,
    stable_hash,
)


class TestHashing:
    def test_canonical_json_is_key_order_stable(self):
        a = canonical_json({"x": 1, "y": 2})
        b = canonical_json({"y": 2, "x": 1})
        assert a == b

    def test_stable_hash_is_deterministic(self):
        assert stable_hash({"a": [1, 2]}) == stable_hash({"a": [1, 2]})

    def test_messages_hash_ignores_transport_fields(self):
        # `name`, `tool_call_id` don't change LLM semantics for our usage;
        # including them would break hash-based replay matching after a
        # code refactor.
        m1 = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        m2 = [
            {"role": "system", "content": "s", "name": "foo"},
            {"role": "user", "content": "u", "tool_call_id": "t_1"},
        ]
        assert messages_hash(m1) == messages_hash(m2)

    def test_messages_hash_changes_with_content(self):
        m1 = [{"role": "system", "content": "A"}]
        m2 = [{"role": "system", "content": "B"}]
        assert messages_hash(m1) != messages_hash(m2)


class TestRecordReplayRoundTrip:
    def test_tool_call_round_trip(self, tmp_path):
        with TraceRecorder("test_trace", tmp_path) as rec:
            rec.record_tool_call(
                "get_address",
                {"chain": "eth", "address": "0xabc"},
                {"owner": "Binance"},
                duration_ms=42,
            )
        recording_path = rec.out_path
        assert recording_path is not None and recording_path.exists()

        # Replay: same tool_name + args → same result
        replay = TraceRecorder.for_replay(recording_path)
        result = replay.replay_tool_call(
            "get_address", {"chain": "eth", "address": "0xabc"}
        )
        assert result == {"owner": "Binance"}

    def test_arg_order_does_not_matter_for_replay(self, tmp_path):
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_tool_call(
                "get_address",
                {"chain": "eth", "address": "0xabc"},
                {"ok": True},
                duration_ms=1,
            )
        replay = TraceRecorder.for_replay(rec.out_path)
        # Swap key order — stable_hash normalizes.
        result = replay.replay_tool_call(
            "get_address", {"address": "0xabc", "chain": "eth"}
        )
        assert result == {"ok": True}

    def test_missing_replay_raises(self, tmp_path):
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_tool_call("get_address", {"a": 1}, {}, duration_ms=1)
        replay = TraceRecorder.for_replay(rec.out_path)
        with pytest.raises(MissingReplayEvent):
            replay.replay_tool_call("get_address", {"a": 2})

    def test_duplicate_key_events_fifo(self, tmp_path):
        """Two identical tool calls must come back in recording order."""
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_tool_call("all_txs", {"a": 1}, {"n": 1}, duration_ms=1)
            rec.record_tool_call("all_txs", {"a": 1}, {"n": 2}, duration_ms=1)
            rec.record_tool_call("all_txs", {"a": 1}, {"n": 3}, duration_ms=1)

        replay = TraceRecorder.for_replay(rec.out_path)
        assert replay.replay_tool_call("all_txs", {"a": 1}) == {"n": 1}
        assert replay.replay_tool_call("all_txs", {"a": 1}) == {"n": 2}
        assert replay.replay_tool_call("all_txs", {"a": 1}) == {"n": 3}
        with pytest.raises(MissingReplayEvent):
            replay.replay_tool_call("all_txs", {"a": 1})

    def test_llm_call_round_trip(self, tmp_path):
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_llm_call(
                prompt_name="hop_classifier",
                prompt_version="v1",
                model="gpt-5-mini",
                family="reasoning",
                reasoning_effort="medium",
                input_hash="abc123",
                content='{"role": "cex_deposit"}',
                parsed={"role": "cex_deposit"},
                usage={"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 7, "cached_tokens": 0},
                latency_ms=123,
                decision_id="hop_classifier_xyz",
            )
        replay = TraceRecorder.for_replay(rec.out_path)
        evt = replay.replay_llm_call("hop_classifier", "abc123")
        assert evt["parsed"] == {"role": "cex_deposit"}
        assert evt["usage"]["reasoning_tokens"] == 7
        assert evt["model"] == "gpt-5-mini"

    def test_recorded_tool_error_re_raises(self, tmp_path):
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_tool_call(
                "bridge_analyze", {"x": 1}, None,
                duration_ms=1, error="tool_timeout",
            )
        replay = TraceRecorder.for_replay(rec.out_path)
        with pytest.raises(RuntimeError, match="tool_timeout"):
            replay.replay_tool_call("bridge_analyze", {"x": 1})


class TestReplayIntegrity:
    def test_unused_events_detected(self, tmp_path):
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_tool_call("a", {}, {}, duration_ms=1)
            rec.record_tool_call("b", {}, {}, duration_ms=1)
        replay = TraceRecorder.for_replay(rec.out_path)
        replay.replay_tool_call("a", {})
        leftover = list(replay.unused_replay_events())
        assert len(leftover) == 1
        assert leftover[0]["tool_name"] == "b"


class TestBuildRecordingFilename:
    """Pure-function tests for the new human-scannable filename scheme.

    Format contract (pinned here so callers can rely on it):
        {YYYYMMDDThhmmssZ}_{chain}_{asset}_{tx_prefix}__{trace_suffix}.jsonl
    """

    TS = datetime(2026, 4, 20, 14, 30, 12, tzinfo=timezone.utc)

    def test_full_context(self):
        name = build_recording_filename(
            trace_id="50c334fbc63b4948bdfb79996d6736d3",
            timestamp=self.TS,
            chain="eth",
            asset="ETH",
            tx_hash="0xe0c92b5519912358a0cb8c95dfc4831b083596db9ef81590463500480802ce86",
        )
        assert name == "20260420T143012Z_eth_ETH_0xe0c92b55__50c334fb.jsonl"

    def test_missing_tx_uses_victim_address(self):
        name = build_recording_filename(
            trace_id="abcdef0123456789",
            timestamp=self.TS,
            chain="trx",
            asset="USDT",
            tx_hash=None,
            victim_address="TRkzFBecAKHJ3rKzstq2XpVSXefXyhZ74z",
        )
        # addr<last 8 of victim>
        assert name.endswith("_trx_USDT_addrXyhZ74z__abcdef01.jsonl") or (
            "addr" in name and name.endswith("abcdef01.jsonl")
        )

    def test_missing_all_context_falls_back(self):
        name = build_recording_filename(
            trace_id="deadbeefdeadbeef",
            timestamp=self.TS,
        )
        assert name == "20260420T143012Z_unknown_NA_noctx__deadbeef.jsonl"

    def test_chain_is_lowered_asset_is_uppered(self):
        name = build_recording_filename(
            trace_id="abcd1234",
            timestamp=self.TS,
            chain="ETH",
            asset="usdt",
            tx_hash="0xabc123",
        )
        assert "_eth_USDT_" in name

    def test_unsafe_symbol_chars_sanitized(self):
        """``BUSD.E`` and similar shouldn't appear verbatim — dot /
        slash break path joining on some filesystems."""
        name = build_recording_filename(
            trace_id="abcd1234",
            timestamp=self.TS,
            chain="eth",
            asset="BUSD.E",
            tx_hash="0x1122",
        )
        assert "BUSD_E" in name
        assert "." not in name.rsplit(".", 1)[0]  # only the final .jsonl dot

    def test_trace_suffix_bounded(self):
        """Long trace ids get truncated to 8 chars so the filename stays
        bounded even on SDKs that return 64-char hashes."""
        long_id = "x" * 64
        name = build_recording_filename(
            trace_id=long_id, timestamp=self.TS, chain="eth",
            asset="ETH", tx_hash="0xabc",
        )
        # Suffix chunk between __ and .jsonl should be exactly 8 chars.
        suffix = name.rsplit("__", 1)[1].removesuffix(".jsonl")
        assert len(suffix) == 8


class TestLazyFilenameMaterialization:
    """Filename is decided on the first ``record_*`` call, not at
    recorder construction — so a call to ``set_context`` between them
    must influence the resulting path."""

    def test_set_context_after_init_affects_filename(self, tmp_path):
        rec = TraceRecorder(
            trace_id="50c334fbc63b4948",
            out_dir=tmp_path,
        )
        # No file yet — filename hasn't been built.
        assert rec.out_path is None
        rec.set_context(chain="eth", asset="USDT", tx_hash="0xdeadbeef00")
        rec.record_tool_call("get_address", {"x": 1}, {"ok": True}, duration_ms=1)
        assert rec.out_path is not None
        assert "_eth_USDT_0xdeadbee" in rec.out_path.name

    def test_record_without_set_context_uses_fallbacks(self, tmp_path):
        rec = TraceRecorder(
            trace_id="abcdefabcdef",
            out_dir=tmp_path,
        )
        rec.record_tool_call("get_address", {}, {}, duration_ms=1)
        assert rec.out_path is not None
        assert "_unknown_NA_noctx__" in rec.out_path.name

    def test_set_context_is_partial_update(self, tmp_path):
        rec = TraceRecorder(trace_id="aaaabbbbcccc", out_dir=tmp_path)
        rec.set_context(chain="trx")
        rec.set_context(asset="USDT")
        rec.record_tool_call("a", {}, {}, duration_ms=1)
        assert "_trx_USDT_" in rec.out_path.name

    def test_set_context_is_noop_after_first_write(self, tmp_path):
        """Once the file is open, later ``set_context`` calls don't
        move the file — the original filename stays valid and the
        stream doesn't get re-created."""
        rec = TraceRecorder(trace_id="1111222233334444", out_dir=tmp_path)
        rec.set_context(chain="eth", asset="ETH", tx_hash="0xaa")
        rec.record_tool_call("a", {}, {}, duration_ms=1)
        original_path = rec.out_path
        rec.set_context(chain="trx", asset="USDT", tx_hash="0xbb")
        rec.record_tool_call("b", {}, {}, duration_ms=1)
        assert rec.out_path == original_path
        # Both events land in the same file.
        lines = rec.out_path.read_text().splitlines()
        assert len(lines) == 2


class TestOutputFormat:
    def test_jsonl_event_ids_increment(self, tmp_path):
        with TraceRecorder("t", tmp_path) as rec:
            rec.record_tool_call("a", {}, {}, duration_ms=1)
            rec.record_tool_call("b", {}, {}, duration_ms=1)
        lines = rec.out_path.read_text().splitlines()
        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])
        assert e1["event_id"] == 1
        assert e2["event_id"] == 2
        assert "ts_ns" in e1
        assert e1["event_type"] == "tool_call"
