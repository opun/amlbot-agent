"""Tests for the DecisionRef addition to Step / TraceResult."""
from __future__ import annotations

import json

from agent.models import (
    Annotation,
    CaseMeta,
    DecisionRef,
    Entity,
    Path,
    Step,
    TraceResult,
    TraceStats,
)


def _make_decision(name: str = "hop_classifier") -> DecisionRef:
    return DecisionRef(
        prompt_name=name,
        prompt_version="v1",
        model="gpt-5-mini",
        family="reasoning",
        reasoning_effort="medium",
        input_hash="a" * 64,
        output_summary={"role": "cex_deposit", "terminal": True},
        usage={"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 50, "cached_tokens": 0},
        latency_ms=340,
        decision_id="hop_classifier_xyz",
    )


def _make_step(with_decisions: list[DecisionRef] | None = None) -> Step:
    return Step(
        step_index=0,
        **{"from": "0xA", "to": "0xB"},
        tx_hash="0xabc",
        chain="eth",
        asset="USDT",
        amount_estimate=1000.0,
        time=1700000000,
        direction="out",
        step_type="direct_transfer",
        llm_decisions=with_decisions or [],
    )


class TestStepWithDecisions:
    def test_default_llm_decisions_is_empty(self):
        step = _make_step()
        assert step.llm_decisions == []

    def test_accepts_decision_refs(self):
        d = _make_decision()
        step = _make_step([d])
        assert len(step.llm_decisions) == 1
        assert step.llm_decisions[0].prompt_name == "hop_classifier"

    def test_accepts_dicts_for_decisions(self):
        """The agentic loop serializes decisions to dict via model_dump()
        before shoving them into step dicts. Pydantic should accept either."""
        d = _make_decision()
        step_data = {
            "step_index": 0,
            "from": "0xA",
            "to": "0xB",
            "tx_hash": "0xabc",
            "chain": "eth",
            "asset": "USDT",
            "amount_estimate": 1000.0,
            "direction": "out",
            "step_type": "direct_transfer",
            "llm_decisions": [d.model_dump()],
        }
        step = Step(**step_data)
        assert step.llm_decisions[0].prompt_name == "hop_classifier"

    def test_roundtrip_json(self):
        step = _make_step([_make_decision(), _make_decision("hop_selector")])
        raw = step.model_dump_json(by_alias=True)
        data = json.loads(raw)
        assert len(data["llm_decisions"]) == 2
        assert data["llm_decisions"][0]["prompt_name"] == "hop_classifier"
        assert data["llm_decisions"][1]["prompt_name"] == "hop_selector"

    def test_backwards_compat_step_without_decisions(self):
        """Historical fixtures without llm_decisions must still validate."""
        step_data = {
            "step_index": 0,
            "from": "0xA",
            "to": "0xB",
            "tx_hash": "0xabc",
            "chain": "eth",
            "asset": "USDT",
            "amount_estimate": 1000.0,
            "direction": "out",
            "step_type": "direct_transfer",
        }
        step = Step.model_validate(step_data)
        assert step.llm_decisions == []


class TestTraceResultWithDecisionLog:
    def _base(self, *, decision_log=None) -> TraceResult:
        return TraceResult(
            case_meta=CaseMeta(
                case_id="c",
                victim_address="0xA",
                blockchain_name="eth",
                chains=["eth"],
                asset_symbol="USDT",
            ),
            paths=[Path(path_id="1", description="x", steps=[])],
            entities=[Entity(address="0xA", chain="eth", role="victim")],
            annotations=[],
            trace_stats=TraceStats(initial_amount_estimate=0.0, explored_paths=0),
            decision_log=decision_log or [],
        )

    def test_default_decision_log_is_empty(self):
        r = self._base()
        assert r.decision_log == []

    def test_decision_log_roundtrip(self):
        r = self._base(decision_log=[_make_decision("validator")])
        raw = json.loads(r.to_json())
        assert len(raw["decision_log"]) == 1
        assert raw["decision_log"][0]["prompt_name"] == "validator"

    def test_backwards_compat_trace_result_without_decision_log(self):
        """TraceResult JSON pre-DecisionRef must still parse."""
        data = self._base().model_dump(by_alias=True)
        data.pop("decision_log")
        reparsed = TraceResult.model_validate(data)
        assert reparsed.decision_log == []
