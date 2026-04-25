"""Regression: when the linearity check splits a path into siblings,
only the FINAL split (the one containing the original last step)
inherits the original ``stop_reason``. Earlier splits get
``stop_reason=None`` because they broke off before any terminal/dust
event fired.

Background — the bbd7c0fa trace produced one wide path:

    seed → swap-out-1 → swap-out-2 → swap-out-3 → swap-out-4
         → native-TRX → continuation-hop

The dust trim fired on the LAST step (a 1000 TRX outflow); without
this guard, the resulting ``Below dust threshold`` stop_reason
propagated to every split sibling and the visualization's dust-trim
rule (drop the LAST step of any dust-marked path) silently hid the
4 swap-out edges.
"""
from __future__ import annotations

from agent.models import (
    CaseMeta,
    Path as TPath,
    Step,
    TraceResult,
    TraceStats,
)
from agent.trace_postprocess import postprocess_trace_result


def _step(from_addr: str, to_addr: str, tx: str, amount: float = 100.0) -> Step:
    return Step(
        step_index=0,
        from_address=from_addr,
        to_address=to_addr,
        tx_hash=tx,
        chain="trx",
        asset="USDT",
        amount_estimate=amount,
        attributed_amount=amount,
        time=1775066040,
        direction="out",
        step_type="direct_transfer",
        service_label=None,
        protocol=None,
        reasoning=None,
    )


def _trace_with_path(steps, stop_reason):
    return TraceResult(
        case_meta=CaseMeta(
            case_id="test",
            trace_id="t",
            description="",
            victim_address="V",
            blockchain_name="trx",
            chains=["trx"],
            asset_symbol="USDT",
            token_id=9,
        ),
        paths=[TPath(
            path_id="1",
            description="d",
            steps=steps,
            stop_reason=stop_reason,
        )],
        entities=[],
        annotations=[],
        trace_stats=TraceStats(initial_amount_estimate=0.0, explored_paths=1),
    )


def test_split_only_propagates_stop_reason_to_last_split():
    """Path with 5 sibling outflows + final dust step splits into 5
    pieces; only the final one keeps the dust stop_reason."""
    steps = [
        _step("V", "TLW", "seed"),       # prev_to=TLW
        _step("TLW", "POOL", "sw1"),     # from=TLW matches → keep
        _step("TLW", "POOL", "sw2"),     # from=TLW != prev_to=POOL → SPLIT
        _step("TLW", "POOL", "sw3"),     # SPLIT
        _step("TLW", "POOL", "sw4"),     # SPLIT
        _step("TLW", "VICTIM", "feee", amount=1000.0),  # SPLIT (dust step)
    ]
    result = _trace_with_path(steps, "Below dust threshold (0.21% of stolen amount)")
    out = postprocess_trace_result(result)

    # 5 paths after split (sw1 stays with seed, sw2/3/4/feee each fork).
    assert len(out.paths) == 5
    # Only the FINAL split (the one containing the original last step
    # ``feee``) keeps the dust stop_reason.
    final_path = next(p for p in out.paths if any(s.tx_hash == "feee" for s in p.steps))
    assert final_path.stop_reason == "Below dust threshold (0.21% of stolen amount)"
    # Every other split path gets stop_reason=None.
    for p in out.paths:
        if p is final_path:
            continue
        assert p.stop_reason is None, (
            f"split path {p.path_id} (steps={[s.tx_hash for s in p.steps]}) "
            f"must NOT inherit dust stop_reason"
        )


def test_unsplit_path_keeps_original_stop_reason():
    """Linear path that doesn't need splitting keeps its original
    stop_reason — sanity check the new branch only kicks in on splits."""
    steps = [
        _step("V", "TLW", "seed"),
        _step("TLW", "X", "h1"),
        _step("X", "Y", "h2"),
    ]
    result = _trace_with_path(steps, "Reached terminal entity")
    out = postprocess_trace_result(result)
    assert len(out.paths) == 1
    assert out.paths[0].stop_reason == "Reached terminal entity"


def test_split_with_no_stop_reason_stays_none():
    """If the original path has no stop_reason, neither should any of
    its splits — the propagation guard must not fabricate one."""
    steps = [
        _step("V", "A", "seed"),
        _step("V", "B", "sw1"),  # from=V != prev_to=A → SPLIT
    ]
    result = _trace_with_path(steps, None)
    out = postprocess_trace_result(result)
    assert len(out.paths) == 2
    for p in out.paths:
        assert p.stop_reason is None


def test_dust_step_at_path_tail_only_marks_its_split():
    """A path ending in a dust step but with non-dust siblings earlier:
    the dust stop_reason must only land on the tail-containing split,
    so the visualization's dust-trim doesn't drop the legitimate
    siblings' last steps."""
    steps = [
        _step("V", "Mule", "seed"),
        _step("Mule", "Big", "h1", amount=100000.0),   # large
        _step("Mule", "Big2", "h2", amount=50000.0),   # large
        _step("Mule", "Tiny", "h3", amount=1.0),       # dust
    ]
    result = _trace_with_path(steps, "Below dust threshold (0.0002% of stolen amount)")
    out = postprocess_trace_result(result)

    # 3 splits: [seed, h1], [h2], [h3].
    assert len(out.paths) == 3
    tail_path = next(p for p in out.paths if any(s.tx_hash == "h3" for s in p.steps))
    assert tail_path.stop_reason and tail_path.stop_reason.startswith("Below dust threshold")
    other_paths = [p for p in out.paths if p is not tail_path]
    for p in other_paths:
        assert p.stop_reason is None
