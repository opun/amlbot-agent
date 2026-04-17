"""Unit tests for :class:`HopScheduler`.

These tests encode the invariants that fix the "trace stops one hop before
the real CEX deposit" bug:

1. Larger ``attributed_amount`` branches are dequeued first, so high-value
   paths reach their CEX/ChangeNOW terminal before the FIFO cap closes on
   smaller sibling mules.
2. The outer loop is bounded by the count of *completed* paths, not by the
   number of processed HopJobs. Continuations never starve dead-ends, and
   dead-ends never starve continuations.
3. FIFO tiebreak keeps ordering deterministic for equal priorities.
4. The iteration safety net prevents runaway loops on pathological inputs.
"""

from __future__ import annotations

import pytest

from agent.base_tracer import HopJob, HopScheduler


def _job(path_id: str, amount: float, hop_index: int = 1) -> HopJob:
    return HopJob(
        path_id=path_id,
        current_address=f"0x{path_id:0>40}",
        incoming_tx_hash=None,
        incoming_amount=amount,
        incoming_time=None,
        chain="eth",
        asset="ETH",
        token_id=0,
        hop_index=hop_index,
        attributed_amount=amount,
    )


class TestPriorityOrdering:
    def test_largest_attributed_amount_first(self):
        scheduler = HopScheduler(max_completed=10)
        scheduler.push(_job("small", 1.0))
        scheduler.push(_job("big", 100.0))
        scheduler.push(_job("medium", 10.0))

        assert scheduler.pop().path_id == "big"
        assert scheduler.pop().path_id == "medium"
        assert scheduler.pop().path_id == "small"

    def test_tiebreak_by_hop_index_ascending(self):
        """Equal attribution -> shallower hop wins (BFS-ish for even splits)."""
        scheduler = HopScheduler(max_completed=10)
        scheduler.push(_job("deep", 10.0, hop_index=5))
        scheduler.push(_job("shallow", 10.0, hop_index=2))
        scheduler.push(_job("mid", 10.0, hop_index=3))

        assert scheduler.pop().path_id == "shallow"
        assert scheduler.pop().path_id == "mid"
        assert scheduler.pop().path_id == "deep"

    def test_fifo_tiebreak_for_identical_priority(self):
        """With identical (amount, hop_index), insertion order wins."""
        scheduler = HopScheduler(max_completed=10)
        scheduler.push(_job("first", 10.0, hop_index=2))
        scheduler.push(_job("second", 10.0, hop_index=2))
        scheduler.push(_job("third", 10.0, hop_index=2))

        assert scheduler.pop().path_id == "first"
        assert scheduler.pop().path_id == "second"
        assert scheduler.pop().path_id == "third"

    def test_zero_attributed_amount_is_lowest_priority(self):
        scheduler = HopScheduler(max_completed=10)
        scheduler.push(_job("no_attr", 0.0))
        scheduler.push(_job("tiny", 0.0001))

        assert scheduler.pop().path_id == "tiny"
        assert scheduler.pop().path_id == "no_attr"


class TestCompletionBudget:
    def test_should_continue_while_under_budget(self):
        scheduler = HopScheduler(max_completed=3)
        scheduler.push(_job("a", 10.0))
        scheduler.push(_job("b", 10.0))
        scheduler.push(_job("c", 10.0))

        assert scheduler.should_continue(completed_paths=0)
        assert scheduler.should_continue(completed_paths=2)
        assert not scheduler.should_continue(completed_paths=3)
        assert not scheduler.should_continue(completed_paths=4)

    def test_should_stop_when_queue_empty(self):
        scheduler = HopScheduler(max_completed=10)
        assert not scheduler.should_continue(completed_paths=0)

    def test_continuations_do_not_consume_budget(self):
        """Regression: previously, each HopJob iteration incremented
        ``processed_paths`` on termination but continuations (which enqueue
        hop N+1 jobs) were not counted. The scheduler must not conflate the
        two — continuations should let deeper terminals emerge."""
        scheduler = HopScheduler(max_completed=2)

        scheduler.push(_job("sibling_a", 10.0))
        scheduler.push(_job("sibling_b", 10.0))
        scheduler.push(_job("sibling_c", 10.0))
        scheduler.push(_job("sibling_d", 10.0))

        # Pop two jobs; caller decides they "continue" (enqueue hop N+1)
        # without marking any path completed. We simulate that by popping
        # and then pushing a new hop.
        first = scheduler.pop()
        assert first.path_id in {"sibling_a", "sibling_b", "sibling_c", "sibling_d"}
        scheduler.push(HopJob(
            path_id=first.path_id,
            current_address=f"0x{first.path_id}_hop2",
            incoming_tx_hash=None,
            incoming_amount=first.incoming_amount,
            incoming_time=None,
            chain=first.chain,
            asset=first.asset,
            token_id=0,
            hop_index=first.hop_index + 1,
            attributed_amount=first.attributed_amount,
        ))
        # After popping 1 and pushing a continuation, the scheduler should
        # still let us keep going — no path has completed yet.
        assert scheduler.should_continue(completed_paths=0)
        # Pop another three jobs, mark only one path complete — scheduler
        # should keep running for the rest.
        for _ in range(3):
            scheduler.pop()
        assert scheduler.should_continue(completed_paths=1)


class TestIterationSafetyNet:
    def test_exhausted_flag_triggers_after_cap(self):
        scheduler = HopScheduler(max_completed=10, max_iterations=3)
        scheduler.push(_job("a", 1.0))
        scheduler.push(_job("b", 1.0))
        scheduler.push(_job("c", 1.0))
        scheduler.push(_job("d", 1.0))

        assert not scheduler.exhausted
        scheduler.pop()
        scheduler.pop()
        scheduler.pop()
        assert scheduler.exhausted
        # `should_continue` must return False once exhausted, even if budget
        # isn't otherwise satisfied.
        assert not scheduler.should_continue(completed_paths=0)

    def test_default_safety_net_scales_with_max_completed(self):
        scheduler = HopScheduler(max_completed=10)
        # 10 * 64 = 640 by default; plenty of headroom for realistic traces
        # but still bounded.
        assert scheduler.max_iterations == 640

    def test_rejects_non_positive_max_completed(self):
        with pytest.raises(ValueError):
            HopScheduler(max_completed=0)
        with pytest.raises(ValueError):
            HopScheduler(max_completed=-1)


class TestSchedulerSizeAndIterations:
    def test_len_reflects_pending(self):
        scheduler = HopScheduler(max_completed=5)
        assert len(scheduler) == 0
        scheduler.push(_job("a", 1.0))
        scheduler.push(_job("b", 2.0))
        assert len(scheduler) == 2
        scheduler.pop()
        assert len(scheduler) == 1

    def test_iterations_counter_monotonic(self):
        scheduler = HopScheduler(max_completed=5)
        scheduler.push(_job("a", 1.0))
        scheduler.push(_job("b", 1.0))
        assert scheduler.iterations == 0
        scheduler.pop()
        assert scheduler.iterations == 1
        scheduler.pop()
        assert scheduler.iterations == 2


class TestRegressionScenario:
    """End-to-end scheduler invariants for the case the user reported.

    Scenario: the perpetrator splits ~94 ETH across sibling mules, most of
    which hand off to exchange deposits one hop further (hop 3 terminal).
    Under the old FIFO + ``processed_paths < max_paths`` scheduling, siblings
    that happened to dead-end at hop 2 would consume the budget before hop-3
    jobs ran, so the trace stopped on the mules and the visualization layer
    mislabelled them as "Exchange deposit address".
    """

    def test_continuations_reach_deeper_terminal_when_siblings_also_continue(self):
        """10 equal-size mules all continue to a CEX at hop 3. Under the old
        scheduler, the hop-2 mules would be processed FIFO and hop-3 jobs
        would be queued behind them, but ``processed_paths`` wouldn't have
        incremented for the continuations, so this case used to work. We
        keep it as a sanity regression: all 10 CEX terminals must be
        reached."""
        scheduler = HopScheduler(max_completed=10)

        for idx in range(10):
            scheduler.push(_job(f"mule_{idx}", 10.0, hop_index=2))

        completed = 0
        cex_terminals: list[str] = []
        while scheduler.should_continue(completed):
            job = scheduler.pop()
            if job.hop_index == 2:
                scheduler.push(HopJob(
                    path_id=job.path_id + "_cex",
                    current_address=f"0x{job.path_id}_cex",
                    incoming_tx_hash=None,
                    incoming_amount=job.incoming_amount,
                    incoming_time=None,
                    chain="eth",
                    asset="ETH",
                    token_id=0,
                    hop_index=3,
                    attributed_amount=job.attributed_amount,
                ))
            else:
                cex_terminals.append(job.path_id)
                completed += 1

        assert completed == 10
        assert len(cex_terminals) == 10

    def test_high_value_branches_reach_cex_before_dead_ends_eat_budget(self):
        """Priority ordering must ensure that when some siblings dead-end
        and some continue, the high-attribution continuations complete
        first. This is the exact inversion of the bug: under FIFO with
        ``processed_paths < max_paths``, the dead-ends consumed budget and
        the CEX branches never got their hop-3 turn."""
        scheduler = HopScheduler(max_completed=10)

        # 2 tiny dead-end siblings (low attribution) + 10 high-value mules
        # that will continue to CEX terminals.
        for idx in range(2):
            scheduler.push(_job(f"dead_{idx}", 0.1, hop_index=2))
        for idx in range(10):
            scheduler.push(_job(f"mule_{idx}", 10.0, hop_index=2))

        completed = 0
        cex_terminals: list[str] = []
        dead_ends: list[str] = []

        while scheduler.should_continue(completed):
            job = scheduler.pop()
            if job.hop_index == 2 and job.path_id.startswith("dead_"):
                dead_ends.append(job.path_id)
                completed += 1
            elif job.hop_index == 2:
                scheduler.push(HopJob(
                    path_id=job.path_id + "_cex",
                    current_address=f"0x{job.path_id}_cex",
                    incoming_tx_hash=None,
                    incoming_amount=job.incoming_amount,
                    incoming_time=None,
                    chain="eth",
                    asset="ETH",
                    token_id=0,
                    hop_index=3,
                    attributed_amount=job.attributed_amount,
                ))
            else:
                cex_terminals.append(job.path_id)
                completed += 1

        # The priority queue processed the 10 high-value mules first,
        # enqueued their CEX continuations, and those completed before
        # the 2 dead-ends were ever popped. Budget fills with 10 CEX
        # terminals — the dead-ends never get to consume budget.
        assert len(cex_terminals) == 10, cex_terminals
        assert len(dead_ends) == 0, dead_ends
        assert completed == 10
