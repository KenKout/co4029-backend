"""Tests for grading the candidate's answer on the NATIVE turn path.

This closes the seam between two workstreams that each correctly left it to the
other: the native agent injected a state note but graded nothing, and the fast
sufficiency probe existed but nothing called it. Without this, `outcome_coverage`
never moves on a spoken turn, so the note says "NOT yet covered" forever and the
only thing that ever ends the interview is the hard-stop timer.

Pinned here:
  * the probe runs BEFORE the reminder is built, or the note describes the state
    as it was one turn ago
  * a probe failure must not block the turn, and must not invent coverage
  * the full re-analysis is enqueued, not awaited — grading is allowed minutes
  * runtime state is PERSISTED, or a rejoin resets the hint ladder and the
    refusal budgets and the anti-deadlock bounds silently reset with them
"""

from __future__ import annotations

from typing import Any

import pytest

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.orchestrator.sufficiency import SufficiencyVerdict
from abridgeai.features.interviews.realtime.native_grading import grade_native_turn


class _Recorder:
    """Captures what the turn tried to do, without a DB or a gateway."""

    def __init__(self, verdict: SufficiencyVerdict | None, *, raises: bool = False) -> None:
        self.verdict = verdict
        self.raises = raises
        self.probe_calls: list[dict[str, Any]] = []
        self.enqueued: list[dict[str, Any]] = []
        self.saved = 0

    async def probe(self, **kwargs: Any) -> SufficiencyVerdict:
        self.probe_calls.append(kwargs)
        if self.raises:
            raise RuntimeError("gateway down")
        assert self.verdict is not None
        return self.verdict

    async def enqueue(self, **kwargs: Any) -> None:
        self.enqueued.append(kwargs)

    async def save(self) -> None:
        self.saved += 1


def _state(points: int = 0) -> InterviewRuntimeStateData:
    state = InterviewRuntimeStateData()
    state.outcome_coverage = {"o1": OutcomeCoverageState(outcome_id="o1", coverage_points=points)}
    state.current_outcome_id = "o1"
    return state


async def _grade(rec: _Recorder, state: InterviewRuntimeStateData) -> None:
    await grade_native_turn(
        state=state,
        answer_text="Operational handles transactions; informational supports decisions.",
        question_text="What is the difference?",
        turn_id="t1",
        probe=rec.probe,
        enqueue_reconcile=rec.enqueue,
        save_state=rec.save,
    )


# ── the probe must actually move coverage ─────────────────────────────────────


async def test_a_sufficient_answer_ticks_the_outcome() -> None:
    rec = _Recorder(SufficiencyVerdict(sufficient=True, outcome_ids_touched=["o1"], confidence=0.9))
    state = _state(points=0)
    await _grade(rec, state)

    assert state.outcome_coverage["o1"].coverage_points >= COVERAGE_SUFFICIENT_POINTS, (
        "a sufficient answer did not tick the outcome — the reminder will say "
        "'NOT yet covered' forever and only the hard stop will end the interview"
    )


async def test_an_insufficient_answer_records_partial_credit() -> None:
    rec = _Recorder(
        SufficiencyVerdict(sufficient=False, outcome_ids_touched=["o1"], confidence=0.9)
    )
    state = _state(points=0)
    await _grade(rec, state)

    points = state.outcome_coverage["o1"].coverage_points
    assert 0 < points < COVERAGE_SUFFICIENT_POINTS, (
        f"partial credit should advance without ticking; got {points}"
    )


async def test_an_answer_touching_nothing_awards_nothing() -> None:
    rec = _Recorder(SufficiencyVerdict(sufficient=False, outcome_ids_touched=[], confidence=0.9))
    state = _state(points=0)
    await _grade(rec, state)
    assert state.outcome_coverage["o1"].coverage_points == 0


# ── failure must be safe, not silent coverage ────────────────────────────────


async def test_a_probe_failure_does_not_block_the_turn_or_invent_coverage() -> None:
    rec = _Recorder(None, raises=True)
    state = _state(points=0)
    # Must not raise: a dead gateway cannot be allowed to kill a live interview.
    await _grade(rec, state)
    assert state.outcome_coverage["o1"].coverage_points == 0, "phantom coverage from a failed probe"


async def test_a_probe_failure_still_persists_state() -> None:
    # Tool mutations from THIS turn (hint level, refusal counters) must survive
    # even when grading failed, or a rejoin hands the model a fresh budget.
    rec = _Recorder(None, raises=True)
    await _grade(rec, _state())
    assert rec.saved == 1


# ── the expensive half is deferred, not awaited ───────────────────────────────


async def test_full_reanalysis_is_enqueued_with_the_probe_verdict() -> None:
    verdict = SufficiencyVerdict(sufficient=True, outcome_ids_touched=["o1"], confidence=0.9)
    rec = _Recorder(verdict)
    await _grade(rec, _state())

    assert len(rec.enqueued) == 1, "the full analysis was not deferred to the worker"
    payload = rec.enqueued[0]
    # The worker needs the probe's verdict to compute a delta; without it it
    # cannot revoke what the probe awarded.
    assert payload["turn_id"] == "t1"
    assert payload["probe_verdict"]["sufficient"] is True


async def test_nothing_is_enqueued_when_the_probe_failed() -> None:
    # With no verdict there is no delta to reconcile against.
    rec = _Recorder(None, raises=True)
    await _grade(rec, _state())
    assert rec.enqueued == []


# ── ordering ─────────────────────────────────────────────────────────────────


async def test_state_is_saved_after_coverage_moves() -> None:
    rec = _Recorder(SufficiencyVerdict(sufficient=True, outcome_ids_touched=["o1"], confidence=0.9))
    state = _state(points=0)
    await _grade(rec, state)
    # Saved once, and the saved state is the mutated one (same object).
    assert rec.saved == 1
    assert state.outcome_coverage["o1"].coverage_points >= COVERAGE_SUFFICIENT_POINTS


@pytest.mark.parametrize("answer", ["", "   "])
async def test_a_blank_answer_is_not_graded(answer: str) -> None:
    rec = _Recorder(SufficiencyVerdict(sufficient=True, outcome_ids_touched=["o1"], confidence=0.9))
    state = _state(points=0)
    await grade_native_turn(
        state=state,
        answer_text=answer,
        question_text="Q?",
        turn_id="t1",
        probe=rec.probe,
        enqueue_reconcile=rec.enqueue,
        save_state=rec.save,
    )
    assert rec.probe_calls == [], "a blank transcript must not cost an LLM call"
    assert state.outcome_coverage["o1"].coverage_points == 0


# ── the two "empty after filtering" cases are NOT the same ────────────────────


def test_a_hallucinated_outcome_id_awards_nothing() -> None:
    # A model that cannot repeat an id it was handed is not a model whose
    # "sufficient" verdict should move a grade.
    from abridgeai.features.interviews.orchestrator.sufficiency import verdict_to_evidence

    items = verdict_to_evidence(
        SufficiencyVerdict(
            sufficient=True, outcome_ids_touched=["not-a-real-outcome"], confidence=0.9
        ),
        turn_id="t1",
        target_outcome_id="o1",
        allowed_other=(),
    )
    assert items == []


def test_a_deliberately_empty_list_awards_nothing() -> None:
    # The model said the answer demonstrated NOTHING. In an assessment, inventing
    # credit here is strictly worse than missing it: two off-topic answers would
    # tick an outcome the candidate never demonstrated.
    from abridgeai.features.interviews.orchestrator.sufficiency import verdict_to_evidence

    items = verdict_to_evidence(
        SufficiencyVerdict(sufficient=False, outcome_ids_touched=[], confidence=0.9),
        turn_id="t1",
        target_outcome_id="o1",
        allowed_other=(),
    )
    assert items == [], "an off-topic answer earned phantom coverage"
