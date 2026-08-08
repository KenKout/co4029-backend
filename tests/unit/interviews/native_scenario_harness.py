"""Replay harness for the native interview agent's server-authoritative gates.

Voice quality is not testable; the GUARANTEES are. A scenario here declares a
sequence of candidate turns plus the move the model attempts after each one, the
harness replays it against the REAL functions, and the test asserts on the FINAL
STATE — coverage points, tick status, the hint ladder, the refusal counters,
whether each gate allowed the move. Never on generated wording, because the words
belong to the LLM and the words are the one thing that legitimately changes.

Shape borrowed from ``reference/agents/examples/hotel_receptionist``: declare a
scenario, replay it, grade the end state, veto on divergence. The difference is
that the hotel example grades a real DB after a real LLM run; here there is NO
LLM and NO DB. The probe verdict is injected per turn exactly as
``tests/unit/test_interview_native_grading.py`` does, so a scenario is
deterministic and a regression in any gate fails it.

What is deliberately REAL (a scenario that re-implemented these would prove
nothing):

* ``grade_native_turn`` — the probe → fold → enqueue → save ordering
* ``resolve_next_question`` / ``resolve_end_interview`` / ``resolve_hint_request``
  / ``reset_for_new_question`` — the gates, including their counter side effects
* ``build_turn_reminder`` — the string injected into the LLM context each turn
* ``shadow_check_turn`` — run every turn, as ``native_runtime`` does
* ``hard_stop_deadline_seconds`` — the wall-clock arithmetic, without ``sleep``

Only the sufficiency probe is faked, and it is faked at the narrowest possible
seam: one async callable returning one verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.shadow import shadow_check_turn
from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
from abridgeai.features.interviews.orchestrator.sufficiency import SufficiencyVerdict
from abridgeai.features.interviews.orchestrator.tools import (
    EndInterviewVerdict,
    build_turn_reminder,
    reset_for_new_question,
    resolve_end_interview,
    resolve_hint_request,
    resolve_next_question,
)
from abridgeai.features.interviews.realtime.native_grading import grade_native_turn

# Upper bound on how many times a termination probe will retry before the harness
# calls the interview un-terminable. Deliberately far above MAX_END_REFUSALS: the
# point is to catch a gate that refuses FOREVER, not to re-assert the bound's
# exact value, which the scenarios read from the real constant.
_TERMINATION_ATTEMPT_LIMIT = 20


class Move(str, Enum):  # noqa: UP042 -- StrEnum changes value coercion; match codebase convention
    """What the model attempts after the candidate's turn.

    ``PROBE`` is the common turn: the model just keeps talking and calls no tool.
    The other three are the tool calls the gates adjudicate.
    """

    PROBE = "probe"
    HINT = "hint"
    ADVANCE = "advance"
    END = "end"


@dataclass(frozen=True)
class Bank:
    """The question pool and syllabus one scenario runs against.

    ``question_outcomes`` is one entry per bank question, holding the outcome that
    question targets. ``""`` is the real value for a question with no linked
    outcome (42 of 59 questions in this deployment), so a scenario can declare
    that case without a special flag.
    """

    required_outcome_ids: tuple[str, ...]
    question_outcomes: tuple[str, ...]
    outcome_titles: Mapping[str, str] = field(default_factory=dict)
    max_follow_ups_per_question: int = 2
    time_remaining_seconds: int | None = 900
    below_closing_threshold: bool = False


@dataclass(frozen=True)
class Turn:
    """One candidate turn plus the model's attempted move after it.

    ``verdict`` is what the sufficiency probe WOULD return for this answer. It is
    declared even for turns where the probe must never run (a blank transcript),
    so a scenario can prove the call was not spent rather than merely absent.
    """

    says: str
    verdict: SufficiencyVerdict = field(default_factory=SufficiencyVerdict)
    then: Move = Move.PROBE
    probe_unreachable: bool = False


@dataclass(frozen=True)
class TurnRecord:
    """What the gates did on one replayed turn."""

    index: int
    move: Move
    allowed: bool | None
    refusal_message: str
    reminder: str
    probe_spent: bool
    hint_level_after: int


@dataclass
class Replay:
    """The final state of a replayed scenario, plus the per-turn trail."""

    bank: Bank
    state: InterviewRuntimeStateData
    records: list[TurnRecord]
    probe_answers: list[str]
    enqueued: list[dict[str, Any]]
    saves: int
    questions_remaining: int
    ended: bool

    # ── final-state readers: what a scenario asserts on ──────────────────────

    def points(self, outcome_id: str) -> int:
        coverage = self.state.outcome_coverage.get(outcome_id)
        return coverage.coverage_points if coverage is not None else 0

    def ticked(self, outcome_id: str) -> bool:
        return self.points(outcome_id) >= COVERAGE_SUFFICIENT_POINTS

    def all_required_ticked(self) -> bool:
        return all(self.ticked(oid) for oid in self.bank.required_outcome_ids)

    def supporting_turn_ids(self, outcome_id: str) -> list[str]:
        coverage = self.state.outcome_coverage.get(outcome_id)
        return list(coverage.supporting_turn_ids) if coverage is not None else []

    def moves(self, move: Move) -> list[TurnRecord]:
        return [r for r in self.records if r.move is move]

    def allowed_for(self, move: Move) -> list[bool | None]:
        return [r.allowed for r in self.moves(move)]

    def reminders(self) -> list[str]:
        return [r.reminder for r in self.records]

    def probes_spent(self) -> int:
        return sum(1 for r in self.records if r.probe_spent)

    def end_allowed_now(self) -> bool:
        """Whether ``end_interview`` would be permitted, without burning budget."""
        return _end_verdict(_detach(self.state), self).allowed

    def refusals_before_end_allowed(self) -> int:
        """How many more refusals stand between the model and a closed session.

        Drives the REAL ``resolve_end_interview`` on a detached copy, so the live
        refusal budget is untouched and the answer is the gate's, not the
        harness's. Raises when the gate never gives way — that failure IS the
        deadlock this harness exists to catch.
        """
        probe = _detach(self.state)
        for attempt in range(_TERMINATION_ATTEMPT_LIMIT):
            if _end_verdict(probe, self).allowed:
                return attempt
        raise AssertionError(
            f"end_interview refused {_TERMINATION_ATTEMPT_LIMIT} times in a row — "
            "the candidate cannot get out of this session by ending it"
        )


def _detach(state: InterviewRuntimeStateData) -> InterviewRuntimeStateData:
    return InterviewRuntimeStateData.from_dict(state.to_dict())


def _end_verdict(state: InterviewRuntimeStateData, replay: Replay) -> EndInterviewVerdict:
    return resolve_end_interview(
        state,
        required_outcome_ids=list(replay.bank.required_outcome_ids),
        questions_remaining=replay.questions_remaining,
        below_closing_threshold=replay.bank.below_closing_threshold,
        outcome_titles=dict(replay.bank.outcome_titles),
    )


class _Probe:
    """The one faked seam: an async callable returning a declared verdict."""

    def __init__(self) -> None:
        self.answers: list[str] = []
        self._verdict = SufficiencyVerdict()
        self._unreachable = False

    def arm(self, turn: Turn) -> None:
        self._verdict = turn.verdict
        self._unreachable = turn.probe_unreachable

    async def __call__(self, **kwargs: Any) -> SufficiencyVerdict:
        self.answers.append(str(kwargs.get("answer_text", "")))
        if self._unreachable:
            raise RuntimeError("sufficiency gateway unreachable")
        return self._verdict


async def replay(bank: Bank, turns: Sequence[Turn]) -> Replay:
    """Replay a candidate's turns against the real gates. No LLM, no DB.

    The turn order mirrors ``native_runtime.on_user_turn_completed``: grade first
    (so the reminder describes THIS turn, not the previous one), then refresh the
    question count, then run the shadow check, then build the reminder. The
    attempted move is adjudicated last, which is where the tool call would land.
    """
    state = InterviewRuntimeStateData()
    asked = 1 if bank.question_outcomes else 0
    state.current_outcome_id = bank.question_outcomes[0] if bank.question_outcomes else None
    probe = _Probe()
    result = Replay(
        bank=bank,
        state=state,
        records=[],
        probe_answers=probe.answers,
        enqueued=[],
        saves=0,
        questions_remaining=len(bank.question_outcomes) - asked,
        ended=False,
    )

    for index, turn in enumerate(turns):
        probe.arm(turn)
        before = len(probe.answers)
        await grade_native_turn(
            state=state,
            answer_text=turn.says,
            question_text="(bank question text)",
            turn_id=f"t{index}",
            probe=probe,
            enqueue_reconcile=_recorder(result.enqueued),
            save_state=_saver(result),
        )
        _assert_shadow_is_read_only(state, result)
        reminder = build_turn_reminder(
            state,
            current_outcome_id=state.current_outcome_id,
            required_outcome_ids=list(bank.required_outcome_ids),
            questions_remaining=result.questions_remaining,
            max_follow_ups_per_question=bank.max_follow_ups_per_question,
            below_closing_threshold=bank.below_closing_threshold,
            outcome_titles=dict(bank.outcome_titles),
            time_remaining_seconds=bank.time_remaining_seconds,
        )
        allowed, message = _apply_move(turn.then, state, result)
        if turn.then is Move.ADVANCE and allowed:
            asked += 1
            result.questions_remaining = max(0, len(bank.question_outcomes) - asked)
            state.current_outcome_id = (
                bank.question_outcomes[asked - 1] if asked <= len(bank.question_outcomes) else None
            )
            reset_for_new_question(state)
        result.records.append(
            TurnRecord(
                index=index,
                move=turn.then,
                allowed=allowed,
                refusal_message=message,
                reminder=reminder,
                probe_spent=len(probe.answers) > before,
                hint_level_after=state.hint_level,
            )
        )
    return result


def _apply_move(
    move: Move, state: InterviewRuntimeStateData, result: Replay
) -> tuple[bool | None, str]:
    """Adjudicate the model's attempted tool call through the real gate."""
    if move is Move.PROBE:
        return None, ""
    if move is Move.HINT:
        grant = resolve_hint_request(state)
        return grant.granted, "" if grant.granted else "hint ladder spent"
    if move is Move.ADVANCE:
        verdict = resolve_next_question(
            state,
            current_outcome_id=state.current_outcome_id,
            questions_remaining=result.questions_remaining,
            below_closing_threshold=result.bank.below_closing_threshold,
            max_follow_ups_per_question=result.bank.max_follow_ups_per_question,
        )
        return verdict.allowed, verdict.message
    end = _end_verdict(state, result)
    if end.allowed:
        result.ended = True
    return end.allowed, end.message


def _assert_shadow_is_read_only(state: InterviewRuntimeStateData, result: Replay) -> None:
    """The audit pass runs on every graded turn; it must never move the state."""
    snapshot = state.to_dict()
    shadow_check_turn(
        state=state,
        intent=IntentClassification(
            intent=StudentIntent.ANSWER, confidence=0.0, rationale="harness"
        ),
        model_advanced=False,
        questions_remaining=result.questions_remaining,
        time_fraction_remaining=None,
    )
    assert state.to_dict() == snapshot, "shadow_check_turn mutated a graded interview's state"


def _recorder(sink: list[dict[str, Any]]) -> Any:
    async def enqueue(**kwargs: Any) -> None:
        sink.append(kwargs)

    return enqueue


def _saver(result: Replay) -> Any:
    async def save() -> None:
        result.saves += 1

    return save


# ── verdict constructors, so a scenario reads as behaviour not as arithmetic ──


def ticks(outcome_id: str, *, confidence: float = 0.9) -> SufficiencyVerdict:
    """A confident answer that fully demonstrates the outcome (2 points)."""
    return SufficiencyVerdict(
        sufficient=True, outcome_ids_touched=[outcome_id], confidence=confidence
    )


def partial(outcome_id: str, *, confidence: float = 0.9) -> SufficiencyVerdict:
    """A confident answer that only partly demonstrates the outcome (1 point)."""
    return SufficiencyVerdict(
        sufficient=False, outcome_ids_touched=[outcome_id], confidence=confidence
    )


def demonstrates_nothing(*, confidence: float = 0.9) -> SufficiencyVerdict:
    """The probe's confident report that the answer showed no outcome at all."""
    return SufficiencyVerdict(sufficient=False, outcome_ids_touched=[], confidence=confidence)


def suborned(outcome_id: str) -> SufficiencyVerdict:
    """A probe that has DONE WHAT THE INJECTED TEXT ASKED and claims sufficiency.

    Used to show the allowlist is the real defence: even a fully compromised probe
    cannot tick an outcome it was not handed.
    """
    return SufficiencyVerdict(sufficient=True, outcome_ids_touched=[outcome_id], confidence=1.0)


__all__ = [
    "Bank",
    "Move",
    "Replay",
    "Turn",
    "TurnRecord",
    "demonstrates_nothing",
    "partial",
    "replay",
    "suborned",
    "ticks",
]
