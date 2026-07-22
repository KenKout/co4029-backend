"""Property / invariant tests for the interviewer decision policy (Slice 1).

These tests assert the *invariants* the adaptive spec requires, exercised across
a parameterized matrix of intent × answer-quality × time-remaining ×
probe-budget × question-pool state. Unlike ``test_interview_decision_selection``
(which pins specific example decisions), this suite proves properties that must
hold for EVERY reachable ``decide_next_action`` output.

``decide_next_action`` is pure (no DB, no LLM), so the whole reachable input
space is enumerable with plain objects.

Invariants under test (adaptive spec §Test Plan → property/invariant):
  I1. Exactly one primary action per turn.
  I2. Non-academic requests never record academic evidence.
  I3. A turn never both advances the question AND probes the same one.
  I4. No decision creates an infinite probe loop (budget is always respected).
  I5. Low-confidence / weak analysis cannot itself begin closing early.
  I6. Ending only ever arises from an explicit end intent, exhausted pool,
      low time, or full outcome coverage — never invented from a normal answer
      while questions remain and time is ample.
  I7. When the question pool is exhausted, the only forward move is closing
      (never a phantom advance to a non-existent question).

Selection-side invariants (``select_next_question``) live at the bottom:
  S1. A returned candidate is never an already-asked or skipped question.
  S2. Selection never repeats a question (idempotent under re-query).
"""

from __future__ import annotations

import itertools

import pytest

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
)
from abridgeai.features.interviews.orchestrator.decision import (
    DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
    DEFAULT_MAX_TOTAL_FOLLOWUPS,
    DecisionInputs,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.selection import (
    CandidateQuestion,
    SelectionContext,
    select_next_question,
)

# ── input-space builders ─────────────────────────────────────────────────────

# Actions that move OFF the current question (advance to next or close). Mirrors
# adaptive.ADVANCE_ACTIONS conceptually but derived from decision semantics so
# this test stays independent of the wiring layer.
_ADVANCE_ACTIONS: frozenset[InterviewerActionType] = frozenset(
    {
        InterviewerActionType.TRANSITION_TOPIC,
        InterviewerActionType.SKIP_QUESTION,
    }
)

_CLOSING_ACTIONS: frozenset[InterviewerActionType] = frozenset(
    {
        InterviewerActionType.BEGIN_CLOSING,
        InterviewerActionType.CLOSE_INTERVIEW,
    }
)

# Actions that keep the SAME question in play (probes / assistance / redirect).
_STAY_ACTIONS: frozenset[InterviewerActionType] = frozenset(
    {
        InterviewerActionType.PROBE_DEEPER,
        InterviewerActionType.ASK_FOR_EXAMPLE,
        InterviewerActionType.CHALLENGE_REASONING,
        InterviewerActionType.EXPLORE_TRADEOFF,
        InterviewerActionType.RESOLVE_CONTRADICTION,
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        InterviewerActionType.PROVIDE_NEUTRAL_HINT,
        InterviewerActionType.REPEAT_QUESTION,
        InterviewerActionType.REFRAME_QUESTION,
        InterviewerActionType.REDIRECT_TO_TOPIC,
        InterviewerActionType.OFFER_BRIEF_PAUSE,
        InterviewerActionType.HANDLE_TECHNICAL_ISSUE,
    }
)

_NON_ACADEMIC_INTENTS: tuple[StudentIntent, ...] = (
    StudentIntent.ASK_TO_REPEAT,
    StudentIntent.ASK_FOR_CLARIFICATION,
    StudentIntent.ASK_FOR_HINT,
    StudentIntent.ASK_FOR_MORE_TIME,
    StudentIntent.TECHNICAL_ISSUE,
    StudentIntent.END_INTERVIEW,
    StudentIntent.SKIP_QUESTION,
)

_ACADEMIC_INTENTS: tuple[StudentIntent, ...] = (
    StudentIntent.ANSWER,
    StudentIntent.PARTIAL_ANSWER,
)


def _intent(kind: StudentIntent, confidence: float = 0.9) -> IntentClassification:
    return IntentClassification(intent=kind, confidence=confidence, rationale="test")


def _analysis(
    *,
    relevance: Relevance = Relevance.RELEVANT,
    correctness: Correctness = Correctness.MOSTLY_CORRECT,
    probe: ProbeType = ProbeType.NONE,
    confidence: float = 0.7,
) -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=relevance,
        correctness=correctness,
        recommended_probe_type=probe,
        confidence=confidence,
    )


def _inputs(**overrides: object) -> DecisionInputs:
    base: dict[str, object] = {
        "intent": _intent(StudentIntent.ANSWER),
        "analysis": _analysis(),
        "current_question_follow_up_count": 0,
        "total_follow_up_count": 0,
        "time_fraction_remaining": 0.8,
        "has_next_question": True,
        "all_required_outcomes_covered": False,
    }
    base.update(overrides)
    return DecisionInputs(**base)  # type: ignore[arg-type]


# The full reachable input matrix. Kept modest per-axis so the cartesian product
# stays fast (< a few thousand cases) while still covering every branch.
_INTENTS = (
    _ACADEMIC_INTENTS
    + _NON_ACADEMIC_INTENTS
    + (
        StudentIntent.CANNOT_ANSWER,
        StudentIntent.OFF_TOPIC,
    )
)
_TIME_FRACTIONS: tuple[float | None, ...] = (None, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0)
_PER_Q_FOLLOWUPS = (0, 1, DEFAULT_MAX_FOLLOWUPS_PER_QUESTION)
_TOTAL_FOLLOWUPS = (0, DEFAULT_MAX_TOTAL_FOLLOWUPS)
_HAS_NEXT = (True, False)
_ALL_COVERED = (True, False)
_PROBES = (ProbeType.NONE, ProbeType.ASK_FOR_EXAMPLE, ProbeType.PROBE_REASONING)


def _matrix() -> list[DecisionInputs]:
    cases: list[DecisionInputs] = []
    for intent, tf, pq, tot, hn, cov, probe in itertools.product(
        _INTENTS,
        _TIME_FRACTIONS,
        _PER_Q_FOLLOWUPS,
        _TOTAL_FOLLOWUPS,
        _HAS_NEXT,
        _ALL_COVERED,
        _PROBES,
    ):
        analysis = _analysis(probe=probe) if intent in _ACADEMIC_INTENTS else _analysis()
        cases.append(
            _inputs(
                intent=_intent(intent),
                analysis=analysis,
                time_fraction_remaining=tf,
                current_question_follow_up_count=pq,
                total_follow_up_count=tot,
                has_next_question=hn,
                all_required_outcomes_covered=cov,
            )
        )
    return cases


_ALL_CASES = _matrix()


def _primary_action_count(d: InterviewerDecision) -> int:
    """A decision advances XOR stays XOR closes — exactly one 'primary' move."""
    advances = d.action in _ADVANCE_ACTIONS
    closes = d.action in _CLOSING_ACTIONS
    stays = d.action in _STAY_ACTIONS
    opening = d.action is InterviewerActionType.OPENING
    ask_main = d.action is InterviewerActionType.ASK_MAIN_QUESTION
    acknowledge = d.action is InterviewerActionType.ACKNOWLEDGE
    transition = d.action is InterviewerActionType.TRANSITION_TOPIC
    return sum([advances or transition, closes, stays, opening, ask_main, acknowledge])


# ── I1: exactly one primary action per turn ──────────────────────────────────


def test_every_decision_has_exactly_one_recognised_action() -> None:
    for inp in _ALL_CASES:
        d = decide_next_action(inp)
        assert isinstance(d.action, InterviewerActionType)
        assert isinstance(d.reason_code, ReasonCode)


def test_advance_and_stay_are_mutually_exclusive() -> None:
    # I3 — a decision never both advances the question AND keeps probing it.
    for inp in _ALL_CASES:
        d = decide_next_action(inp)
        advancing = d.action in _ADVANCE_ACTIONS or d.action in _CLOSING_ACTIONS
        staying = d.action in _STAY_ACTIONS
        assert not (advancing and staying), f"action {d.action} both advances and stays"


def test_should_advance_flag_consistent_with_action() -> None:
    # A stay-action must never set should_advance_question; closing never advances.
    for inp in _ALL_CASES:
        d = decide_next_action(inp)
        if d.action in _STAY_ACTIONS:
            assert d.should_advance_question is False, f"{d.action} set advance flag"
        if d.action in _CLOSING_ACTIONS:
            assert d.should_advance_question is False, "closing must not advance"


# ── I2: non-academic requests never record academic evidence ─────────────────


@pytest.mark.parametrize("intent", _NON_ACADEMIC_INTENTS)
def test_non_academic_requests_never_record_evidence(intent: StudentIntent) -> None:
    for tf, pq, hn in itertools.product(_TIME_FRACTIONS, _PER_Q_FOLLOWUPS, _HAS_NEXT):
        d = decide_next_action(
            _inputs(
                intent=_intent(intent),
                time_fraction_remaining=tf,
                current_question_follow_up_count=pq,
                has_next_question=hn,
            )
        )
        assert d.should_record_academic_evidence is False, (
            f"non-academic intent {intent} recorded evidence"
        )


def test_repeat_clarify_hint_never_consume_probe_budget_semantics() -> None:
    # Repeat / clarify / hint keep the same question and must not be scored,
    # so they can never advance either (they are pure assistance).
    for intent in (
        StudentIntent.ASK_TO_REPEAT,
        StudentIntent.ASK_FOR_CLARIFICATION,
        StudentIntent.ASK_FOR_HINT,
    ):
        d = decide_next_action(_inputs(intent=_intent(intent)))
        assert d.should_record_academic_evidence is False
        assert d.should_advance_question is False


# ── I4: probe budget is always respected (no infinite loop) ──────────────────


def test_probe_never_issued_once_per_question_budget_exhausted() -> None:
    for probe in (ProbeType.ASK_FOR_EXAMPLE, ProbeType.PROBE_REASONING):
        d = decide_next_action(
            _inputs(
                analysis=_analysis(probe=probe),
                current_question_follow_up_count=DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
            )
        )
        assert d.action not in _STAY_ACTIONS or d.action in (
            InterviewerActionType.REPEAT_QUESTION,
        ), "probe issued despite per-question budget exhausted"
        assert d.should_advance_question is True


def test_probe_never_issued_once_total_budget_exhausted() -> None:
    for probe in (ProbeType.ASK_FOR_EXAMPLE, ProbeType.PROBE_REASONING):
        d = decide_next_action(
            _inputs(
                analysis=_analysis(probe=probe),
                total_follow_up_count=DEFAULT_MAX_TOTAL_FOLLOWUPS,
            )
        )
        assert d.should_advance_question is True
        assert d.reason_code is ReasonCode.FOLLOWUP_LIMIT_REACHED


def test_academic_answers_bounded_probe_then_advance() -> None:
    # Across the whole academic matrix, any probing decision must have budget
    # left; otherwise it must advance. This is the loop-freedom invariant.
    for inp in _ALL_CASES:
        if inp.intent.intent not in _ACADEMIC_INTENTS:
            continue
        d = decide_next_action(inp)
        is_probe = d.action in (
            InterviewerActionType.PROBE_DEEPER,
            InterviewerActionType.ASK_FOR_EXAMPLE,
            InterviewerActionType.CHALLENGE_REASONING,
            InterviewerActionType.EXPLORE_TRADEOFF,
            InterviewerActionType.RESOLVE_CONTRADICTION,
        )
        if is_probe:
            assert inp.current_question_follow_up_count < DEFAULT_MAX_FOLLOWUPS_PER_QUESTION
            assert inp.total_follow_up_count < DEFAULT_MAX_TOTAL_FOLLOWUPS


# ── I5 / I6: closing is never invented from a normal answer ──────────────────


def test_strong_answer_ample_time_and_pool_does_not_close() -> None:
    # A good answer, plenty of time, questions remaining, not all covered →
    # must NOT begin closing. Closing here would be an early-exit bug.
    for probe in _PROBES:
        d = decide_next_action(
            _inputs(
                intent=_intent(StudentIntent.ANSWER),
                analysis=_analysis(correctness=Correctness.CORRECT, probe=probe, confidence=0.9),
                time_fraction_remaining=0.9,
                has_next_question=True,
                all_required_outcomes_covered=False,
                current_question_follow_up_count=0,
                total_follow_up_count=0,
            )
        )
        assert d.action not in _CLOSING_ACTIONS, "closed early on a strong answer"


def test_low_confidence_answer_never_triggers_all_covered_closing() -> None:
    # A low-confidence answer must not, by itself, flip to
    # ALL_REQUIRED_OUTCOMES_COVERED closing — coverage is decided by the caller's
    # all_required_outcomes_covered input, not by answer confidence.
    d = decide_next_action(
        _inputs(
            analysis=_analysis(correctness=Correctness.INCORRECT, confidence=0.1),
            all_required_outcomes_covered=False,
            time_fraction_remaining=0.9,
            has_next_question=True,
        )
    )
    assert d.reason_code is not ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED


def test_closing_reason_is_always_justified() -> None:
    # I6 — every closing decision must be attributable to a legitimate trigger:
    # explicit end intent, exhausted pool, low/at time, or full coverage.
    for inp in _ALL_CASES:
        d = decide_next_action(inp)
        if d.action not in _CLOSING_ACTIONS:
            continue
        legit = (
            inp.intent.intent is StudentIntent.END_INTERVIEW
            or inp.has_next_question is False
            or (inp.time_fraction_remaining is not None and inp.time_fraction_remaining <= 0.2)
            or inp.all_required_outcomes_covered is True
        )
        assert legit, (
            f"closing without justification: intent={inp.intent.intent} "
            f"time={inp.time_fraction_remaining} has_next={inp.has_next_question} "
            f"covered={inp.all_required_outcomes_covered} reason={d.reason_code}"
        )


# ── I7: exhausted pool ⇒ only forward move is closing ─────────────────────────


def test_no_next_question_never_advances() -> None:
    for inp in _ALL_CASES:
        if inp.has_next_question:
            continue
        d = decide_next_action(inp)
        # With no next question, a genuine-answer turn can still stay (probe) on
        # the CURRENT question, but it must never claim to advance to a
        # non-existent one.
        if d.should_advance_question:
            pytest.fail(f"advanced with empty pool: intent={inp.intent.intent} action={d.action}")


# ── selection-side invariants ────────────────────────────────────────────────


def _q(qid: str, *, outcome: str | None, pos: int | None) -> CandidateQuestion:
    return CandidateQuestion(
        question_id=qid,
        linked_outcome_id=outcome,
        question_type="conceptual",
        difficulty="mid_level",
        position=pos,
    )


def test_selection_never_returns_asked_or_skipped() -> None:
    # S1 — across combinations of asked/skipped sets, a pick is never one of them.
    candidates = [_q(f"q{i}", outcome=f"o{i}", pos=i) for i in range(1, 6)]
    ids = [c.question_id for c in candidates]
    for asked_n, skipped_n in itertools.product(range(len(ids) + 1), repeat=2):
        asked = frozenset(ids[:asked_n])
        skipped = frozenset(ids[len(ids) - skipped_n :])
        ctx = SelectionContext(
            asked_question_ids=asked,
            skipped_question_ids=skipped,
            outcome_evidence_counts={},
            uncovered_required_outcome_ids=frozenset(),
        )
        picked = select_next_question(candidates, ctx)
        if picked is not None:
            assert picked.candidate.question_id not in asked
            assert picked.candidate.question_id not in skipped


def test_selection_is_deterministic_and_non_repeating() -> None:
    # S2 — same inputs → same pick (deterministic); and simulating asking the
    # picked question never re-selects it, so the walk terminates (no loop).
    candidates = [_q(f"q{i}", outcome=f"o{i % 3}", pos=i) for i in range(1, 8)]
    asked: set[str] = set()
    seen: set[str] = set()
    for _ in range(len(candidates) + 2):
        ctx = SelectionContext(
            asked_question_ids=frozenset(asked),
            skipped_question_ids=frozenset(),
            outcome_evidence_counts={},
            uncovered_required_outcome_ids=frozenset(),
        )
        first = select_next_question(candidates, ctx)
        second = select_next_question(candidates, ctx)
        # Determinism: two calls with identical state agree.
        assert (first is None) == (second is None)
        if first is None or second is None:
            break
        assert first.candidate.question_id == second.candidate.question_id
        qid = first.candidate.question_id
        assert qid not in seen, "selection repeated an already-picked question"
        seen.add(qid)
        asked.add(qid)
    # Every question was eventually offered exactly once, then the pool drained.
    assert len(seen) == len(candidates)
