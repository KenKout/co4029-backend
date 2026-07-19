"""Unit tests for the interviewer decision policy (Phase 4) and adaptive
question selection (Phase 5).

Both modules are pure (no DB, no LLM), so these tests exercise the full
decision precedence ladder and the scorer's ranking / fallback semantics
directly with plain objects.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
)
from abridgeai.features.interviews.orchestrator.decision import (
    DecisionInputs,
    InterviewerActionType,
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
    score_candidate,
    select_next_question,
    sequential_pick,
)


def _intent(kind: StudentIntent) -> IntentClassification:
    return IntentClassification(intent=kind, confidence=0.9, rationale="test")


def _analysis(
    *,
    relevance: Relevance = Relevance.RELEVANT,
    correctness: Correctness = Correctness.MOSTLY_CORRECT,
    probe: ProbeType = ProbeType.NONE,
) -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=relevance,
        correctness=correctness,
        recommended_probe_type=probe,
        confidence=0.7,
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


# ── decision precedence ──────────────────────────────────────────────────────


def test_technical_issue_never_scored() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.TECHNICAL_ISSUE)))
    assert d.action is InterviewerActionType.HANDLE_TECHNICAL_ISSUE
    assert d.should_record_academic_evidence is False
    assert d.should_advance_question is False


def test_repeat_request_repeats_without_scoring() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.ASK_TO_REPEAT)))
    assert d.action is InterviewerActionType.REPEAT_QUESTION
    assert d.reason_code is ReasonCode.STUDENT_REQUESTED_REPEAT
    assert d.should_record_academic_evidence is False
    assert d.should_advance_question is False


def test_clarification_does_not_leak_answer_and_not_scored() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.ASK_FOR_CLARIFICATION)))
    assert d.action is InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER
    assert d.should_record_academic_evidence is False


def test_skip_request_advances_and_marks_skipped() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.SKIP_QUESTION)))
    assert d.action is InterviewerActionType.SKIP_QUESTION
    assert d.should_advance_question is True
    assert "skipped" in d.tags


def test_cannot_answer_records_insufficient_and_advances() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.CANNOT_ANSWER)))
    assert d.should_advance_question is True
    assert d.should_record_academic_evidence is True
    assert "insufficient_evidence" in d.tags


def test_end_request_begins_closing() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.END_INTERVIEW)))
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.STUDENT_REQUESTED_END


def test_off_topic_redirects_once_then_advances() -> None:
    # First encounter (no follow-ups yet) → redirect.
    first = decide_next_action(
        _inputs(
            intent=_intent(StudentIntent.OFF_TOPIC),
            current_question_follow_up_count=0,
        )
    )
    assert first.action is InterviewerActionType.REDIRECT_TO_TOPIC
    # Persisting off-topic (already redirected once) → advance.
    second = decide_next_action(
        _inputs(
            intent=_intent(StudentIntent.OFF_TOPIC),
            current_question_follow_up_count=1,
        )
    )
    assert second.should_advance_question is True


def test_probe_recommended_probes_and_records_evidence() -> None:
    d = decide_next_action(_inputs(analysis=_analysis(probe=ProbeType.ASK_FOR_EXAMPLE)))
    assert d.action is InterviewerActionType.ASK_FOR_EXAMPLE
    assert d.reason_code is ReasonCode.MISSING_EXAMPLE
    assert d.should_record_academic_evidence is True
    assert d.should_advance_question is False


def test_probe_suppressed_when_followup_limit_reached() -> None:
    d = decide_next_action(
        _inputs(
            analysis=_analysis(probe=ProbeType.PROBE_REASONING),
            current_question_follow_up_count=2,  # == default cap
        )
    )
    # Limit hit → advance instead of probing.
    assert d.should_advance_question is True
    assert d.reason_code is ReasonCode.FOLLOWUP_LIMIT_REACHED


def test_probe_suppressed_when_time_low() -> None:
    d = decide_next_action(
        _inputs(
            analysis=_analysis(probe=ProbeType.PROBE_REASONING),
            time_fraction_remaining=0.15,  # below low-time threshold
        )
    )
    assert d.action is not InterviewerActionType.PROBE_DEEPER


def test_closing_threshold_time_wraps_up() -> None:
    d = decide_next_action(_inputs(time_fraction_remaining=0.05))
    assert d.reason_code is ReasonCode.TIME_RUNNING_LOW


def test_all_required_covered_begins_closing() -> None:
    d = decide_next_action(
        _inputs(
            analysis=_analysis(probe=ProbeType.NONE),
            all_required_outcomes_covered=True,
        )
    )
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED


def test_no_next_question_begins_closing() -> None:
    d = decide_next_action(
        _inputs(analysis=_analysis(probe=ProbeType.NONE), has_next_question=False)
    )
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.CLOSING_REQUIRED


def test_public_dict_excludes_internal_rationale() -> None:
    d = decide_next_action(_inputs())
    pub = d.to_public_dict()
    assert "internal_rationale" not in pub
    assert "should_record_academic_evidence" not in pub
    assert "action" in pub
    assert "reason_code" in pub
    # Audit dict keeps the internals.
    assert "internal_rationale" in d.to_audit_dict()


# ── adaptive selection ───────────────────────────────────────────────────────


def _q(
    qid: str,
    *,
    outcome: str | None,
    pos: int | None,
    weight: int = 1,
    difficulty: str | None = "mid_level",
) -> CandidateQuestion:
    return CandidateQuestion(
        question_id=qid,
        linked_outcome_id=outcome,
        question_type="conceptual",
        difficulty=difficulty,
        position=pos,
        importance_weight=weight,
    )


def test_selector_prioritises_uncovered_required_outcome() -> None:
    candidates = [
        _q("q1", outcome="o_covered", pos=1, weight=5),
        _q("q2", outcome="o_required", pos=2, weight=1),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o_covered": 3},  # already covered
        uncovered_required_outcome_ids=frozenset({"o_required"}),
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    # q2 wins despite lower weight + later position: it targets the uncovered
    # REQUIRED outcome, while q1's outcome is already amply covered (penalised).
    assert picked.candidate.question_id == "q2"


def test_selector_never_returns_asked_question() -> None:
    candidates = [_q("q1", outcome="o1", pos=1), _q("q2", outcome="o2", pos=2)]
    ctx = SelectionContext(
        asked_question_ids=frozenset({"q1"}),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    assert picked.candidate.question_id == "q2"


def test_selector_skips_skipped_questions() -> None:
    candidates = [_q("q1", outcome="o1", pos=1)]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset({"q1"}),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
    )
    assert select_next_question(candidates, ctx) is None


def test_selector_penalises_recent_outcome_fixation() -> None:
    candidates = [
        _q("q_same", outcome="o1", pos=1),
        _q("q_other", outcome="o2", pos=2),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
        last_targeted_outcome_id="o1",
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    # Both uncovered, but q_same repeats the last outcome → penalised → q_other.
    assert picked.candidate.question_id == "q_other"


def test_empty_pool_returns_none() -> None:
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
    )
    assert select_next_question([], ctx) is None


def test_sequential_fallback_preserves_teacher_order() -> None:
    candidates = [
        _q("q3", outcome=None, pos=3),
        _q("q1", outcome=None, pos=1),
        _q("q2", outcome=None, pos=2),
    ]
    # None asked → first by position.
    first = sequential_pick(candidates, frozenset())
    assert first is not None
    assert first.question_id == "q1"
    # q1 asked → q2 next.
    second = sequential_pick(candidates, frozenset({"q1"}))
    assert second is not None
    assert second.question_id == "q2"
    # all asked → None.
    assert sequential_pick(candidates, frozenset({"q1", "q2", "q3"})) is None


def test_score_breakdown_is_populated() -> None:
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset({"o1"}),
    )
    scored = score_candidate(_q("q1", outcome="o1", pos=1, weight=5), ctx)
    assert scored.breakdown["uncovered_outcome"] > 0
    assert scored.breakdown["required_outcome"] > 0
    assert scored.breakdown["importance"] == 40.0  # 8.0 * 5
    assert scored.score > 0
