"""Unit tests for phase threading in apply_state_updates (Slice 7, v2).

Pure state mutation — no DB, no LLM. Verifies that when the phase policy hands
apply_state_updates a target phase, it authoritatively drives data.phase and
maintains the turns_in_phase dwell counter; and that with target_phase=None the
legacy v1 transitions run unchanged (parity guard).
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.state import (
    InterviewPhase,
    InterviewRuntimeStateData,
)
from abridgeai.features.interviews.orchestrator.turn_state import apply_state_updates


def _answer_intent() -> IntentClassification:
    return IntentClassification(
        intent=StudentIntent.ANSWER, confidence=0.9, rationale="", source="rules"
    )


def _advance_decision() -> InterviewerDecision:
    return InterviewerDecision(
        action=InterviewerActionType.TRANSITION_TOPIC,
        reason_code=ReasonCode.OUTCOME_SUFFICIENTLY_COVERED,
        should_advance_question=True,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
    )


def test_target_phase_change_resets_dwell_counter() -> None:
    data = InterviewRuntimeStateData(phase=InterviewPhase.OPENING, turns_in_phase=5)
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        target_phase=InterviewPhase.WARMUP,
    )
    assert data.phase is InterviewPhase.WARMUP
    assert data.turns_in_phase == 0


def test_target_phase_same_increments_dwell_counter() -> None:
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE, turns_in_phase=2)
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        target_phase=InterviewPhase.CORE,
    )
    assert data.phase is InterviewPhase.CORE
    assert data.turns_in_phase == 3


def test_no_target_phase_preserves_v1_transitions() -> None:
    # v1 parity: with target_phase=None, an advance out of OPENING lands in CORE
    # (the legacy hardcoded transition) and never touches turns_in_phase.
    data = InterviewRuntimeStateData(phase=InterviewPhase.OPENING, turns_in_phase=0)
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        target_phase=None,
    )
    assert data.phase is InterviewPhase.CORE
    assert data.turns_in_phase == 0


def test_cross_turn_records_bounded_claims() -> None:
    """Slice 9: with cross_turn on, evidence summaries append to outcome claims,
    bounded to the last 3; with the flag off, no claims are recorded."""
    from abridgeai.features.interviews.orchestrator.analysis import (
        AnswerAnalysis,
        EvidenceType,
        OutcomeEvidence,
    )
    from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState

    def _analysis_with_claim(summary: str) -> AnswerAnalysis:
        return AnswerAnalysis(
            confidence=0.9,
            evidence=[
                OutcomeEvidence(
                    outcome_id="o-1",
                    turn_id="t-1",
                    evidence_type=EvidenceType.SUPPORTS,
                    summary=summary,
                    confidence=0.9,
                )
            ],
        )

    data = InterviewRuntimeStateData()
    data.outcome_coverage["o-1"] = OutcomeCoverageState(outcome_id="o-1")
    # Four claims, bounded to the last 3.
    for i in range(4):
        apply_state_updates(
            data,
            intent=_answer_intent(),
            analysis=_analysis_with_claim(f"claim {i}"),
            decision=_advance_decision(),
            selected_question_id="q-2",
            target_outcome_id="o-1",
            cross_turn_enabled=True,
        )
    assert data.outcome_coverage["o-1"].claims == ["claim 1", "claim 2", "claim 3"]


def test_cross_turn_off_records_no_claims() -> None:
    from abridgeai.features.interviews.orchestrator.analysis import (
        AnswerAnalysis,
        EvidenceType,
        OutcomeEvidence,
    )
    from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState

    data = InterviewRuntimeStateData()
    data.outcome_coverage["o-1"] = OutcomeCoverageState(outcome_id="o-1")
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=AnswerAnalysis(
            confidence=0.9,
            evidence=[
                OutcomeEvidence(
                    outcome_id="o-1",
                    turn_id="t-1",
                    evidence_type=EvidenceType.SUPPORTS,
                    summary="a claim",
                    confidence=0.9,
                )
            ],
        ),
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        cross_turn_enabled=False,
    )
    assert data.outcome_coverage["o-1"].claims == []


def _hint_decision() -> InterviewerDecision:
    return InterviewerDecision(
        action=InterviewerActionType.PROVIDE_NEUTRAL_HINT,
        reason_code=ReasonCode.STUDENT_REQUESTED_HINT,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
    )


def test_hint_ladder_increments_on_repeated_hint() -> None:
    """Slice 11: a hint on the same question escalates hint_level when enabled."""
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE)
    for _ in range(2):
        apply_state_updates(
            data,
            intent=_answer_intent(),
            analysis=None,
            decision=_hint_decision(),
            selected_question_id=None,
            target_outcome_id=None,
            hint_ladder_enabled=True,
        )
    assert data.hint_level == 2


def test_hint_ladder_off_leaves_level_zero() -> None:
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE)
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_hint_decision(),
        selected_question_id=None,
        target_outcome_id=None,
        hint_ladder_enabled=False,
    )
    assert data.hint_level == 0


def test_advance_resets_hint_ladder() -> None:
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE, hint_level=3, reframe_count=2)
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        hint_ladder_enabled=True,
    )
    assert data.hint_level == 0
    assert data.reframe_count == 0


def test_per_outcome_competence_updates_on_evidence() -> None:
    """Slice 12: a strong answer raises the linked outcome's competence estimate."""
    from abridgeai.features.interviews.orchestrator.analysis import (
        AnswerAnalysis,
        Completeness,
        Correctness,
        EvidenceType,
        OutcomeEvidence,
        Relevance,
        Specificity,
    )
    from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState

    strong = AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=Correctness.CORRECT,
        specificity=Specificity.SPECIFIC,
        confidence=0.9,
        evidence=[
            OutcomeEvidence(
                outcome_id="o-1",
                turn_id="t-1",
                evidence_type=EvidenceType.SUPPORTS,
                summary="good",
                confidence=0.9,
            )
        ],
    )
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE)
    data.outcome_coverage["o-1"] = OutcomeCoverageState(outcome_id="o-1")
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=strong,
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        per_outcome_difficulty_enabled=True,
    )
    assert data.outcome_coverage["o-1"].competence_estimate > 0.5


def test_per_outcome_competence_off_leaves_prior() -> None:
    from abridgeai.features.interviews.orchestrator.analysis import (
        AnswerAnalysis,
        Completeness,
        Correctness,
        EvidenceType,
        OutcomeEvidence,
        Relevance,
        Specificity,
    )
    from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState

    strong = AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=Correctness.CORRECT,
        specificity=Specificity.SPECIFIC,
        confidence=0.9,
        evidence=[
            OutcomeEvidence(
                outcome_id="o-1",
                turn_id="t-1",
                evidence_type=EvidenceType.SUPPORTS,
                summary="good",
                confidence=0.9,
            )
        ],
    )
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE)
    data.outcome_coverage["o-1"] = OutcomeCoverageState(outcome_id="o-1")
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=strong,
        decision=_advance_decision(),
        selected_question_id="q-2",
        target_outcome_id="o-1",
        per_outcome_difficulty_enabled=False,
    )
    assert data.outcome_coverage["o-1"].competence_estimate == 0.5


def _closing_decision(action: InterviewerActionType, reason: ReasonCode) -> InterviewerDecision:
    return InterviewerDecision(
        action=action,
        reason_code=reason,
        should_advance_question=False,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
    )


def test_rich_closing_marker_advances_reflection_to_questions() -> None:
    """Slice 13: closing sub-actions flip phase to CLOSING and advance the marker."""
    data = InterviewRuntimeStateData(phase=InterviewPhase.CORE)
    # self-reflection → marker "reflection"
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_closing_decision(
            InterviewerActionType.PROMPT_SELF_REFLECTION, ReasonCode.CLOSING_SELF_REFLECTION
        ),
        selected_question_id=None,
        target_outcome_id=None,
    )
    assert data.phase is InterviewPhase.CLOSING
    assert data.closing_step == "reflection"

    # invite questions → marker "questions"
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_closing_decision(
            InterviewerActionType.INVITE_CANDIDATE_QUESTIONS, ReasonCode.CLOSING_INVITE_QUESTIONS
        ),
        selected_question_id=None,
        target_outcome_id=None,
    )
    assert data.closing_step == "questions"


def test_rich_closing_answer_question_keeps_marker() -> None:
    data = InterviewRuntimeStateData(phase=InterviewPhase.CLOSING, closing_step="questions")
    apply_state_updates(
        data,
        intent=_answer_intent(),
        analysis=None,
        decision=_closing_decision(
            InterviewerActionType.ANSWER_CANDIDATE_QUESTION, ReasonCode.CLOSING_ANSWERED_QUESTION
        ),
        selected_question_id=None,
        target_outcome_id=None,
    )
    # Marker stays "questions" so the NEXT turn signs off.
    assert data.closing_step == "questions"
    assert data.phase is InterviewPhase.CLOSING
