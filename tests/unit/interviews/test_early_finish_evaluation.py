"""Pure tests for early-finished interview grading semantics."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.services.evaluation import (
    _build_question_evaluation_context,
    _fail_unanswered_outcomes,
)


def test_context_includes_approved_unanswered_questions() -> None:
    answered_id, unanswered_id, pending_id = uuid4(), uuid4(), uuid4()
    session_question_id = uuid4()
    questions = [
        SimpleNamespace(
            id=answered_id,
            prompt_text="Answered prompt",
            review_status="approved",
            linked_outcome_id=None,
        ),
        SimpleNamespace(
            id=unanswered_id,
            prompt_text="Unanswered prompt",
            review_status="approved",
            linked_outcome_id=None,
        ),
        SimpleNamespace(
            id=pending_id,
            prompt_text="Draft prompt",
            review_status="pending",
            linked_outcome_id=None,
        ),
    ]
    asked = [
        SimpleNamespace(
            id=session_question_id,
            interview_question_id=answered_id,
        )
    ]
    answers = [
        SimpleNamespace(
            session_question_id=session_question_id,
            content_text="Submitted answer",
            metadata_json={},
        )
    ]

    gradeable, prompts, expected_ids, answered_ids = _build_question_evaluation_context(
        questions, asked, answers
    )

    assert [question.id for question in gradeable] == [answered_id, unanswered_id]
    assert prompts == {session_question_id: "Answered prompt"}
    assert expected_ids == [session_question_id, unanswered_id]
    assert answered_ids == {answered_id}


def test_outcome_with_only_unanswered_questions_is_forced_not_met() -> None:
    answered_outcome_id, unanswered_outcome_id = uuid4(), uuid4()
    answered_question_id, unanswered_question_id = uuid4(), uuid4()
    verdicts = build_outcome_verdicts(
        [
            OutcomeVerdict(answered_outcome_id, True, "Judge credited it."),
            OutcomeVerdict(unanswered_outcome_id, True, "Judge inferred it."),
        ]
    )
    questions = [
        SimpleNamespace(
            id=answered_question_id,
            linked_outcome_id=answered_outcome_id,
        ),
        SimpleNamespace(
            id=unanswered_question_id,
            linked_outcome_id=unanswered_outcome_id,
        ),
    ]

    adjusted = _fail_unanswered_outcomes(
        verdicts,
        questions=questions,
        answered_question_ids={answered_question_id},
    )

    by_id = {verdict.outcome_id: verdict for verdict in adjusted.verdicts}
    assert by_id[answered_outcome_id].met is True
    assert by_id[unanswered_outcome_id].met is False
    assert by_id[unanswered_outcome_id].evidence is None
