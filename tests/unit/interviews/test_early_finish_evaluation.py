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


def test_context_denominator_is_the_asked_set() -> None:
    """Asked-but-unanswered stays zero-filled; never-asked is excluded.

    The agent (server-owned advance + coverage close) decides what gets
    asked, so never-asked questions are the agent's choice, not a skipped
    answer — zero-filling them wrote "did not submit" reports for candidates
    who answered everything they were asked.
    """
    answered_id, unanswered_asked_id, never_asked_id, pending_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    answered_session_id, unanswered_session_id = uuid4(), uuid4()
    questions = [
        SimpleNamespace(
            id=answered_id,
            prompt_text="Answered prompt",
            review_status="approved",
            linked_outcome_id=None,
        ),
        SimpleNamespace(
            id=unanswered_asked_id,
            prompt_text="Asked but unanswered prompt",
            review_status="approved",
            linked_outcome_id=None,
        ),
        SimpleNamespace(
            id=never_asked_id,
            prompt_text="Never asked prompt",
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
            id=answered_session_id,
            interview_question_id=answered_id,
        ),
        SimpleNamespace(
            id=unanswered_session_id,
            interview_question_id=unanswered_asked_id,
        ),
    ]
    answers = [
        SimpleNamespace(
            session_question_id=answered_session_id,
            content_text="Submitted answer",
            metadata_json={},
        )
    ]

    gradeable, prompts, expected_ids, answered_ids = _build_question_evaluation_context(
        questions, asked, answers
    )

    assert [question.id for question in gradeable] == [answered_id, unanswered_asked_id]
    assert prompts == {
        answered_session_id: "Answered prompt",
        unanswered_session_id: "Asked but unanswered prompt",
    }
    assert expected_ids == [answered_session_id, unanswered_session_id]
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


def test_outcome_with_only_never_asked_questions_keeps_judge_verdict() -> None:
    """No linked question was asked → nothing was stonewalled → judge decides.

    The caller passes the ASKED question set; an outcome whose questions were
    never asked has no linked ids in that set and must keep the judge's
    transcript-based verdict instead of being force-failed as "no answer
    submitted".
    """
    asked_outcome_id, never_asked_outcome_id = uuid4(), uuid4()
    asked_question_id = uuid4()
    verdicts = build_outcome_verdicts(
        [
            OutcomeVerdict(never_asked_outcome_id, True, "Judge inferred it from coverage."),
        ]
    )
    asked_questions = [
        # A question for a DIFFERENT outcome — the never-asked outcome has no
        # linked question in the asked set.
        SimpleNamespace(
            id=asked_question_id,
            linked_outcome_id=asked_outcome_id,
        ),
    ]

    adjusted = _fail_unanswered_outcomes(
        verdicts,
        questions=asked_questions,
        answered_question_ids={asked_question_id},
    )

    by_id = {verdict.outcome_id: verdict for verdict in adjusted.verdicts}
    assert by_id[never_asked_outcome_id].met is True
    assert by_id[never_asked_outcome_id].reasoning == "Judge inferred it from coverage."
