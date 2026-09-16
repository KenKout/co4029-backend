from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.quizzes.schemas.attempt import ReviewVisibilityFlags
from abridgeai.features.quizzes.services import attempt_reading, feedback, review_visibility


class _Result:
    def __init__(self, rows: object = None):
        self.rows = [] if rows is None else rows

    def first(self):
        return self.rows[0] if self.rows else None

    def scalars(self):
        return self

    def all(self):
        return self.rows


def _attempt(*, status: str = "submitted", answers: list[object] | None = None):
    return SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        student_id=uuid4(),
        attempt_number=1,
        status=status,
        started_at=datetime(2026, 9, 16, tzinfo=UTC),
        submitted_at=datetime(2026, 9, 16, 0, 5, tzinfo=UTC),
        graded_at=None,
        time_taken_seconds=300,
        score_points=Decimal("8"),
        score_percent=Decimal("80"),
        passed=True,
        total_questions=10,
        correct_count=8,
        answers=[] if answers is None else answers,
        layout=None,
    )


@pytest.mark.asyncio
async def test_project_in_progress_attempt_only_adds_pending_flag():
    attempt = _attempt(status="in_progress")
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result([(uuid4(),)])), get=AsyncMock())

    result = await attempt_reading.project_attempt_summary(db, attempt)

    assert result.grading_pending is True
    assert result.score_percent == Decimal("80")
    db.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_submitted_attempt_masks_score_when_hidden(monkeypatch):
    attempt = _attempt()
    quiz = SimpleNamespace(id=attempt.quiz_id)
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result()), get=AsyncMock(return_value=quiz))
    monkeypatch.setattr(
        review_visibility,
        "resolve_review_visibility",
        lambda *_args: ReviewVisibilityFlags(show_score=False),
    )

    result = await attempt_reading.project_attempt_summary(db, attempt)

    assert result.score_points is None
    assert result.score_percent is None
    assert result.passed is None
    assert result.correct_count is None


@pytest.mark.asyncio
async def test_project_submitted_attempt_requires_quiz():
    attempt = _attempt()
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result()), get=AsyncMock(return_value=None))
    with pytest.raises(NotFoundError, match=str(attempt.quiz_id)):
        await attempt_reading.project_attempt_summary(db, attempt)


@pytest.mark.asyncio
async def test_attempt_history_projects_every_owned_attempt(monkeypatch):
    attempts = [_attempt(), _attempt(status="in_progress")]
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result(attempts)))
    projected = [SimpleNamespace(id=row.id) for row in attempts]
    monkeypatch.setattr(
        attempt_reading, "project_attempt_summary", AsyncMock(side_effect=projected)
    )
    actor = SimpleNamespace(user_id=uuid4())

    result = await attempt_reading.get_attempt_history(db, uuid4(), actor)

    assert result == projected
    assert attempt_reading.project_attempt_summary.await_count == 2


@pytest.mark.asyncio
async def test_attempt_review_returns_none_for_unowned_attempt(monkeypatch):
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "get_attempt_for_review",
        AsyncMock(return_value=None),
    )
    assert (
        await attempt_reading.get_attempt_review(
            object(), attempt_id=uuid4(), actor=SimpleNamespace(user_id=uuid4())
        )
        is None
    )


@pytest.mark.asyncio
async def test_attempt_review_projects_structured_answers_and_feedback(monkeypatch):
    question_id = uuid4()
    answer = SimpleNamespace(
        question_id=question_id,
        selected_option_id=None,
        answer_text="Paris",
        is_correct=True,
        points_awarded=Decimal("1"),
        hint_used=False,
        t_actual_ms=500,
    )
    attempt = _attempt(answers=[answer])
    question = SimpleNamespace(
        id=question_id,
        position=1,
        question_type="fill_blank",
        prompt_text="Capital: ____",
        explanation="France's capital",
        hint_text="City",
        match_pairs=None,
        ordering_sequence=None,
        original_generated_payload={"correct_answer": ["Paris"]},
    )
    option = SimpleNamespace(
        id=uuid4(), option_key="A", option_text="Paris", is_correct=True, position=1
    )
    quiz = SimpleNamespace(id=attempt.quiz_id)
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "get_attempt_for_review",
        AsyncMock(return_value=attempt),
    )
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "list_quiz_questions_with_options",
        AsyncMock(return_value=[(question, [option])]),
    )
    monkeypatch.setattr(
        review_visibility,
        "resolve_review_visibility",
        lambda *_args: ReviewVisibilityFlags(),
    )
    monkeypatch.setattr(
        attempt_reading,
        "project_attempt_summary",
        AsyncMock(return_value=attempt_reading.QuizAttemptRead.model_validate(attempt)),
    )
    monkeypatch.setattr(
        feedback,
        "select_overall_feedback",
        AsyncMock(return_value=SimpleNamespace(feedback_text="Great", feedback_format="plain")),
    )
    db = SimpleNamespace(get=AsyncMock(return_value=quiz))

    result = await attempt_reading.get_attempt_review(
        db, attempt_id=attempt.id, actor=SimpleNamespace(user_id=attempt.student_id)
    )

    assert result is not None
    assert result.questions[0].fill_blank_correct == ["Paris"]
    assert result.questions[0].options[0].is_correct is True
    assert result.overall_feedback_text == "Great"


@pytest.mark.asyncio
async def test_attempt_review_masks_pending_grade_details(monkeypatch):
    attempt = _attempt()
    question = SimpleNamespace(
        id=uuid4(),
        position=1,
        question_type="short_answer",
        prompt_text="Explain",
        explanation="Hidden",
        hint_text=None,
        match_pairs=None,
        ordering_sequence=None,
        original_generated_payload={"correct_answer": "Expected"},
    )
    option = SimpleNamespace(
        id=uuid4(), option_key="A", option_text="Option", is_correct=True, position=1
    )
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "get_attempt_for_review",
        AsyncMock(return_value=attempt),
    )
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "list_quiz_questions_with_options",
        AsyncMock(return_value=[(question, [option])]),
    )
    monkeypatch.setattr(
        review_visibility,
        "resolve_review_visibility",
        lambda *_args: ReviewVisibilityFlags(),
    )
    pending = attempt_reading.QuizAttemptRead.model_validate(attempt).model_copy(
        update={"grading_pending": True}
    )
    monkeypatch.setattr(
        attempt_reading, "project_attempt_summary", AsyncMock(return_value=pending)
    )
    select_feedback = AsyncMock()
    monkeypatch.setattr(feedback, "select_overall_feedback", select_feedback)
    db = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(id=attempt.quiz_id)))

    result = await attempt_reading.get_attempt_review(
        db, attempt_id=attempt.id, actor=SimpleNamespace(user_id=attempt.student_id)
    )

    assert result is not None
    assert result.attempt.score_percent is None
    assert result.questions[0].is_correct is False
    assert result.questions[0].points_awarded == 0
    assert result.questions[0].options[0].is_correct is True
    select_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_attempt_progress_returns_none_for_non_active_attempt(monkeypatch):
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "get_in_progress_attempt",
        AsyncMock(return_value=None),
    )
    assert (
        await attempt_reading.get_attempt_progress(
            object(), attempt_id=uuid4(), actor=SimpleNamespace(user_id=uuid4())
        )
        is None
    )


@pytest.mark.asyncio
async def test_attempt_progress_rebuilds_take_payload_without_answer_leaks(monkeypatch):
    from abridgeai.features.quizzes.services import taking

    question_id = uuid4()
    answer = SimpleNamespace(
        question_id=question_id,
        selected_option_id=None,
        answer_text="student input",
        hint_used=True,
        t_actual_ms=250,
        is_correct=True,
        points_awarded=Decimal("1"),
    )
    attempt = _attempt(status="in_progress", answers=[answer])
    quiz = SimpleNamespace(
        id=attempt.quiz_id,
        title="Published quiz",
        slug="published-quiz",
        description=None,
        status="published",
        passing_score_percent=Decimal("70"),
        time_limit_seconds=None,
        max_attempts=2,
        allow_retakes=True,
        cooldown_hours=None,
        show_hints=True,
        require_camera=False,
        available_from=None,
        available_until=None,
        due_at=None,
        review_options={},
    )
    question = SimpleNamespace(
        id=question_id,
        quiz_id=attempt.quiz_id,
        position=4,
        question_type="short_answer",
        prompt_text="Explain",
        hint_text="Hint",
        prompt_format="plain",
        hint_format="plain",
        single_answer=True,
        options=[],
        learning_outcome_id=None,
        outcome_position=None,
        outcome_code=None,
        match_pairs=None,
        ordering_sequence=None,
        original_generated_payload={"correct_answer": "secret"},
    )
    monkeypatch.setattr(
        attempt_reading.published_queries,
        "get_in_progress_attempt",
        AsyncMock(return_value=attempt),
    )
    monkeypatch.setattr(taking, "_require_quiz", AsyncMock(return_value=quiz))
    monkeypatch.setattr(
        taking, "_load_quiz_questions_for_taking", AsyncMock(return_value=[question])
    )

    result = await attempt_reading.get_attempt_progress(
        object(), attempt_id=attempt.id, actor=SimpleNamespace(user_id=attempt.student_id)
    )

    assert result is not None
    assert result.take.quiz.id == attempt.quiz_id
    assert result.take.questions[0].position == 1
    assert result.answers[0].answer_text == "student input"
    assert "is_correct" not in result.answers[0].model_dump()
    assert "points_awarded" not in result.answers[0].model_dump()
