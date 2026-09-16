from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.features.quizzes.services import audit, gradebook, manual_grading


class _Result:
    def __init__(self, *, rows: list[object] | None = None, scalar: object = ..., first=None):
        self.rows = rows or []
        self.scalar = scalar
        self.first_value = first

    def all(self):
        return self.rows

    def scalar_one_or_none(self):
        if self.scalar is ...:
            raise AssertionError("scalar result was not configured")
        return self.scalar

    def first(self):
        return self.first_value


@pytest.mark.asyncio
async def test_list_needs_grading_projects_join_rows():
    triples = [
        (SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())),
        (SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())),
    ]
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result(rows=triples)))

    assert await manual_grading.list_needs_grading(db, quiz_id=uuid4()) == triples


@pytest.mark.asyncio
async def test_manual_grade_rejects_negative_and_missing_answer():
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result(scalar=None)))
    with pytest.raises(AppError, match=">= 0"):
        await manual_grading.grade_answer_manually(
            db,
            quiz_id=uuid4(),
            answer_id=uuid4(),
            score=Decimal("-1"),
            feedback=None,
            grader_id=uuid4(),
        )
    with pytest.raises(NotFoundError, match="not found"):
        await manual_grading.grade_answer_manually(
            db,
            quiz_id=uuid4(),
            answer_id=uuid4(),
            score=Decimal("1"),
            feedback=None,
            grader_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_manual_grade_keeps_attempt_pending_when_other_answers_need_grading(monkeypatch):
    now = datetime(2026, 9, 16, tzinfo=UTC)
    quiz_id = uuid4()
    grader_id = uuid4()
    answer = SimpleNamespace(
        id=uuid4(),
        attempt_id=uuid4(),
        manual_score=None,
        manual_feedback=None,
        graded_by=None,
        graded_at=None,
        points_awarded=Decimal("0"),
        is_correct=False,
        needs_manual_grade=True,
    )
    attempt = SimpleNamespace(
        id=answer.attempt_id,
        student_id=uuid4(),
        score_points=Decimal("3"),
        score_percent=Decimal("60"),
        passed=False,
    )
    quiz = SimpleNamespace(passing_score_percent=Decimal("70"))
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[_Result(scalar=answer), _Result(first=(uuid4(),))]
        ),
        get=AsyncMock(side_effect=[attempt, quiz]),
        refresh=AsyncMock(),
    )
    flush = AsyncMock()
    record = AsyncMock()
    monkeypatch.setattr(manual_grading, "flush_or_conflict", flush)
    monkeypatch.setattr(manual_grading, "utcnow", lambda: now)
    monkeypatch.setattr(audit, "record_event", record)

    result = await manual_grading.grade_answer_manually(
        db,
        quiz_id=quiz_id,
        answer_id=answer.id,
        score=Decimal("0.5"),
        feedback="Partial",
        grader_id=grader_id,
    )

    assert result is answer
    assert answer.manual_score == Decimal("0.5")
    assert answer.points_awarded == Decimal("0.5")
    assert answer.is_correct is True
    assert answer.needs_manual_grade is False
    assert attempt.score_points is None
    assert attempt.score_percent is None
    assert attempt.passed is None
    record.assert_awaited_once()
    db.refresh.assert_awaited_once_with(answer)


@pytest.mark.asyncio
async def test_manual_grade_finalizes_attempt_recomputes_gradebook_and_audit(monkeypatch):
    now = datetime(2026, 9, 16, tzinfo=UTC)
    quiz_id = uuid4()
    answer = SimpleNamespace(
        id=uuid4(),
        attempt_id=uuid4(),
        manual_score=None,
        manual_feedback=None,
        graded_by=None,
        graded_at=None,
        points_awarded=Decimal("0"),
        is_correct=False,
        needs_manual_grade=True,
    )
    attempt = SimpleNamespace(
        id=answer.attempt_id,
        student_id=uuid4(),
        score_points=None,
        score_percent=None,
        passed=None,
        status="submitted",
        graded_at=None,
    )
    quiz = SimpleNamespace(passing_score_percent=Decimal("70"))
    db = SimpleNamespace(
        execute=AsyncMock(side_effect=[_Result(scalar=answer), _Result(first=None)]),
        get=AsyncMock(side_effect=[attempt, quiz]),
        refresh=AsyncMock(),
    )
    monkeypatch.setattr(manual_grading, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(manual_grading, "utcnow", lambda: now)
    recompute = AsyncMock(
        return_value=(Decimal("4"), Decimal("80"), 4, 5)
    )
    monkeypatch.setattr(manual_grading, "_recompute_attempt_score", recompute)
    gradebook_recompute = AsyncMock()
    record = AsyncMock()
    monkeypatch.setattr(gradebook, "recompute_final_grade", gradebook_recompute)
    monkeypatch.setattr(audit, "record_event", record)

    await manual_grading.grade_answer_manually(
        db,
        quiz_id=quiz_id,
        answer_id=answer.id,
        score=Decimal("1"),
        feedback=None,
        grader_id=None,
    )

    assert attempt.score_points == Decimal("4")
    assert attempt.score_percent == Decimal("80")
    assert attempt.passed is True
    assert attempt.status == "graded"
    assert attempt.graded_at == now
    gradebook_recompute.assert_awaited_once_with(db, quiz, attempt.student_id)
    record.assert_awaited_once_with(
        db,
        event_name="attempt_manually_graded",
        quiz_id=quiz_id,
        actor_user_id=None,
        subject_attempt_id=attempt.id,
        subject_user_id=attempt.student_id,
        payload={"answer_id": str(answer.id), "score": "1"},
    )


@pytest.mark.asyncio
async def test_manual_grade_still_returns_when_parent_rows_disappeared(monkeypatch):
    answer = SimpleNamespace(
        id=uuid4(),
        attempt_id=uuid4(),
        manual_score=None,
        manual_feedback=None,
        graded_by=None,
        graded_at=None,
        points_awarded=Decimal("0"),
        is_correct=False,
        needs_manual_grade=True,
    )
    db = SimpleNamespace(
        execute=AsyncMock(return_value=_Result(scalar=answer)),
        get=AsyncMock(side_effect=[None, None]),
        refresh=AsyncMock(),
    )
    monkeypatch.setattr(manual_grading, "flush_or_conflict", AsyncMock())

    assert (
        await manual_grading.grade_answer_manually(
            db,
            quiz_id=uuid4(),
            answer_id=answer.id,
            score=Decimal("0"),
            feedback=None,
            grader_id=None,
        )
        is answer
    )
    assert answer.is_correct is False
