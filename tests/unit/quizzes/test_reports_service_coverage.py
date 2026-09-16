from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.quizzes.routers import authoring as authoring_router
from abridgeai.features.quizzes.services import reports


class _Result:
    def __init__(self, rows: object, *, scalar: object = ...):
        self.rows = rows
        self.scalar = scalar

    def scalar_one_or_none(self):
        if self.scalar is ...:
            raise AssertionError("scalar result was not configured")
        return self.scalar

    def scalars(self) -> _Result:
        return self

    def all(self):
        return self.rows


def test_answer_text_helpers_cover_supported_fallbacks():
    correct = SimpleNamespace(option_text="A", is_correct=True)
    wrong = SimpleNamespace(option_text="B", is_correct=False)
    mcq = SimpleNamespace(question_type="multiple_choice", original_generated_payload=None)
    short = SimpleNamespace(
        question_type="short_answer",
        original_generated_payload={"correct_answer": "blue"},
    )
    blanks = SimpleNamespace(
        question_type="fill_blank",
        original_generated_payload={"correct_answer": ["one", "two"]},
    )
    manual = SimpleNamespace(question_type="code", original_generated_payload=None)

    assert reports._correct_answer_text(mcq, [correct, wrong]) == "A"
    assert reports._correct_answer_text(mcq, [wrong]) == "(none)"
    assert reports._correct_answer_text(short, []) == "blue"
    assert reports._correct_answer_text(blanks, []) == "one | two"
    assert reports._correct_answer_text(manual, []) == "(manual)"

    option_id = uuid4()
    option = SimpleNamespace(option_text="Selected")
    assert reports._student_answer_text(None, {}) == "(no answer)"
    assert (
        reports._student_answer_text(
            SimpleNamespace(selected_option_id=option_id, answer_text=None),
            {option_id: option},
        )
        == "Selected"
    )
    assert (
        reports._student_answer_text(
            SimpleNamespace(selected_option_id=uuid4(), answer_text=None), {}
        )
        == "(unknown option)"
    )
    assert (
        reports._student_answer_text(
            SimpleNamespace(selected_option_id=None, answer_text="free text"), {}
        )
        == "free text"
    )


@pytest.mark.asyncio
async def test_require_quiz_raises_for_missing_quiz():
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result([], scalar=None)))
    with pytest.raises(NotFoundError, match="not found"):
        await reports._require_quiz(db, uuid4())


@pytest.mark.asyncio
async def test_build_responses_report_projects_answered_and_missing_rows(monkeypatch):
    quiz_id = uuid4()
    student_id = uuid4()
    question_a = SimpleNamespace(
        id=uuid4(),
        position=1,
        prompt_text="Choose",
        question_type="multiple_choice",
        original_generated_payload=None,
        explanation=None,
    )
    question_b = SimpleNamespace(
        id=uuid4(),
        position=2,
        prompt_text="Explain",
        question_type="short_answer",
        original_generated_payload={"correct_answer": "expected"},
        explanation=None,
    )
    option = SimpleNamespace(
        id=uuid4(),
        question_id=question_a.id,
        option_text="Correct",
        is_correct=True,
    )
    attempt = SimpleNamespace(
        id=uuid4(), student_id=student_id, attempt_number=1
    )
    answer = SimpleNamespace(
        attempt_id=attempt.id,
        question_id=question_a.id,
        selected_option_id=option.id,
        answer_text=None,
        is_correct=True,
        points_awarded=Decimal("1"),
    )
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result([], scalar=SimpleNamespace(id=quiz_id)),
                _Result([question_a, question_b]),
                _Result([option]),
                _Result([attempt]),
                _Result([answer]),
            ]
        )
    )
    monkeypatch.setattr(
        authoring_router,
        "_resolve_student_names",
        AsyncMock(return_value={student_id: "Student One"}),
    )

    result = await reports.build_responses_report(db, quiz_id)

    assert len(result.rows) == 2
    assert result.rows[0].student_name == "Student One"
    assert result.rows[0].student_answer == "Correct"
    assert result.rows[0].correct_answer == "Correct"
    assert result.rows[0].is_correct is True
    assert result.rows[1].student_answer == "(no answer)"
    assert result.rows[1].correct_answer == "expected"
    assert result.rows[1].points_awarded == 0


@pytest.mark.asyncio
async def test_build_responses_report_skips_option_and_answer_queries_when_empty(monkeypatch):
    quiz_id = uuid4()
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result([], scalar=SimpleNamespace(id=quiz_id)),
                _Result([]),
                _Result([]),
            ]
        )
    )
    monkeypatch.setattr(authoring_router, "_resolve_student_names", AsyncMock(return_value={}))

    result = await reports.build_responses_report(db, quiz_id)

    assert result.rows == []
    assert db.execute.await_count == 3


@pytest.mark.asyncio
async def test_build_statistics_report_computes_facility_and_empty_question(monkeypatch):
    quiz_id = uuid4()
    q_answered = SimpleNamespace(id=uuid4(), position=1, prompt_text="Q1")
    q_empty = SimpleNamespace(id=uuid4(), position=2, prompt_text="Q2")
    attempt_a = SimpleNamespace(id=uuid4(), score_points=Decimal("8"))
    attempt_b = SimpleNamespace(id=uuid4(), score_points=Decimal("2"))
    answers = [
        SimpleNamespace(
            question_id=q_answered.id, attempt_id=attempt_a.id, is_correct=True
        ),
        SimpleNamespace(
            question_id=q_answered.id, attempt_id=attempt_b.id, is_correct=False
        ),
    ]
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result([], scalar=SimpleNamespace(id=quiz_id)),
                _Result([q_answered, q_empty]),
                _Result([attempt_a, attempt_b]),
                _Result(answers),
            ]
        )
    )
    monkeypatch.setattr(
        reports,
        "point_biserial",
        lambda flags, totals: (0.75, "ok") if flags else (None, "insufficient"),
    )

    result = await reports.build_statistics_report(db, quiz_id)

    assert result.attempts_analyzed == 2
    assert result.rows[0].answered_count == 2
    assert result.rows[0].correct_count == 1
    assert result.rows[0].facility_index == 0.5
    assert result.rows[0].discrimination_index == 0.75
    assert result.rows[1].answered_count == 0
    assert result.rows[1].facility_index is None
    assert result.rows[1].discrimination_note == "insufficient"


@pytest.mark.asyncio
async def test_build_statistics_report_skips_answers_query_without_attempts():
    quiz_id = uuid4()
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result([], scalar=SimpleNamespace(id=quiz_id)),
                _Result([]),
                _Result([]),
            ]
        )
    )

    result = await reports.build_statistics_report(db, quiz_id)

    assert result.attempts_analyzed == 0
    assert result.rows == []
    assert db.execute.await_count == 3
