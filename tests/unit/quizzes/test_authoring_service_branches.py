from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.features.quizzes.services import authoring


class _Result:
    def __init__(self, rows: object = None, *, scalar: object = ...):
        self.rows = [] if rows is None else rows
        self.scalar = scalar

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows

    def scalar_one(self):
        if self.scalar is ...:
            raise AssertionError("scalar result was not configured")
        return self.scalar


def _option(text: str, correct: bool, key: str = "A") -> SimpleNamespace:
    return SimpleNamespace(option_text=text, is_correct=correct, option_key=key)


def test_coerce_patch_value_handles_datetime_integrity_and_passthrough():
    instant = datetime(2026, 9, 16, tzinfo=UTC)
    assert authoring._coerce_patch_value("title", "Quiz") == "Quiz"
    assert authoring._coerce_patch_value("available_from", instant) is instant
    assert authoring._coerce_patch_value("available_until", " ") is None
    assert authoring._coerce_patch_value("due_at", "2026-09-16T10:00:00Z") == instant.replace(
        hour=10
    )
    assert authoring._coerce_patch_value("integrity_weight_tab_switch", "5") == 5

    with pytest.raises(AppError, match="ISO-8601"):
        authoring._coerce_patch_value("due_at", "tomorrow")
    with pytest.raises(AppError, match="ISO-8601"):
        authoring._coerce_patch_value("due_at", 123)
    with pytest.raises(AppError, match="integer"):
        authoring._coerce_patch_value("integrity_score_threshold", "bad")
    with pytest.raises(AppError, match="between 1 and 5"):
        authoring._coerce_patch_value("integrity_weight_focus_lost", 6)


def test_apply_patch_and_plain_json_support_schema_like_values():
    model = SimpleNamespace(title="Old", due_at=None)
    payload = SimpleNamespace(
        model_dump=lambda **_kwargs: {"title": "New", "due_at": "2026-09-16T00:00:00Z"}
    )
    authoring._apply_patch(model, payload)
    assert model.title == "New"
    assert model.due_at == datetime(2026, 9, 16, tzinfo=UTC)

    wrapped = SimpleNamespace(model_dump=lambda: {"values": [Decimal("1.5"), {"x": 1}]})
    assert authoring._as_plain_json(wrapped) == {"values": [Decimal("1.5"), {"x": 1}]}


def test_published_edit_guards_distinguish_content_and_safe_settings():
    draft = SimpleNamespace(status="draft")
    published = SimpleNamespace(status="published")
    authoring._assert_quiz_editable(draft)
    authoring._assert_quiz_settings_editable(draft, {"passing_score_percent"})
    authoring._assert_quiz_settings_editable(published, {"title", "due_at"})

    with pytest.raises(ConflictError, match="questions cannot be edited"):
        authoring._assert_quiz_editable(published)
    with pytest.raises(ConflictError, match="passing_score_percent"):
        authoring._assert_quiz_settings_editable(
            published, {"title", "passing_score_percent"}
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" MCQ ", "multiple_choice"),
        ("true/false", "true_false"),
        ("TF", "true_false"),
        ("fill_in_the_blank", "fill_blank"),
        ("ordering", "ordering"),
        (None, "multiple_choice"),
    ],
)
def test_normalize_question_type(raw: object, expected: str):
    assert authoring._normalize_question_type(raw) == expected


def test_validate_multiple_choice_and_true_false_options():
    authoring._validate_question_options(
        "multiple_choice", [_option("A", True), _option("B", False, "B")]
    )
    authoring._validate_question_options(
        "multiple_choice",
        [_option("A", True), _option("B", True, "B")],
        single_answer=False,
    )
    authoring._validate_question_options(
        "true_false", [_option("True", True, "T"), _option("False", False, "F")]
    )

    with pytest.raises(AppError, match="between 2 and 10"):
        authoring._validate_question_options("multiple_choice", [_option("A", True)])
    with pytest.raises(AppError, match="option text"):
        authoring._validate_question_options(
            "multiple_choice", [_option(" ", True), _option("B", False, "B")]
        )
    with pytest.raises(AppError, match="exactly one"):
        authoring._validate_question_options(
            "multiple_choice", [_option("A", True), _option("B", True, "B")]
        )
    with pytest.raises(AppError, match="at least one"):
        authoring._validate_question_options(
            "multiple_choice",
            [_option("A", False), _option("B", False, "B")],
            single_answer=False,
        )
    with pytest.raises(AppError, match="exactly two"):
        authoring._validate_question_options("true_false", [_option("True", True, "T")])
    with pytest.raises(AppError, match="keys must be T, F"):
        authoring._validate_question_options(
            "true_false", [_option("Yes", True, "Y"), _option("No", False, "N")]
        )


def test_validate_fill_blank_and_optionless_types():
    authoring._validate_question_options(
        "fill_blank", [_option("Paris", True), _option("London", False, "B")]
    )
    for options, message in [
        ([], "non-empty"),
        ([_option(" ", True), _option("x", False)], "option text"),
        ([_option("x", False), _option("y", False)], "correct answer"),
        ([_option("x", True), _option("y", True)], "distractor"),
    ]:
        with pytest.raises(AppError, match=message):
            authoring._validate_question_options("fill_blank", options)
    authoring._validate_question_options("short_answer", [])
    with pytest.raises(AppError, match="do not support options"):
        authoring._validate_question_options("short_answer", [_option("x", True)])


@pytest.mark.asyncio
async def test_require_helpers_and_course_resolution(monkeypatch):
    quiz_id = uuid4()
    question_id = uuid4()
    monkeypatch.setattr(
        authoring.authoring_queries, "get_quiz_for_authoring", AsyncMock(return_value=None)
    )
    with pytest.raises(NotFoundError, match=str(quiz_id)):
        await authoring._require_quiz(object(), quiz_id)

    db = SimpleNamespace(get=AsyncMock(return_value=None))
    with pytest.raises(NotFoundError, match=str(question_id)):
        await authoring._require_question(db, question_id)

    module_id = uuid4()
    monkeypatch.setattr(authoring.courses_api, "get_module_by_id", AsyncMock(return_value=None))
    with pytest.raises(NotFoundError, match=str(module_id)):
        await authoring._resolve_module_course(object(), module_id)

    course_id = uuid4()
    authoring.courses_api.get_module_by_id.return_value = SimpleNamespace(course_id=course_id)
    assert await authoring._resolve_module_course(object(), module_id) == course_id


@pytest.mark.asyncio
async def test_small_database_projections_cover_positions_and_in_flight_runs():
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result([("first",), ("second",)]),
                _Result(scalar=4),
                _Result(scalar=3),
                _Result([(uuid4(),)]),
                _Result([]),
            ]
        )
    )
    assert await authoring._taken_quiz_slugs(db, uuid4()) == {"first", "second"}
    assert await authoring._next_question_position(db, uuid4()) == 4
    assert await authoring._next_revision_no(db, uuid4()) == 3
    assert await authoring._quiz_has_in_flight_run(db, uuid4()) is True
    assert await authoring._quiz_has_in_flight_run(db, uuid4()) is False
