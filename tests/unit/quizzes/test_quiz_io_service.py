from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import AppError
from abridgeai.features.quizzes.services import quiz_io
from abridgeai.features.quizzes.services.formats._types import (
    ParsedOption,
    ParsedQuestion,
    ParseResult,
)


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarRows:
        return self

    def all(self) -> list[object]:
        return self._rows


def _question(kind: str = "multiple_choice", *, correct_answer: object = None):
    return ParsedQuestion(
        question_type=kind,
        prompt_text="Prompt",
        options=[
            ParsedOption(text="Yes", is_correct=True),
            ParsedOption(text="No", is_correct=False),
        ],
        correct_answer=correct_answer,
        explanation="Because",
    )


def test_parse_rejects_unknown_format():
    with pytest.raises(AppError, match="Unsupported import format"):
        quiz_io._parse("content", "csv")


def test_create_payload_assigns_stable_option_keys_and_open_answer():
    parsed = ParsedQuestion(
        question_type="short_answer",
        prompt_text="Name it",
        options=[ParsedOption(text=str(i), is_correct=i == 0) for i in range(12)],
        correct_answer="answer",
        explanation=None,
    )

    payload = quiz_io._to_create_payload(parsed)

    assert [o.option_key for o in payload.options[:10]] == list("ABCDEFGHIJ")
    assert [o.option_key for o in payload.options[10:]] == ["11", "12"]
    assert payload.correct_answer == "answer"
    assert payload.review_status == "pending"


@pytest.mark.asyncio
async def test_import_questions_collects_warnings_sets_tf_keys_and_open_answer(monkeypatch):
    tf = _question("true_false")
    short = _question("short_answer", correct_answer="blue")
    bad = _question("code")
    monkeypatch.setattr(
        quiz_io,
        "_parse",
        lambda content, fmt: ParseResult(
            questions=[tf, short, bad], warnings=["parser warning"]
        ),
    )
    created_tf = SimpleNamespace(original_generated_payload=None)
    created_short = SimpleNamespace(original_generated_payload=None)
    create = AsyncMock(
        side_effect=[created_tf, created_short, AppError("unsupported question")]
    )
    monkeypatch.setattr(quiz_io._authoring, "create_question", create)

    result = await quiz_io.import_questions_from_file(
        object(), quiz_id=uuid4(), content="ignored", fmt="gift", actor=object()
    )

    assert result == {
        "imported": 2,
        "skipped": 1,
        "warnings": ["parser warning", "Q3: unsupported question"],
    }
    tf_payload = create.await_args_list[0].args[2]
    assert [o.option_key for o in tf_payload.options] == ["T", "F"]
    assert created_short.original_generated_payload == {"correct_answer": "blue"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("fmt", "marker"), [("gift", "Prompt"), ("xml", "<quiz")])
async def test_export_serializes_questions_and_options(fmt: str, marker: str):
    question_id = uuid4()
    correct_id = uuid4()
    question = SimpleNamespace(
        id=question_id,
        question_type="multiple_choice",
        prompt_text="Prompt",
        original_generated_payload=None,
        explanation="Explanation",
    )
    options = [
        SimpleNamespace(
            id=correct_id,
            question_id=question_id,
            option_text="Correct",
            is_correct=True,
        ),
        SimpleNamespace(
            id=uuid4(),
            question_id=question_id,
            option_text="Wrong",
            is_correct=False,
        ),
    ]
    db = SimpleNamespace(
        execute=AsyncMock(side_effect=[_ScalarRows([question]), _ScalarRows(options)])
    )

    rendered = await quiz_io.export_quiz_questions(db, quiz_id=uuid4(), fmt=fmt)

    assert marker in rendered
    assert "Correct" in rendered


@pytest.mark.asyncio
async def test_export_empty_quiz_and_unknown_format():
    db = SimpleNamespace(execute=AsyncMock(return_value=_ScalarRows([])))
    assert await quiz_io.export_quiz_questions(db, quiz_id=uuid4(), fmt="gift") == "\n"

    with pytest.raises(AppError, match="Unsupported export format"):
        await quiz_io.export_quiz_questions(db, quiz_id=uuid4(), fmt="csv")
