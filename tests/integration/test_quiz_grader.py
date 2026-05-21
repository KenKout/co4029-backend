"""Unit tests for the type-aware quiz grader (T5.11 multi-type port).

Exercises the four supported question types against the canonical
expected-answer storage:

* ``multiple_choice`` / ``true_false`` — option lookup.
* ``short_answer`` — case-insensitive whitespace-normalised match.
* ``fill_blank`` — positional list compare against
  ``original_generated_payload['correct_answer']``.

The grader has its own SQLAlchemy import carve-out
(``abridgeai.features.quizzes.services.grader -> sqlalchemy``) so this
test runs against the integration test database via the shared
``session_factory`` fixture, not mocks.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.features.identity import (
    models as _identity_models,  # noqa: F401  -- side-effect: register users table for FK resolution
)
from abridgeai.features.interviews import (
    models as _interview_models,  # noqa: F401  -- side-effect: register InterviewConfig for ModuleItem mapper
)
from abridgeai.features.quizzes.models import QuizQuestion, QuizQuestionOption
from abridgeai.features.quizzes.services.grader import grade_answer

pytestmark = pytest.mark.asyncio


def _async_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


def _ensure_head() -> None:
    cfg = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    from abridgeai.core.config import get_settings

    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def grader_quiz_id(engine: AsyncEngine) -> uuid.UUID:
    """Create the org/user/course/module/quiz scaffold needed for FKs."""
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"grader-{org_id.hex[:8]}", "name": "Grader Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"grader-{user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": user_id,
                "slug": f"grader-course-{course_id.hex[:8]}",
                "title": "Grader Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, 1)"
            ),
            {"id": module_id, "course": course_id, "title": "Grader Module"},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :module, :title, 'draft')"
            ),
            {
                "id": quiz_id,
                "course": course_id,
                "module": module_id,
                "title": "Grader Quiz",
            },
        )
    return quiz_id


async def _make_question(
    session: AsyncSession,
    quiz_id: uuid.UUID,
    question_type: str,
    *,
    payload: dict | None = None,
    position: int = 1,
) -> QuizQuestion:
    question = QuizQuestion(
        quiz_id=quiz_id,
        position=position,
        question_type=question_type,
        prompt_text="probe stem",
        explanation="probe explanation",
        original_generated_payload=payload,
    )
    session.add(question)
    await session.flush()
    await session.refresh(question)
    return question


async def _make_option(
    session: AsyncSession,
    question_id: uuid.UUID,
    *,
    key: str,
    text_: str,
    is_correct: bool,
    position: int,
) -> QuizQuestionOption:
    option = QuizQuestionOption(
        question_id=question_id,
        option_key=key,
        option_text=text_,
        is_correct=is_correct,
        position=position,
    )
    session.add(option)
    await session.flush()
    await session.refresh(option)
    return option


async def test_grade_multiple_choice_correct_and_wrong(
    session_factory: async_sessionmaker[AsyncSession],
    grader_quiz_id: uuid.UUID,
) -> None:
    async with session_factory() as session:
        question = await _make_question(session, grader_quiz_id, "multiple_choice")
        correct = await _make_option(
            session, question.id, key="A", text_="alpha", is_correct=True, position=1
        )
        wrong = await _make_option(
            session, question.id, key="B", text_="beta", is_correct=False, position=2
        )
        await session.commit()

    async with session_factory() as session:
        good = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=correct.id,
            answer_text=None,
        )
        assert good.is_correct is True
        assert good.points_awarded == Decimal("1")

        bad = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=wrong.id,
            answer_text=None,
        )
        assert bad.is_correct is False
        assert bad.points_awarded == Decimal("0")


async def test_grade_true_false_uses_option_lookup(
    session_factory: async_sessionmaker[AsyncSession],
    grader_quiz_id: uuid.UUID,
) -> None:
    async with session_factory() as session:
        question = await _make_question(session, grader_quiz_id, "true_false", position=2)
        truthy = await _make_option(
            session, question.id, key="T", text_="True", is_correct=True, position=1
        )
        await _make_option(
            session, question.id, key="F", text_="False", is_correct=False, position=2
        )
        await session.commit()

    async with session_factory() as session:
        result = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=truthy.id,
            answer_text=None,
        )
        assert result.is_correct is True


async def test_grade_short_answer_case_and_hyphen_insensitive(
    session_factory: async_sessionmaker[AsyncSession],
    grader_quiz_id: uuid.UUID,
) -> None:
    async with session_factory() as session:
        question = await _make_question(
            session,
            grader_quiz_id,
            "short_answer",
            payload={"correct_answer": "Time-Variant"},
            position=3,
        )
        await session.commit()

    async with session_factory() as session:
        for submission in ("time-variant", "  Time variant ", "TIME-variant"):
            result = await grade_answer(
                session,
                question_id=question.id,
                selected_option_id=None,
                answer_text=submission,
            )
            assert result.is_correct is True, submission

        wrong = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=None,
            answer_text="non-volatile",
        )
        assert wrong.is_correct is False


async def test_grade_short_answer_missing_payload_rejects(
    session_factory: async_sessionmaker[AsyncSession],
    grader_quiz_id: uuid.UUID,
) -> None:
    async with session_factory() as session:
        question = await _make_question(
            session, grader_quiz_id, "short_answer", payload=None, position=4
        )
        await session.commit()

    async with session_factory() as session:
        result = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=None,
            answer_text="anything",
        )
        assert result.is_correct is False


async def test_grade_fill_blank_positional_match(
    session_factory: async_sessionmaker[AsyncSession],
    grader_quiz_id: uuid.UUID,
) -> None:
    blanks = ["subject-oriented", "integrated", "non-volatile"]
    async with session_factory() as session:
        question = await _make_question(
            session,
            grader_quiz_id,
            "fill_blank",
            payload={"correct_answer": blanks},
            position=5,
        )
        await session.commit()

    async with session_factory() as session:
        correct = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=None,
            answer_text=json.dumps(blanks),
        )
        assert correct.is_correct is True

        out_of_order = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=None,
            answer_text=json.dumps(["integrated", "subject-oriented", "non-volatile"]),
        )
        assert out_of_order.is_correct is False

        short = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=None,
            answer_text=json.dumps(["subject-oriented", "integrated"]),
        )
        assert short.is_correct is False


async def test_grade_fill_blank_accepts_csv_fallback(
    session_factory: async_sessionmaker[AsyncSession],
    grader_quiz_id: uuid.UUID,
) -> None:
    blanks = ["alpha", "beta"]
    async with session_factory() as session:
        question = await _make_question(
            session,
            grader_quiz_id,
            "fill_blank",
            payload={"correct_answer": blanks},
            position=6,
        )
        await session.commit()

    async with session_factory() as session:
        csv_result = await grade_answer(
            session,
            question_id=question.id,
            selected_option_id=None,
            answer_text="alpha, beta",
        )
        assert csv_result.is_correct is True


async def test_grade_unknown_question_returns_zero(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        result = await grade_answer(
            session,
            question_id=uuid.uuid4(),
            selected_option_id=None,
            answer_text="nope",
        )
    assert result.is_correct is False
    assert result.points_awarded == Decimal("0")
