"""Unit tests for ``features.quizzes.api.public`` (T21).

Verifies the cross-feature read surface + the
:func:`create_generation_run` factory contract. Uses real PostgreSQL
because the ORM relies on PG-specific types (``CITEXT``, ``JSONB``,
``UUID``) and the soft-delete loader-criteria listener.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* FK targets
from abridgeai.ai.models import GenerationRun
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.features.quizzes.api.public import (
    create_generation_run,
    get_attempt_score,
    get_generation_run,
    get_question_with_quiz_context,
    get_quiz_question_id_set_by_lesson,
    get_t_exp_for_question,
)
from abridgeai.features.quizzes.models import QuizQuestion

for _stub_name in ("interview_configs", "learning_materials", "learning_material_versions"):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[3] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[3] / "migrations"),
    )
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_a_id = uuid.uuid4()
    question_b_id = uuid.uuid4()
    question_soft_deleted_id = uuid.uuid4()
    attempt_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qa-{suffix}", "name": "API Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qa-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'API Course', 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'draft')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:id, :module, :slug, 'Lesson', 'draft')"
            ),
            {"id": lesson_id, "module": module_id, "slug": f"lesson-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) "
                "VALUES (:id, :course, :module, 'Quiz', 'draft', 70.00)"
            ),
            {
                "id": quiz_id,
                "course": course_id,
                "module": module_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_source_lessons (quiz_id, lesson_id) VALUES (:quiz_id, :lesson_id)"
            ),
            {"quiz_id": quiz_id, "lesson_id": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs) "
                "VALUES (:id, :quiz, 1, 'multiple_choice', 'Live question A', "
                "60000, '[]'::jsonb)"
            ),
            {"id": question_a_id, "quiz": quiz_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs) "
                "VALUES (:id, :quiz, 2, 'multiple_choice', 'Live question B', "
                "45000, '[]'::jsonb)"
            ),
            {"id": question_b_id, "quiz": quiz_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs, deleted_at) "
                "VALUES (:id, :quiz, 3, 'multiple_choice', 'Tombstoned', "
                "30000, '[]'::jsonb, NOW())"
            ),
            {"id": question_soft_deleted_id, "quiz": quiz_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts "
                "(id, quiz_id, student_id, attempt_number, status, "
                "score_percent, passed) "
                "VALUES (:id, :quiz, :student, 1, 'graded', 88.50, TRUE)"
            ),
            {"id": attempt_id, "quiz": quiz_id, "student": owner_id},
        )

    yield {
        "owner_id": owner_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "quiz_id": quiz_id,
        "question_a_id": question_a_id,
        "question_b_id": question_b_id,
        "question_soft_deleted_id": question_soft_deleted_id,
        "attempt_id": attempt_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m OR course_id = :c"),
            {"m": module_id, "c": course_id},
        )
        await conn.execute(text("DELETE FROM quiz_attempts WHERE quiz_id = :q"), {"q": quiz_id})
        await conn.execute(text("DELETE FROM quiz_questions WHERE quiz_id = :q"), {"q": quiz_id})
        await conn.execute(
            text("DELETE FROM quiz_source_lessons WHERE quiz_id = :q"), {"q": quiz_id}
        )
        await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz_id})
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


@pytest.mark.asyncio
async def test_factory_quiz_and_interview(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """``create_generation_run`` builds rows for both kinds via the factory."""
    async with session_factory() as session, session.begin():
        quiz_run = await create_generation_run(
            session,
            kind="quiz",
            source_scope_kind="module",
            course_id=scenario["course_id"],
            module_id=scenario["module_id"],
            requested_by=scenario["owner_id"],
            config_json={"interview_config_id": str(uuid.uuid4()), "kind_label": "quiz"},
        )
        interview_run = await create_generation_run(
            session,
            kind="interview",
            source_scope_kind="module",
            course_id=scenario["course_id"],
            module_id=scenario["module_id"],
            requested_by=scenario["owner_id"],
            config_json={"interview_config_id": str(uuid.uuid4())},
        )

    assert quiz_run.generation_type == "quiz"
    assert interview_run.generation_type == "interview"
    assert quiz_run.id != interview_run.id
    assert quiz_run.status == "pending"
    assert interview_run.status == "pending"
    assert quiz_run.config_json["kind_label"] == "quiz"

    async with session_factory() as verify_session:
        round_trip = await get_generation_run(verify_session, quiz_run.id)
        assert round_trip is not None
        assert round_trip.id == quiz_run.id
        assert round_trip.generation_type == "quiz"
        assert round_trip.module_id == scenario["module_id"]

        rows = (
            await verify_session.execute(
                text(
                    "SELECT id, generation_type FROM generation_runs "
                    "WHERE id IN (:a, :b) ORDER BY generation_type"
                ),
                {"a": quiz_run.id, "b": interview_run.id},
            )
        ).all()
    assert len(rows) == 2
    assert {r.generation_type for r in rows} == {"interview", "quiz"}


@pytest.mark.asyncio
async def test_get_t_exp_for_question(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Fixture question's ``expected_response_time_ms`` round-trips."""
    async with session_factory() as session:
        t_exp = await get_t_exp_for_question(session, scenario["question_a_id"])
    assert t_exp == 60000

    async with session_factory() as session:
        missing = await get_t_exp_for_question(session, uuid.uuid4())
    assert missing is None


@pytest.mark.asyncio
async def test_get_question_with_quiz_context(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Found, not-found, and soft-deleted questions all return correctly."""
    async with session_factory() as session:
        found = await get_question_with_quiz_context(session, scenario["question_a_id"])
    assert found is not None
    assert found.question_id == scenario["question_a_id"]
    assert found.quiz_id == scenario["quiz_id"]
    assert found.course_id == scenario["course_id"]
    assert found.module_id == scenario["module_id"]
    assert found.prompt_text == "Live question A"
    assert found.source_refs == []

    async with session_factory() as session:
        absent = await get_question_with_quiz_context(session, uuid.uuid4())
    assert absent is None

    async with session_factory() as session:
        tombstoned = await get_question_with_quiz_context(
            session, scenario["question_soft_deleted_id"]
        )
    assert tombstoned is None


@pytest.mark.asyncio
async def test_get_attempt_score(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """``get_attempt_score`` returns the minimal DTO; missing returns ``None``."""
    async with session_factory() as session:
        score = await get_attempt_score(session, scenario["attempt_id"])
    assert score is not None
    assert score.attempt_id == scenario["attempt_id"]
    assert score.quiz_id == scenario["quiz_id"]
    assert score.student_id == scenario["owner_id"]
    assert score.status == "graded"
    assert score.passed is True
    assert score.score_percent == Decimal("88.50")

    async with session_factory() as session:
        missing = await get_attempt_score(session, uuid.uuid4())
    assert missing is None


@pytest.mark.asyncio
async def test_get_quiz_question_id_set_by_lesson(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """The lesson resolves to the live (non-tombstoned) question id set."""
    async with session_factory() as session:
        ids = await get_quiz_question_id_set_by_lesson(session, scenario["lesson_id"])

    assert isinstance(ids, frozenset)
    assert ids == frozenset({scenario["question_a_id"], scenario["question_b_id"]})

    async with session_factory() as session:
        empty = await get_quiz_question_id_set_by_lesson(session, uuid.uuid4())
    assert empty == frozenset()


@pytest.mark.asyncio
async def test_factory_dto_immutable(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """The DTO returned by the factory is frozen — consumers cannot mutate."""
    async with session_factory() as session, session.begin():
        run = await create_generation_run(
            session,
            kind="knowledge_graph",
            source_scope_kind="lesson",
            course_id=scenario["course_id"],
            module_id=scenario["module_id"],
            lesson_id=scenario["lesson_id"],
            requested_by=scenario["owner_id"],
        )

    with pytest.raises((TypeError, ValueError)):
        run.status = "running"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_factory_does_not_lift_orm_into_caller_scope(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Caller never sees a ``GenerationRun`` ORM instance — only the DTO."""
    async with session_factory() as session, session.begin():
        run = await create_generation_run(
            session,
            kind="material_index",
            source_scope_kind="course",
            course_id=scenario["course_id"],
            requested_by=scenario["owner_id"],
        )
    assert not isinstance(run, GenerationRun)
    assert not isinstance(run, QuizQuestion)
    assert run.generation_type == "material_index"
    assert isinstance(run.created_at, datetime)
    assert run.created_at.tzinfo is not None
    assert run.created_at <= datetime.now(tz=UTC)
