"""Integration tests for the publish gate (T7.5.9).

Every ``QuizQuestion`` must have ``expected_response_time_ms`` set to a
positive integer before the quiz can be published. The bulk-set
endpoint exists so teachers can fix gate failures in one request.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
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
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.services import authoring as authoring_service
from abridgeai.features.quizzes.services.authoring import QuizPublishValidationError

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
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
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
    quiz_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"pg-{suffix}", "name": "Publish Gate Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"pg-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Publish Gate Course', 'draft')"
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
                "INSERT INTO quizzes (id, course_id, module_id, title, status, created_by, slug) VALUES (:id, :course, :module, 'Publish Gate Quiz', 'draft', :owner, 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "id": quiz_id,
                "course": course_id,
                "module": module_id,
                "owner": owner_id,
            },
        )

    yield {
        "owner_id": owner_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
        "quiz_id": quiz_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_question_revisions WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id = :q)"
            ),
            {"q": quiz_id},
        )
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = :q"),
            {"q": quiz_id},
        )
        await conn.execute(text("DELETE FROM module_items WHERE quiz_id = :q"), {"q": quiz_id})
        await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def _insert_question(
    engine: AsyncEngine,
    *,
    quiz_id: uuid.UUID,
    position: int,
    expected_response_time_ms: int | None,
    review_status: str = "approved",
) -> uuid.UUID:
    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status, "
                "expected_response_time_ms) "
                "VALUES (:id, :qz, :pos, 'multiple_choice', :prompt, :review_status, :ms)"
            ),
            {
                "id": question_id,
                "qz": quiz_id,
                "pos": position,
                "prompt": f"Question {position}?",
                "review_status": review_status,
                "ms": expected_response_time_ms,
            },
        )
    return question_id


@pytest.mark.asyncio
async def test_publish_with_all_t_exp_set_succeeds(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    quiz_id = scenario["quiz_id"]
    for pos in (1, 2, 3):
        await _insert_question(
            engine, quiz_id=quiz_id, position=pos, expected_response_time_ms=30_000
        )

    async with session_factory() as session, session.begin():
        quiz = await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))

    assert quiz.status == "published"
    assert quiz.published_at is not None


@pytest.mark.asyncio
async def test_publish_with_one_missing_returns_422(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    quiz_id = scenario["quiz_id"]
    q1 = await _insert_question(
        engine, quiz_id=quiz_id, position=1, expected_response_time_ms=30_000
    )
    q2 = await _insert_question(engine, quiz_id=quiz_id, position=2, expected_response_time_ms=None)
    q3 = await _insert_question(
        engine, quiz_id=quiz_id, position=3, expected_response_time_ms=45_000
    )

    async with session_factory() as session, session.begin():
        with pytest.raises(QuizPublishValidationError) as exc_info:
            await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))

    missing = exc_info.value.missing_t_exp_question_ids
    assert q2 in missing
    assert q1 not in missing
    assert q3 not in missing
    assert len(missing) == 1


@pytest.mark.asyncio
async def test_publish_with_zero_t_exp_returns_422(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    quiz_id = scenario["quiz_id"]
    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

        with pytest.raises(IntegrityError):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, "
                    "review_status, expected_response_time_ms) "
                    "VALUES (:id, :qz, 1, 'multiple_choice', 'Q?', 'approved', 0)"
                ),
                {"id": question_id, "qz": quiz_id},
            )

    await _insert_question(engine, quiz_id=quiz_id, position=1, expected_response_time_ms=None)
    async with session_factory() as session, session.begin():
        with pytest.raises(QuizPublishValidationError):
            await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))


@pytest.mark.asyncio
async def test_bulk_set_expected_time(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    quiz_id = scenario["quiz_id"]
    q1 = await _insert_question(engine, quiz_id=quiz_id, position=1, expected_response_time_ms=None)
    q2 = await _insert_question(engine, quiz_id=quiz_id, position=2, expected_response_time_ms=None)

    async with session_factory() as session, session.begin():
        updated = await authoring_service.bulk_set_expected_response_time(
            session,
            quiz_id,
            [(q1, 30_000), (q2, 45_000)],
            _actor(scenario["owner_id"]),
        )
    assert updated == 2

    async with session_factory() as session, session.begin():
        quiz = await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))
    assert quiz.status == "published"


@pytest.mark.asyncio
async def test_partial_publish_pending_questions_do_not_block(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Partial publish: a mix of approved + pending publishes fine.

    Students only ever see approved questions, so a pending draft must not
    block publish — it's retained for later reuse. This is the "generate 30,
    approve 15, publish, keep the rest" workflow.
    """
    quiz_id = scenario["quiz_id"]
    await _insert_question(
        engine,
        quiz_id=quiz_id,
        position=1,
        expected_response_time_ms=30_000,
        review_status="approved",
    )
    # Pending draft with NO expected time — must neither block the approval
    # gate nor the t_exp gate, because students never see it.
    await _insert_question(
        engine,
        quiz_id=quiz_id,
        position=2,
        expected_response_time_ms=None,
        review_status="pending",
    )

    async with session_factory() as session, session.begin():
        quiz = await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))
    assert quiz.status == "published"


@pytest.mark.asyncio
async def test_publish_with_zero_approved_returns_422(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """A quiz with only pending/rejected questions can't publish (empty for students)."""
    from abridgeai.features.quizzes.services.publish_gate import QuizApprovalRequiredError

    quiz_id = scenario["quiz_id"]
    await _insert_question(
        engine,
        quiz_id=quiz_id,
        position=1,
        expected_response_time_ms=30_000,
        review_status="pending",
    )

    async with session_factory() as session, session.begin():
        with pytest.raises(QuizApprovalRequiredError):
            await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))


@pytest.mark.asyncio
async def test_already_published_idempotent(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    quiz_id = scenario["quiz_id"]
    await _insert_question(engine, quiz_id=quiz_id, position=1, expected_response_time_ms=25_000)

    async with session_factory() as session, session.begin():
        first = await authoring_service.publish_quiz(session, quiz_id, _actor(scenario["owner_id"]))
    assert first.status == "published"

    async with session_factory() as session, session.begin():
        second = await authoring_service.publish_quiz(
            session, quiz_id, _actor(scenario["owner_id"])
        )
    assert second.status == "published"
