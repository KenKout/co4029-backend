"""Concurrent question deletes on one quiz must not deadlock.

Reported as a 500 on ``DELETE /teacher/quizzes/{id}/questions/{qid}``:

    sqlalchemy.exc.OperationalError: (psycopg.errors.DeadlockDetected)
    deadlock detected
    CONTEXT: while locking tuple (48,5) in relation "quiz_questions"

``delete_question`` soft-deletes the row then REPACKS every surviving sibling's
``position`` back to a dense 1..N. Two deletes running concurrently against the
same quiz each take row locks across that same sibling set and interleave, so
each waits on a row the other already holds.

This is a normal path, not an edge case: the teacher UI stages bulk deletes and
flushes them with ``Promise.allSettled``, i.e. every DELETE fires at once.

``delete_question`` now takes a transaction-scoped advisory lock keyed on the
quiz before repacking, so same-quiz deletes queue while different-quiz deletes
still run in parallel.

Measured before the fix: 4 of 5 concurrent deletes failed with DeadlockDetected.
After: 5 of 5 succeed and positions stay dense.

Fixture pattern mirrors ``test_quiz_authoring_service.py`` (self-contained
engine + scenario) because the integration suite has no shared DB fixtures.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import Column, Table, select, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- FK targets
import abridgeai.features.courses.models  # noqa: F401  -- FK targets
import abridgeai.features.identity.models  # noqa: F401  -- users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.services import authoring as authoring_service

for _stub_name in ("interview_configs", "learning_materials", "learning_material_versions"):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )

QUESTION_COUNT = 8
CONCURRENT_DELETES = 5


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
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qd-{suffix}", "name": "Delete Concurrency Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qd-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Delete Course', 'draft')"
            ),
            {"id": course_id, "org": org_id, "owner": owner_id, "slug": f"course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'draft')"
            ),
            {"id": module_id, "course": course_id},
        )

    yield {"owner_id": owner_id, "org_id": org_id, "course_id": course_id, "module_id": module_id}

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_question_revisions WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM module_items WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM quizzes WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def _seed_questions(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict
) -> tuple[uuid.UUID, list[uuid.UUID]]:
    async with session_factory() as db:
        quiz = Quiz(
            id=uuid.uuid4(),
            course_id=scenario["course_id"],
            module_id=scenario["module_id"],
            title="deadlock regression",
            status="draft",
            created_by=scenario["owner_id"],
        )
        db.add(quiz)
        await db.flush()
        ids: list[uuid.UUID] = []
        for position in range(1, QUESTION_COUNT + 1):
            question = QuizQuestion(
                id=uuid.uuid4(),
                quiz_id=quiz.id,
                position=position,
                question_type="short_answer",
                prompt_text=f"question {position}",
                explanation="e",
                difficulty="medium",
                bloom_level="understand",
                review_status="pending",
                expected_response_time_ms=60000,
                created_by=scenario["owner_id"],
            )
            db.add(question)
            ids.append(question.id)
        await db.commit()
        return quiz.id, ids


@pytest.mark.asyncio
async def test_concurrent_deletes_do_not_deadlock(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    quiz_id, question_ids = await _seed_questions(session_factory, scenario)
    targets = question_ids[:CONCURRENT_DELETES]
    actor = _actor(scenario["owner_id"])

    async def delete_one(question_id: uuid.UUID) -> str:
        # Own session/transaction per delete — mirrors separate HTTP requests.
        async with session_factory() as db:
            try:
                await authoring_service.delete_question(db, question_id, actor)
                await db.commit()
                return "ok"
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                return type(exc).__name__

    results = await asyncio.gather(*(delete_one(qid) for qid in targets))

    assert all(r == "ok" for r in results), f"concurrent deletes failed: {results}"

    async with session_factory() as db:
        positions = (
            await db.execute(
                select(QuizQuestion.position)
                .where(
                    QuizQuestion.quiz_id == quiz_id,
                    QuizQuestion.deleted_at.is_(None),
                )
                .order_by(QuizQuestion.position)
            )
        ).scalars().all()

    # Positions must stay a dense 1..N with no gaps or duplicates.
    survivors = QUESTION_COUNT - CONCURRENT_DELETES
    assert list(positions) == list(range(1, survivors + 1))
