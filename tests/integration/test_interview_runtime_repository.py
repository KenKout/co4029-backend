"""Integration tests for the adaptive-interviewer runtime-state repository.

Exercises the real Postgres row + optimistic-lock semantics against a live
session, proving the Phase 1 concurrency guarantees:

* lazy initialisation creates a version-0 opening-phase row,
* a version-guarded save bumps the version and persists the payload,
* a stale save (wrong expected_version) is rejected → double-advancement
  is impossible,
* duplicate-turn detection short-circuits a replayed student turn,
* a pre-existing session with NO runtime row still loads (compat).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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

import abridgeai.ai.models  # noqa: F401
import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.features.interviews.orchestrator import repository as repo
from abridgeai.features.interviews.orchestrator.state import (
    CoverageStatus,
    InterviewPhase,
    OutcomeCoverageState,
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
async def live_session(engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """Create the minimal FK chain + one in_progress interview session."""
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    session_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"rt-{suffix}", "name": "Runtime Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"rt-student-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": teacher_id, "email": f"rt-teacher-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'RT Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"rt-course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'published')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, created_by) "
                "VALUES (:id, :course, :module, 'RT Interview', 'published', :teacher)"
            ),
            {
                "id": config_id,
                "course": course_id,
                "module": module_id,
                "teacher": teacher_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode) "
                "VALUES (:id, :config, :student, 1, 'in_progress', 'text')"
            ),
            {"id": session_id, "config": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "config_id": config_id, "student_id": student_id}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_runtime_states WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:s, :t)"),
            {"s": student_id, "t": teacher_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


@pytest.mark.asyncio
async def test_load_or_init_creates_version_zero_row(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    async with session_factory() as db:
        loaded = await repo.load_or_init(db, session_id)
        await db.commit()

    assert loaded.version == 0
    assert loaded.data.phase is InterviewPhase.OPENING
    assert loaded.last_turn_idempotency_key is None


@pytest.mark.asyncio
async def test_load_or_init_is_idempotent_no_duplicate_rows(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    async with session_factory() as db:
        await repo.load_or_init(db, session_id)
        await db.commit()
    async with session_factory() as db:
        loaded = await repo.load_or_init(db, session_id)
        await db.commit()
        count = (
            await db.execute(
                text("SELECT count(*) FROM interview_runtime_states WHERE session_id = :s"),
                {"s": session_id},
            )
        ).scalar_one()
    assert count == 1
    assert loaded.version == 0


@pytest.mark.asyncio
async def test_save_bumps_version_and_persists_payload(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    async with session_factory() as db:
        loaded = await repo.load_or_init(db, session_id)
        loaded.data.phase = InterviewPhase.CORE
        loaded.data.asked_question_ids = ["q-1"]
        loaded.data.outcome_coverage["o-1"] = OutcomeCoverageState(
            outcome_id="o-1", evidence_count=1, status=CoverageStatus.PARTIAL
        )
        new_version = await repo.save(
            db, session_id, loaded.data, expected_version=0, turn_idempotency_key="turn-1"
        )
        await db.commit()

    assert new_version == 1

    async with session_factory() as db:
        reloaded = await repo.load_or_init(db, session_id)
        await db.commit()

    assert reloaded.version == 1
    assert reloaded.data.phase is InterviewPhase.CORE
    assert reloaded.data.asked_question_ids == ["q-1"]
    assert reloaded.data.outcome_coverage["o-1"].evidence_count == 1
    assert reloaded.last_turn_idempotency_key == "turn-1"


@pytest.mark.asyncio
async def test_stale_save_is_rejected(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    """A save with an out-of-date expected_version must raise — this is what
    prevents a retried REST call / duplicate LiveKit callback from advancing
    the interview twice."""
    session_id = live_session["session_id"]
    async with session_factory() as db:
        loaded = await repo.load_or_init(db, session_id)
        await repo.save(db, session_id, loaded.data, expected_version=0)
        await db.commit()

    # Second save still thinks it's at version 0, but the row is now at 1.
    async with session_factory() as db:
        loaded_stale = repo.LoadedRuntimeState(
            data=loaded.data, version=0, last_turn_idempotency_key=None
        )
        with pytest.raises(repo.StaleStateError):
            await repo.save(db, session_id, loaded_stale.data, expected_version=0)
        await db.rollback()


@pytest.mark.asyncio
async def test_duplicate_turn_detection(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    async with session_factory() as db:
        loaded = await repo.load_or_init(db, session_id)
        await repo.save(
            db, session_id, loaded.data, expected_version=0, turn_idempotency_key="turn-abc"
        )
        await db.commit()

    async with session_factory() as db:
        reloaded = await repo.load_or_init(db, session_id)
        await db.commit()

    # Same key → duplicate replay; a different/None key → not a duplicate.
    assert repo.is_duplicate_turn(reloaded, "turn-abc") is True
    assert repo.is_duplicate_turn(reloaded, "turn-xyz") is False
    assert repo.is_duplicate_turn(reloaded, None) is False
