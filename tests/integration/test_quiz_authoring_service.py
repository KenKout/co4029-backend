"""Integration tests for ``features.quizzes.services.authoring`` (T5.13).

Covers the create-quiz happy path + start_generation_run enqueue
contract.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    Table,
    text,
)
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
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import GenerationRun, Quiz
from abridgeai.features.quizzes.services import authoring as authoring_service

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
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qa-{suffix}", "name": "Auth Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qa-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Auth Course', 'draft')"
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

    yield {
        "owner_id": owner_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


class _CreatePayload:
    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def model_dump(self, exclude_unset: bool = False) -> dict:
        del exclude_unset
        return dict(self._fields)


@pytest.mark.asyncio
async def test_create_quiz_inserts_row_and_links_module_item(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    payload = _CreatePayload(
        title="Photosynthesis Quiz",
        description="Auto-graded MCQs",
        time_limit_seconds=600,
    )

    async with session_factory() as session, session.begin():
        quiz = await authoring_service.create_quiz(
            session, scenario["module_id"], payload, _actor(scenario["owner_id"])
        )

    assert isinstance(quiz, Quiz)
    assert quiz.title == "Photosynthesis Quiz"
    assert quiz.module_id == scenario["module_id"]
    assert quiz.course_id == scenario["course_id"]

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    text("SELECT item_type, quiz_id FROM module_items WHERE module_id = :m"),
                    {"m": scenario["module_id"]},
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
    assert rows[0]["item_type"] == "quiz"
    assert rows[0]["quiz_id"] == quiz.id


@pytest.mark.asyncio
async def test_start_generation_run_creates_run_quiz_and_enqueues_job(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    payload = SimpleNamespace(
        quiz_id=None,
        title="AI-Generated Quiz",
        description="Created by start_generation_run",
        question_count=5,
        question_types=["multiple_choice"],
        difficulty="medium",
        bloom_distribution={"understand": 1.0},
        include_prerequisites=True,
        model_preference=None,
        source_lesson_ids=[],
        generation_mode="topic",
        focus_topics=[],
        avoid_topics=[],
        extra_instructions=None,
        append=False,
        coverage_options=None,
        config_json={},
    )

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            scenario["module_id"],
            payload,
            _actor(scenario["owner_id"]),
            arq_pool=arq_pool,
        )

    assert isinstance(run, GenerationRun)
    assert run.status == "pending"
    assert run.config_json["quiz_id"]
    assert run.module_id == scenario["module_id"]
    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "generate_quiz"
    assert invocation.args[1] == str(run.id)
