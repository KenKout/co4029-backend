"""Integration tests for ``features.quizzes.services.generation`` (T5.13).

Acceptance gate: the dispatcher routes 3 distinct config shapes to 3
distinct pipelines.

* ``run.config_json["question_id"]`` set → ``run_question_regeneration``.
* ``run.config_json["generation_mode"] == "coverage"`` →
  ``run_coverage_pipeline``.
* otherwise → ``run_full_pipeline``.

Each pipeline is mocked via ``monkeypatch`` so the test asserts which
pipeline was invoked, not the pipeline's internals (those have their
own integration tests).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import (  # noqa: E402
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
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
from abridgeai.ai.models import GenerationRun
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.features.quizzes.services import generation as generation_service

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
            {"id": org_id, "slug": f"qg-{suffix}", "name": "Gen Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qg-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Gen Test Course', 'draft')"
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
                "VALUES (:id, :course, 'Module 1', 1, 'draft')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :module, 'Dispatch Quiz', 'draft')"
            ),
            {"id": quiz_id, "course": course_id, "module": module_id},
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
            text("DELETE FROM generation_runs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_options WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id = :q)"
            ),
            {"q": quiz_id},
        )
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = :q"),
            {"q": quiz_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE id = :id"), {"id": quiz_id})
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def _seed_run(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    requested_by: uuid.UUID,
    config: dict,
) -> uuid.UUID:
    async with session_factory() as session, session.begin():
        run = GenerationRun(
            generation_type="quiz",
            source_scope_kind="module",
            course_id=course_id,
            module_id=module_id,
            requested_by=requested_by,
            status="pending",
            config_json=config,
        )
        session.add(run)
        await session.flush()
        return run.id


@pytest.mark.asyncio
async def test_dispatch_full_mode_routes_to_full_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    full_mock = AsyncMock(return_value=[])
    coverage_mock = AsyncMock(return_value=[])
    regen_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(generation_service.full_pipeline, "run_full_pipeline", full_mock)
    monkeypatch.setattr(
        generation_service.coverage_pipeline, "run_coverage_pipeline", coverage_mock
    )
    monkeypatch.setattr(
        generation_service.regenerate_pipeline, "run_question_regeneration", regen_mock
    )

    run_id = await _seed_run(
        session_factory,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        requested_by=scenario["owner_id"],
        config={"quiz_id": str(scenario["quiz_id"]), "question_count": 3},
    )

    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    assert full_mock.await_count == 1
    assert coverage_mock.await_count == 0
    assert regen_mock.await_count == 0


@pytest.mark.asyncio
async def test_dispatch_coverage_mode_routes_to_coverage_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    full_mock = AsyncMock(return_value=[])
    coverage_mock = AsyncMock(return_value=[])
    regen_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(generation_service.full_pipeline, "run_full_pipeline", full_mock)
    monkeypatch.setattr(
        generation_service.coverage_pipeline, "run_coverage_pipeline", coverage_mock
    )
    monkeypatch.setattr(
        generation_service.regenerate_pipeline, "run_question_regeneration", regen_mock
    )
    # The service now precomputes outlines+budget (requires source_lesson_ids
    # and real chunks) BEFORE dispatching; this test is about ROUTING only.
    monkeypatch.setattr(
        generation_service,
        "_precompute_coverage_inputs",
        AsyncMock(return_value=([], {})),
    )

    run_id = await _seed_run(
        session_factory,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        requested_by=scenario["owner_id"],
        config={
            "quiz_id": str(scenario["quiz_id"]),
            "generation_mode": "coverage",
            "question_count": 3,
        },
    )

    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    assert coverage_mock.await_count == 1
    assert full_mock.await_count == 0
    assert regen_mock.await_count == 0


@pytest.mark.asyncio
async def test_dispatch_regenerate_routes_to_regenerate_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    full_mock = AsyncMock(return_value=[])
    coverage_mock = AsyncMock(return_value=[])
    regen_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(generation_service.full_pipeline, "run_full_pipeline", full_mock)
    monkeypatch.setattr(
        generation_service.coverage_pipeline, "run_coverage_pipeline", coverage_mock
    )
    monkeypatch.setattr(
        generation_service.regenerate_pipeline, "run_question_regeneration", regen_mock
    )

    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status) "
                "VALUES (:id, :quiz_id, 1, 'multiple_choice', 'seed?', 'pending')"
            ),
            {"id": question_id, "quiz_id": scenario["quiz_id"]},
        )

    run_id = await _seed_run(
        session_factory,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        requested_by=scenario["owner_id"],
        config={"question_id": str(question_id), "quiz_id": str(scenario["quiz_id"])},
    )

    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    assert regen_mock.await_count == 1
    assert full_mock.await_count == 0
    assert coverage_mock.await_count == 0


@pytest.mark.asyncio
async def test_pipeline_failure_marks_run_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    boom = AsyncMock(side_effect=ValueError("pipeline exploded"))
    monkeypatch.setattr(generation_service.full_pipeline, "run_full_pipeline", boom)

    run_id = await _seed_run(
        session_factory,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        requested_by=scenario["owner_id"],
        config={"quiz_id": str(scenario["quiz_id"]), "question_count": 3},
    )

    async with session_factory() as session:
        with pytest.raises(ValueError, match="pipeline exploded"):
            await generation_service.run_quiz_generation(session, run_id)

    async with session_factory() as session:
        run = await session.get(GenerationRun, run_id)
        assert run is not None
        assert run.status == "failed"
        assert run.finished_at is not None
        assert run.config_json.get("failure", {}).get("message") == "pipeline exploded"


@pytest.mark.asyncio
async def test_dispatch_completion_marks_run_completed(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    monkeypatch.setattr(
        generation_service.full_pipeline, "run_full_pipeline", AsyncMock(return_value=[])
    )

    run_id = await _seed_run(
        session_factory,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        requested_by=scenario["owner_id"],
        config={"quiz_id": str(scenario["quiz_id"]), "question_count": 3},
    )

    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    async with session_factory() as session:
        run = await session.get(GenerationRun, run_id)
        assert run is not None
        assert run.status == "completed"
        assert run.started_at is not None
        assert run.finished_at is not None
