"""Integration tests for ``features.interviews.routers`` (T6.12).

Smoke coverage of the seven learner + fourteen authoring endpoints:

* router metadata (prefix + route count) for both routers
* generation lifecycle: ``POST /generate`` enqueues
  ``run_interview_generation_task`` and returns 202
* finish lifecycle: ``POST /finish`` enqueues
  ``evaluate_interview_session_task``
* learner ``GET /interview-configs/{id}`` payload omits authoring-only
  fields (no answer leak)
* cross-user session boundary: user A starts session, user B GET 403
* audio_storage_object_id passes through but is not transcribed
* FIX-SEC-1 source-grep guard on ``authoring.py`` (no bare
  ``Depends(get_current_user)``)
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz tables (GenerationRun)
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.interviews.routers import authoring_router, learner_router
from abridgeai.features.interviews.routers.authoring import (
    get_arq_pool as get_authoring_arq_pool,
)
from abridgeai.features.interviews.routers.learner import get_arq_pool as get_learner_arq_pool

for _stub_name in ("learning_materials", "learning_material_versions"):
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
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[FastAPI, AsyncMock]]:
    arq_pool = AsyncMock()
    arq_pool.enqueue_job = AsyncMock()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _override_arq_pool() -> object:
        return arq_pool

    fastapi_app = FastAPI()
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    fastapi_app.dependency_overrides[get_authoring_arq_pool] = _override_arq_pool
    fastapi_app.dependency_overrides[get_learner_arq_pool] = _override_arq_pool
    yield fastapi_app, arq_pool
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: tuple[FastAPI, AsyncMock]) -> AsyncIterator[httpx.AsyncClient]:
    fastapi_app, _ = app
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return session_id


@pytest_asyncio.fixture
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    module_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Interview Test Module', 1, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )

    yield {"course_id": seeded_users.course_id, "module_id": module_id}

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM interview_session_messages WHERE session_id IN ("
                "  SELECT id FROM interview_sessions WHERE interview_config_id IN ("
                "    SELECT id FROM interview_configs WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_session_questions WHERE session_id IN ("
                "  SELECT id FROM interview_sessions WHERE interview_config_id IN ("
                "    SELECT id FROM interview_configs WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcome_evaluations WHERE session_id IN ("
                "  SELECT id FROM interview_sessions WHERE interview_config_id IN ("
                "    SELECT id FROM interview_configs WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM gap_reports WHERE source_interview_session_id IN ("
                "  SELECT id FROM interview_sessions WHERE interview_config_id IN ("
                "    SELECT id FROM interview_configs WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_sessions WHERE interview_config_id IN ("
                "  SELECT id FROM interview_configs WHERE module_id = :m"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_questions WHERE interview_config_id IN ("
                "  SELECT id FROM interview_configs WHERE module_id = :m"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcomes WHERE interview_config_id IN ("
                "  SELECT id FROM interview_configs WHERE module_id = :m"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Router metadata
# ---------------------------------------------------------------------------


def test_authoring_endpoints_registered() -> None:
    assert authoring_router.prefix == "/teacher"
    paths = {route.path for route in authoring_router.routes}  # type: ignore[attr-defined]
    expected = {
        "/teacher/courses/{course_id}/interview-configs",
        "/teacher/interview-configs/{config_id}",
        "/teacher/interview-configs/{config_id}/publish",
        "/teacher/interview-configs/{config_id}/generate",
        "/teacher/interview-configs/{config_id}/generation-runs/{run_id}",
        "/teacher/interview-configs/{config_id}/questions",
        "/teacher/interview-configs/{config_id}/questions/{question_id}",
        "/teacher/interview-configs/{config_id}/questions/{question_id}/regenerate",
        "/teacher/interview-configs/{config_id}/outcomes",
        "/teacher/interview-configs/{config_id}/outcomes/{outcome_id}",
        "/teacher/interview-sessions/{session_id}/gap-report",
    }
    assert expected.issubset(paths), expected - paths


def test_learner_session_endpoints_registered() -> None:
    assert learner_router.prefix == ""
    paths = {route.path for route in learner_router.routes}  # type: ignore[attr-defined]
    expected = {
        "/interview-configs/{config_id}",
        "/interview-configs/{config_id}/sessions",
        "/interview-sessions/{session_id}",
        "/interview-sessions/{session_id}/respond",
        "/interview-sessions/{session_id}/finish",
        "/interview-sessions/{session_id}/gap-report",
        "/me/interview-sessions",
    }
    assert expected.issubset(paths), expected - paths


# ---------------------------------------------------------------------------
# Generation lifecycle (T6.12 acceptance gate)
# ---------------------------------------------------------------------------


async def test_generate_returns_202_and_enqueues_arq(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    app: tuple[FastAPI, AsyncMock],
    seeded_users: SeededUsers,
) -> None:
    _, arq_pool = app
    arq_pool.enqueue_job.reset_mock()

    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/interview-configs",
        json={
            "title": "Lifecycle Interview",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "supported_modes": "text",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    config_id = create_resp.json()["id"]

    gen_resp = await client.post(
        f"/api/v1/teacher/interview-configs/{config_id}/generate",
        json={
            "mode": "topic",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "question_count": 3,
            "focus_topics": ["lists"],
        },
        headers=_auth(admin_bearer),
    )
    assert gen_resp.status_code == 202, gen_resp.text
    run = gen_resp.json()
    assert run["status"] == "pending"
    run_id = run["run_id"]

    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "run_interview_generation_task"

    poll_resp = await client.get(
        f"/api/v1/teacher/interview-configs/{config_id}/generation-runs/{run_id}",
        headers=_auth(admin_bearer),
    )
    assert poll_resp.status_code == 200, poll_resp.text
    assert poll_resp.json()["run_id"] == run_id
    del seeded_users


# ---------------------------------------------------------------------------
# Cross-user session boundary (FIX-SEC-1 explicit acceptance)
# ---------------------------------------------------------------------------


async def _seed_published_config(
    engine: AsyncEngine,
    *,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> uuid.UUID:
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, supported_modes, created_by) "
                "VALUES (:id, :c, :m, 'Cross-user', 'published', 'text', :u)"
            ),
            {"id": config_id, "c": course_id, "m": module_id, "u": actor_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, position, question_type, prompt_text, "
                " review_status, ai_generated, source_refs_json, created_by) "
                "VALUES (:id, :cfg, 1, 'conceptual', 'What is recursion?', "
                "        'approved', false, '[]'::jsonb, :u)"
            ),
            {"id": question_id, "cfg": config_id, "u": actor_id},
        )
    return config_id


async def test_other_user_session_403(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )

    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    teacher_sid = await _seed_session(engine, seeded_users.teacher_id)
    teacher_token = create_access_token(user_id=seeded_users.teacher_id, session_id=teacher_sid)

    start_resp = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "text"},
        headers=_auth(student_token),
    )
    assert start_resp.status_code == 201, start_resp.text
    session_id = start_resp.json()["session_id"]

    intruder = await client.get(
        f"/api/v1/interview-sessions/{session_id}",
        headers=_auth(teacher_token),
    )
    assert intruder.status_code == 403, intruder.text

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id IN (:s, :t)"),
            {"s": student_sid, "t": teacher_sid},
        )


# ---------------------------------------------------------------------------
# Voice future-proofing — audio_storage_object_id accepted, not transcribed
# ---------------------------------------------------------------------------


async def test_audio_storage_object_id_accepted_but_not_transcribed(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    monkeypatch: object,
) -> None:
    from abridgeai.features.interviews.services import taking as taking_service  # noqa: PLC0415

    async def _no_followup(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(taking_service, "maybe_generate_followup", _no_followup)  # type: ignore[attr-defined]

    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    start_resp = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "hybrid"},
        headers=_auth(student_token),
    )
    assert start_resp.status_code == 201, start_resp.text
    session_id = uuid.UUID(start_resp.json()["session_id"])

    audio_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type, uploaded_by) "
                "VALUES (:id, 'interviews', :key, 'audio/webm', :u)"
            ),
            {"id": audio_id, "key": f"audio/{audio_id}.webm", "u": seeded_users.student_id},
        )
    respond_resp = await client.post(
        f"/api/v1/interview-sessions/{session_id}/respond",
        json={
            "session_id": str(session_id),
            "session_question_id": str(uuid.uuid4()),
            "answer_text": "spoken answer transcript",
            "audio_object_id": str(audio_id),
        },
        headers=_auth(student_token),
    )
    assert respond_resp.status_code == 200, respond_resp.text

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT audio_object_id FROM interview_session_messages "
                    "WHERE session_id = :s AND role = 'user' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"s": session_id},
            )
        ).first()
    assert row is not None
    assert row[0] == audio_id

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


# ---------------------------------------------------------------------------
# Finish enqueues evaluation
# ---------------------------------------------------------------------------


async def test_finish_returns_score_and_enqueues_evaluation(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    app: tuple[FastAPI, AsyncMock],
    seeded_users: SeededUsers,
) -> None:
    _, arq_pool = app
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    start_resp = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "text"},
        headers=_auth(student_token),
    )
    assert start_resp.status_code == 201
    session_id = start_resp.json()["session_id"]

    arq_pool.enqueue_job.reset_mock()

    finish_resp = await client.post(
        f"/api/v1/interview-sessions/{session_id}/finish",
        headers=_auth(student_token),
    )
    assert finish_resp.status_code == 200, finish_resp.text
    body = finish_resp.json()
    assert body["session_id"] == session_id
    assert body["status"] == "completed"

    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "evaluate_interview_session_task"

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


# ---------------------------------------------------------------------------
# Learner — no is_correct / authoring fields leak
# ---------------------------------------------------------------------------


async def test_no_is_correct_in_learner_responses(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    take_resp = await client.get(
        f"/api/v1/interview-configs/{config_id}",
        headers=_auth(student_token),
    )
    assert take_resp.status_code == 200, take_resp.text
    serialized = json.dumps(take_resp.json())
    forbidden_keys = (
        "is_correct",
        "importance_weight",
        "difficulty",
        "review_status",
        "ai_generated",
        "source_refs_json",
        "supplementary_instructions",
        "min_outcomes_to_pass",
    )
    for key in forbidden_keys:
        assert key not in serialized, f"learner take payload leaks {key!r}"

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


# ---------------------------------------------------------------------------
# Source-grep guard (FIX-SEC-1 perimeter)
# ---------------------------------------------------------------------------


def test_no_bare_get_current_user_on_authoring_writes() -> None:
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "interviews"
        / "routers"
        / "authoring.py"
    ).read_text(encoding="utf-8")
    code_only = re.sub(r'"""[\s\S]*?"""', "", src)
    bare = re.findall(r"Depends\(get_current_user\)", code_only)
    assert bare == [], f"authoring.py uses bare Depends(get_current_user): {bare}"
