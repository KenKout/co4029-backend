"""Integration tests for ``features.quizzes.routers`` (T5.14).

Smoke coverage of the eleven authoring + six learner endpoints:

* router metadata (prefix, route count) for both routers
* generation lifecycle: ``POST /generate`` enqueues ARQ + returns
  ``run_id``, ``GET /generation-runs/{run_id}`` polls status
* learner ``GET /quizzes/{quiz_id}`` and ``POST /attempts`` payloads
  do NOT leak ``is_correct``
* FIX-SEC-1 source-grep guard on ``authoring.py`` (no bare
  ``Depends(get_current_user)`` outside dependency factories)
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
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.quizzes.routers import authoring_router, learner_router
from abridgeai.features.quizzes.routers.authoring import get_arq_pool

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
    fastapi_app.dependency_overrides[get_arq_pool] = _override_arq_pool
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
                "VALUES (:m, :c, 'Quiz Test Module', 1, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )

    yield {"course_id": seeded_users.course_id, "module_id": module_id}

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_attempt_answers WHERE attempt_id IN ("
                "  SELECT id FROM quiz_attempts WHERE quiz_id IN ("
                "    SELECT id FROM quizzes WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_attempts WHERE quiz_id IN ("
                "  SELECT id FROM quizzes WHERE module_id = :m"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_revisions WHERE question_id IN ("
                "  SELECT id FROM quiz_questions WHERE quiz_id IN ("
                "    SELECT id FROM quizzes WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_options WHERE question_id IN ("
                "  SELECT id FROM quiz_questions WHERE quiz_id IN ("
                "    SELECT id FROM quizzes WHERE module_id = :m"
                "  )"
                ")"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "  SELECT id FROM quizzes WHERE module_id = :m"
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


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Router metadata
# ---------------------------------------------------------------------------


def test_authoring_router_metadata() -> None:
    assert authoring_router.prefix == "/teacher"
    paths = {route.path for route in authoring_router.routes}  # type: ignore[attr-defined]
    assert "/teacher/courses/{course_id}/quizzes" in paths
    assert "/teacher/quizzes/{quiz_id}" in paths
    assert "/teacher/quizzes/{quiz_id}/publish" in paths
    assert "/teacher/quizzes/{quiz_id}/generate" in paths
    assert "/teacher/quizzes/{quiz_id}/generation-runs/{run_id}" in paths
    assert "/teacher/quizzes/{quiz_id}/questions" in paths
    assert "/teacher/quizzes/{quiz_id}/questions/{question_id}" in paths
    assert "/teacher/quizzes/{quiz_id}/questions/{question_id}/regenerate" in paths


def test_learner_router_metadata() -> None:
    assert learner_router.prefix == ""
    paths = {route.path for route in learner_router.routes}  # type: ignore[attr-defined]
    assert "/quizzes/{quiz_id}" in paths
    assert "/quizzes/{quiz_id}/attempts" in paths
    assert "/attempts/{attempt_id}/answers" in paths
    assert "/attempts/{attempt_id}/submit" in paths
    assert "/attempts/{attempt_id}" in paths
    assert "/me/quizzes/{quiz_id}/attempts" in paths


# ---------------------------------------------------------------------------
# Generation lifecycle (T5.14 acceptance gate)
# ---------------------------------------------------------------------------


async def test_generation_lifecycle_enqueue_and_poll(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    app: tuple[FastAPI, AsyncMock],
    seeded_users: SeededUsers,
) -> None:
    _, arq_pool = app
    arq_pool.enqueue_job.reset_mock()

    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={
            "module_id": str(scenario["module_id"]),
            "title": "Lifecycle Quiz",
            "description": "Manually created",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = create_resp.json()["id"]

    gen_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/generate",
        json={
            "mode": "full",
            "target_count": 5,
            "focus_topics": [],
            "source_lessons": [],
        },
        headers=_auth(admin_bearer),
    )
    assert gen_resp.status_code == 202, gen_resp.text
    run = gen_resp.json()
    assert run["status"] == "pending"
    run_id = run["id"]
    assert run["quiz_id"]

    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "run_quiz_generation_task"

    poll_resp = await client.get(
        f"/api/v1/teacher/quizzes/{run['quiz_id']}/generation-runs/{run_id}",
        headers=_auth(admin_bearer),
    )
    assert poll_resp.status_code == 200, poll_resp.text
    assert poll_resp.json()["id"] == run_id
    assert poll_resp.json()["status"] in {"pending", "running", "completed", "failed"}
    del seeded_users


# ---------------------------------------------------------------------------
# Learner — no is_correct leak
# ---------------------------------------------------------------------------


async def test_learner_quiz_response_omits_is_correct(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={
            "module_id": str(scenario["module_id"]),
            "title": "Leak Test Quiz",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201
    quiz_id = uuid.UUID(create_resp.json()["id"])

    pub_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert pub_resp.status_code == 200, pub_resp.text

    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status) "
                "VALUES (:id, :qz, 1, 'multiple_choice', 'What is 2+2?', 'approved')"
            ),
            {"id": question_id, "qz": quiz_id},
        )
        for option_key, option_text, is_correct, position in (
            ("A", "3", False, 1),
            ("B", "4", True, 2),
            ("C", "5", False, 3),
            ("D", "6", False, 4),
        ):
            await conn.execute(
                text(
                    "INSERT INTO quiz_question_options "
                    "(id, question_id, option_key, option_text, is_correct, position) "
                    "VALUES (uuid_generate_v4(), :q, :k, :t, :c, :p)"
                ),
                {
                    "q": question_id,
                    "k": option_key,
                    "t": option_text,
                    "c": is_correct,
                    "p": position,
                },
            )

    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    quiz_resp = await client.get(
        f"/api/v1/quizzes/{quiz_id}",
        headers=_auth(student_token),
    )
    assert quiz_resp.status_code == 200, quiz_resp.text
    body = quiz_resp.json()
    assert "is_correct" not in json.dumps(body)

    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201, attempt_resp.text
    take_body = attempt_resp.json()
    assert take_body["questions"]
    serialized = json.dumps(take_body)
    assert "is_correct" not in serialized
    for question in take_body["questions"]:
        for option in question["options"]:
            assert "is_correct" not in option

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


# ---------------------------------------------------------------------------
# Source-grep guard (FIX-SEC-1 perimeter)
# ---------------------------------------------------------------------------


def test_no_bare_get_current_user_on_quiz_authoring_endpoints() -> None:
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "quizzes"
        / "routers"
        / "authoring.py"
    ).read_text(encoding="utf-8")
    code_only = re.sub(r'"""[\s\S]*?"""', "", src)
    bare = re.findall(r"Depends\(get_current_user\)", code_only)
    assert bare == [], f"authoring.py uses bare Depends(get_current_user): {bare}"
