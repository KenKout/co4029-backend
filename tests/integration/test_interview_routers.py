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

import abridgeai.ai.models  # noqa: F401  -- register generation_runs (FK target for interview_configs.generation_run_id)
import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz tables
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
        "/teacher/interview-sessions/{session_id}",
        "/teacher/interview-sessions/{session_id}/gap-report",
        "/teacher/courses/{course_id}/interview-sessions",
        "/teacher/courses/{course_id}/students/{student_id}/interview-sessions",
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
    outcome_id = uuid.uuid4()
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
        # Thesis §4.3: a startable interview must have at least one learning
        # outcome (start_session blocks outcome-less configs).
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes "
                "(id, interview_config_id, position, outcome_text, outcome_type, "
                " importance_weight, created_by) "
                "VALUES (:id, :cfg, 1, 'Understand recursion', 'knowledge', 3, :u)"
            ),
            {"id": outcome_id, "cfg": config_id, "u": actor_id},
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


async def test_finish_returns_status_and_enqueues_evaluation(
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


async def test_get_interview_config_with_existing_questions_does_not_500(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Regression: GET /teacher/interview-configs/{id} must not trigger async
    lazy-load on the ``questions`` relationship when rows already exist.

    Previously _populate_aggregates iterated ``data.questions`` on the ORM
    instance returned by db.get(); since the GET handler did not eager-load
    that relationship, sqlalchemy attempted sync lazy loading in an async
    session and raised MissingGreenlet, surfacing as a 500. The frontend
    rendered that as "Interview set not found".
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/interview-configs",
        json={
            "title": "Has Questions",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "supported_modes": "text",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    config_id = create_resp.json()["id"]

    # Insert a question directly so the relationship has unloaded rows on
    # the next GET (no in-process identity-map prefetch).
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "  id, interview_config_id, prompt_text, question_type,"
                "  review_status, ai_generated, position, created_by"
                ") VALUES ("
                "  :id, :cid, 'What is recursion?', 'conceptual',"
                "  'approved', false, 1, :uid"
                ")"
            ),
            {
                "id": uuid.uuid4(),
                "cid": uuid.UUID(config_id),
                "uid": seeded_users.admin_id,
            },
        )

    get_resp = await client.get(
        f"/api/v1/teacher/interview-configs/{config_id}",
        headers=_auth(admin_bearer),
    )
    assert get_resp.status_code == 200, get_resp.text
    payload = get_resp.json()
    assert payload["config"]["id"] == config_id
    assert len(payload["questions"]) == 1


async def test_publish_rejects_when_no_approved_question(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Regression: publish must reject configs with no approved questions.

    Without this gate, AI-generated drafts (all stored as review_status='pending'
    until reviewed) reach the student-facing fetch, which filters
    review_status='approved' and silently returns first_question=null — the
    learner UI then shows a blank/voice-agent-silent screen with no diagnostic.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/interview-configs",
        json={
            "title": "Pending Only",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "supported_modes": "text",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    config_id = create_resp.json()["id"]

    # Only a pending question exists — publish must fail.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "  id, interview_config_id, prompt_text, question_type,"
                "  review_status, ai_generated, position, created_by"
                ") VALUES ("
                "  :id, :cid, 'Pending question', 'conceptual',"
                "  'pending', true, 1, :uid"
                ")"
            ),
            {
                "id": uuid.uuid4(),
                "cid": uuid.UUID(config_id),
                "uid": seeded_users.admin_id,
            },
        )

    publish_resp = await client.post(
        f"/api/v1/teacher/interview-configs/{config_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert publish_resp.status_code == 400, publish_resp.text
    assert "interview_no_approved_questions" in publish_resp.text

    # Approving the question lifts the question block — but §4.3 also requires
    # at least one learning outcome before publish can succeed.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_questions SET review_status='approved' "
                "WHERE interview_config_id = :cid"
            ),
            {"cid": uuid.UUID(config_id)},
        )
    publish_no_outcome = await client.post(
        f"/api/v1/teacher/interview-configs/{config_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert publish_no_outcome.status_code == 400, publish_no_outcome.text
    assert "interview_no_outcomes" in publish_no_outcome.text

    # Adding an outcome lifts the final block.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes ("
                "  id, interview_config_id, position, outcome_text, outcome_type,"
                "  importance_weight, created_by"
                ") VALUES ("
                "  :id, :cid, 1, 'Understand recursion', 'knowledge', 3, :uid"
                ")"
            ),
            {
                "id": uuid.uuid4(),
                "cid": uuid.UUID(config_id),
                "uid": seeded_users.admin_id,
            },
        )
    publish_again = await client.post(
        f"/api/v1/teacher/interview-configs/{config_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert publish_again.status_code == 200, publish_again.text
    assert publish_again.json()["status"] == "published"


async def test_start_session_rejects_config_without_outcomes(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Regression (thesis §4.3): starting an interview whose config has zero
    learning outcomes is a guaranteed automatic fail with nothing to evaluate.

    Pre-fix, such sessions ran to completion then showed "you did not meet
    enough learning outcomes" because the per-outcome verdict gate had an empty
    set (total=0 → unconditional fail). The guard now blocks the start instead.
    """
    config_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs ("
                "  id, course_id, module_id, title, status, supported_modes,"
                "  created_by, published_at"
                ") VALUES ("
                "  :cid, :course, :module, 'No Outcomes Repro', 'published',"
                "  'text', :uid, NOW()"
                ")"
            ),
            {
                "cid": config_id,
                "course": scenario["course_id"],
                "module": scenario["module_id"],
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "  id, interview_config_id, prompt_text, question_type,"
                "  review_status, ai_generated, position, created_by"
                ") VALUES ("
                "  :qid, :cid, 'What is recursion?', 'conceptual',"
                "  'approved', true, 1, :uid"
                ")"
            ),
            {"qid": uuid.uuid4(), "cid": config_id, "uid": seeded_users.admin_id},
        )

    student_sid = await _seed_session(engine, seeded_users.student_id)
    try:
        student_bearer = create_access_token(
            user_id=seeded_users.student_id, session_id=student_sid
        )
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_bearer),
        )
        assert start_resp.status_code == 400, start_resp.text
        assert "interview_no_outcomes" in start_resp.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_start_session_self_heals_stale_empty_session(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Regression: a stale in_progress session created before any question
    was approved must NOT block the learner forever once questions reach
    review_status='approved'.

    Repro:
      1. Insert a published config with only pending questions.
      2. Insert an in_progress session for the student with zero attached
         interview_session_questions (mirrors the pre-fix race where
         start_session ran while _first_published_question returned None).
      3. Approve the question.
      4. POST /sessions — must return the same session id (idempotent) AND
         the first_question payload (self-healed attachment).
    """
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    session_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs ("
                "  id, course_id, module_id, title, status, supported_modes,"
                "  lock_quiz_ef_until_pass, created_by, published_at"
                ") VALUES ("
                "  :cid, :course, :module, 'Stale Session Repro', 'published',"
                "  'text', false, :uid, NOW()"
                ")"
            ),
            {
                "cid": config_id,
                "course": scenario["course_id"],
                "module": scenario["module_id"],
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "  id, interview_config_id, prompt_text, question_type,"
                "  review_status, ai_generated, position, created_by"
                ") VALUES ("
                "  :qid, :cid, 'What is recursion?', 'conceptual',"
                "  'pending', true, 1, :uid"
                ")"
            ),
            {
                "qid": question_id,
                "cid": config_id,
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes ("
                "  id, interview_config_id, position, outcome_text, outcome_type,"
                "  importance_weight, created_by"
                ") VALUES (:oid, :cid, 1, 'Understand recursion', 'knowledge', 3, :uid)"
            ),
            {"oid": uuid.uuid4(), "cid": config_id, "uid": seeded_users.admin_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions ("
                "  id, interview_config_id, student_id, attempt_number,"
                "  status, input_mode, started_at"
                ") VALUES ("
                "  :sid, :cid, :uid, 1, 'in_progress', 'text', NOW()"
                ")"
            ),
            {
                "sid": session_id,
                "cid": config_id,
                "uid": seeded_users.student_id,
            },
        )

    # Teacher approves the question now.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_questions SET review_status='approved' "
                "WHERE id = :qid"
            ),
            {"qid": question_id},
        )

    # Mint a student session + bearer.
    student_sid = await _seed_session(engine, seeded_users.student_id)
    try:
        student_bearer = create_access_token(
            user_id=seeded_users.student_id, session_id=student_sid
        )
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_bearer),
        )
        assert start_resp.status_code == 201, start_resp.text
        body = start_resp.json()
        # Same session reused (idempotent) AND self-healed with the question.
        assert body["session_id"] == str(session_id)
        assert body["first_question"] is not None
        assert body["first_question"]["id"] == str(question_id)
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_start_session_upgrades_input_mode_when_config_allows(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Regression: a stale text session must be upgradeable to voice when
    the config is hybrid.

    Pre-fix, ``start_session`` short-circuited on any in_progress session
    without checking whether the caller asked for a different input_mode.
    A student whose first POST landed as ``text`` then got HTTP 409
    "session is not a voice interview" from ``/realtime-token`` because
    the session row stayed at ``input_mode='text'`` forever.
    """
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    session_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs ("
                "  id, course_id, module_id, title, status, supported_modes,"
                "  lock_quiz_ef_until_pass, created_by, published_at"
                ") VALUES ("
                "  :cid, :course, :module, 'Mode Upgrade Repro', 'published',"
                "  'hybrid', false, :uid, NOW()"
                ")"
            ),
            {
                "cid": config_id,
                "course": scenario["course_id"],
                "module": scenario["module_id"],
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "  id, interview_config_id, prompt_text, question_type,"
                "  review_status, ai_generated, position, created_by"
                ") VALUES ("
                "  :qid, :cid, 'What is recursion?', 'conceptual',"
                "  'approved', false, 1, :uid"
                ")"
            ),
            {
                "qid": question_id,
                "cid": config_id,
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes ("
                "  id, interview_config_id, position, outcome_text, outcome_type,"
                "  importance_weight, created_by"
                ") VALUES (:oid, :cid, 1, 'Understand recursion', 'knowledge', 3, :uid)"
            ),
            {"oid": uuid.uuid4(), "cid": config_id, "uid": seeded_users.admin_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions ("
                "  id, interview_config_id, student_id, attempt_number,"
                "  status, input_mode, started_at"
                ") VALUES ("
                "  :sid, :cid, :uid, 1, 'in_progress', 'text', NOW()"
                ")"
            ),
            {
                "sid": session_id,
                "cid": config_id,
                "uid": seeded_users.student_id,
            },
        )

    student_sid = await _seed_session(engine, seeded_users.student_id)
    try:
        student_bearer = create_access_token(
            user_id=seeded_users.student_id, session_id=student_sid
        )
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "voice"},
            headers=_auth(student_bearer),
        )
        assert start_resp.status_code == 201, start_resp.text
        assert start_resp.json()["session_id"] == str(session_id)

        # Verify the upgrade persisted on the session row.
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT input_mode FROM interview_sessions WHERE id = :sid"),
                    {"sid": session_id},
                )
            ).first()
        assert row is not None and row[0] == "voice"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_start_session_does_not_upgrade_when_config_text_only(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Regression: a text-only config must NOT silently flip a session to voice.

    Defence-in-depth: the realtime-token endpoint also gates on input_mode,
    but the upgrade helper itself should refuse to grant a mode the config
    forbids.
    """
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    session_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs ("
                "  id, course_id, module_id, title, status, supported_modes,"
                "  lock_quiz_ef_until_pass, created_by, published_at"
                ") VALUES ("
                "  :cid, :course, :module, 'Text Only Repro', 'published',"
                "  'text', false, :uid, NOW()"
                ")"
            ),
            {
                "cid": config_id,
                "course": scenario["course_id"],
                "module": scenario["module_id"],
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "  id, interview_config_id, prompt_text, question_type,"
                "  review_status, ai_generated, position, created_by"
                ") VALUES ("
                "  :qid, :cid, 'Explain Big-O.', 'conceptual',"
                "  'approved', false, 1, :uid"
                ")"
            ),
            {
                "qid": question_id,
                "cid": config_id,
                "uid": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes ("
                "  id, interview_config_id, position, outcome_text, outcome_type,"
                "  importance_weight, created_by"
                ") VALUES (:oid, :cid, 1, 'Understand recursion', 'knowledge', 3, :uid)"
            ),
            {"oid": uuid.uuid4(), "cid": config_id, "uid": seeded_users.admin_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions ("
                "  id, interview_config_id, student_id, attempt_number,"
                "  status, input_mode, started_at"
                ") VALUES ("
                "  :sid, :cid, :uid, 1, 'in_progress', 'text', NOW()"
                ")"
            ),
            {
                "sid": session_id,
                "cid": config_id,
                "uid": seeded_users.student_id,
            },
        )

    student_sid = await _seed_session(engine, seeded_users.student_id)
    try:
        student_bearer = create_access_token(
            user_id=seeded_users.student_id, session_id=student_sid
        )
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "voice"},
            headers=_auth(student_bearer),
        )
        # Idempotency still returns the existing session, but mode stays text.
        assert start_resp.status_code == 201, start_resp.text
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT input_mode FROM interview_sessions WHERE id = :sid"),
                    {"sid": session_id},
                )
            ).first()
        assert row is not None and row[0] == "text"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_respond_surfaces_unhandled_error_with_class_and_message(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    monkeypatch: object,
) -> None:
    """Regression: when something inside ``take_session_step`` raises an
    unexpected exception, the route must NOT return a bare ``Internal
    Server Error`` body. Returning the exception class + message lets ops
    diagnose the failure from the user's screenshot alone.
    """
    from abridgeai.features.interviews.services import taking as taking_service  # noqa: PLC0415

    async def _explode(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("boom: simulated unhandled error")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        taking_service, "take_session_step", _explode
    )

    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    try:
        student_token = create_access_token(
            user_id=seeded_users.student_id, session_id=student_sid
        )
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text
        body = start_resp.json()
        session_id = body["session_id"]
        first_question = body["first_question"]
        assert first_question is not None

        respond_resp = await client.post(
            f"/api/v1/interview-sessions/{session_id}/respond",
            json={
                "session_id": session_id,
                "session_question_id": first_question["id"],
                "answer_text": "an answer",
            },
            headers=_auth(student_token),
        )
        assert respond_resp.status_code == 500, respond_resp.text
        detail = respond_resp.json()["detail"]
        assert detail["error"] == "internal_error"
        assert detail["error_class"] == "RuntimeError"
        assert "boom" in detail["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_respond_succeeds_when_followup_stage_raises(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    monkeypatch: object,
) -> None:
    """Regression: a failing follow-up LLM call must not 500 the /respond
    endpoint.

    Pre-fix, ``maybe_generate_followup`` let any exception (missing
    LLM_API_KEY → ConfigError, provider down, etc.) propagate. The router
    only mapped ``AppError`` → 400, so anything else surfaced as HTTP 500
    and the student's answer was lost because the transaction rolled back
    before the answer-record commit. The fix wraps the LLM call inside
    ``maybe_generate_followup`` so the stage falls back to ``None`` and
    the answer commits normally.

    We patch ``LLMGateway.generate_json`` (NOT the stage itself) so the
    test exercises the real best-effort try/except added by the fix.
    """
    from abridgeai.ai.llm import gateway as gateway_module  # noqa: PLC0415

    async def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated gateway failure")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        gateway_module.LLMGateway, "generate_json", _boom
    )

    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    try:
        student_token = create_access_token(
            user_id=seeded_users.student_id, session_id=student_sid
        )

        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text
        body = start_resp.json()
        session_id = body["session_id"]
        first_question = body["first_question"]
        assert first_question is not None

        respond_resp = await client.post(
            f"/api/v1/interview-sessions/{session_id}/respond",
            json={
                "session_id": session_id,
                "session_question_id": first_question["id"],
                "answer_text": "Recursion is when a function calls itself.",
            },
            headers=_auth(student_token),
        )
        # Pre-fix this was 500; post-fix the answer commits and the route
        # responds normally (is_finished=True because the seeded config has
        # only one approved question).
        assert respond_resp.status_code == 200, respond_resp.text
        payload = respond_resp.json()
        assert payload["ai_followup_text"] is None
        assert payload["is_finished"] is True

        # Verify the answer DID land in the DB despite the follow-up failure.
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) FROM interview_session_messages "
                        "WHERE session_id = :sid AND role = 'user'"
                    ),
                    {"sid": uuid.UUID(session_id)},
                )
            ).scalar_one()
        assert int(row) == 1
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


# ---------------------------------------------------------------------------
# Result visibility — thesis p53/p77 student/teacher split
# ---------------------------------------------------------------------------


async def test_my_sessions_carries_title_and_no_leakage(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Thesis p53: student history list carries the interview title + binary
    verdict ONLY — never transcript / score / rubric / per-outcome data.

    Durable leakage guard: fails if a future change widens the student DTO with
    any teacher-only field.
    """
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    try:
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text

        resp = await client.get(
            "/api/v1/me/interview-sessions", headers=_auth(student_token)
        )
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert len(rows) >= 1
        assert "interview_title" in rows[0]
        serialized = json.dumps(rows).lower()
        for forbidden in (
            "transcript",
            "total_score",
            "rubric",
            "internal_summary",
            "hidden_reasoning",
            "verdict_met",
            "evidence",
        ):
            assert forbidden not in serialized, f"student list leaks {forbidden!r}"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_teacher_lists_config_sessions_and_blocks_non_owner(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Teacher (config owner) lists attempts; a student (no authoring access)
    is denied — the attempts list is a teacher-only surface (thesis p77)."""
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    try:
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text

        # Owner (admin) sees the attempt.
        owner = await client.get(
            f"/api/v1/teacher/interview-configs/{config_id}/sessions",
            headers=_auth(admin_bearer),
        )
        assert owner.status_code == 200, owner.text
        rows = owner.json()
        assert len(rows) >= 1
        assert rows[0]["student_id"] == str(seeded_users.student_id)

        # A student has no authoring access — denied.
        denied = await client.get(
            f"/api/v1/teacher/interview-configs/{config_id}/sessions",
            headers=_auth(student_token),
        )
        assert denied.status_code in (403, 404), denied.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_teacher_transcript_returns_turns_and_blocks_student(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Teacher fetches the full Q&A transcript (thesis p77); the student who
    owns the session is NOT a teacher and cannot hit the authoring endpoint."""
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    try:
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text
        body = start_resp.json()
        session_id = body["session_id"]
        first_q = body["first_question"]
        assert first_q is not None

        respond = await client.post(
            f"/api/v1/interview-sessions/{session_id}/respond",
            json={
                "session_id": session_id,
                "session_question_id": first_q["id"],
                "answer_text": "Recursion calls itself with a smaller input until a base case.",
            },
            headers=_auth(student_token),
        )
        assert respond.status_code == 200, respond.text

        # Teacher (owner) gets the transcript with the student's answer.
        transcript = await client.get(
            f"/api/v1/teacher/interview-sessions/{session_id}/transcript",
            headers=_auth(admin_bearer),
        )
        assert transcript.status_code == 200, transcript.text
        turns = transcript.json()["turns"]
        user_turns = [tn for tn in turns if tn["role"] == "user"]
        assert any("recursion" in (tn["content_text"] or "").lower() for tn in user_turns)

        # The student cannot reach the teacher transcript endpoint.
        denied = await client.get(
            f"/api/v1/teacher/interview-sessions/{session_id}/transcript",
            headers=_auth(student_token),
        )
        assert denied.status_code in (403, 404), denied.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_teacher_session_detail_endpoint_returns_200(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """GET /teacher/interview-sessions/{id} — course-scoped teacher access.

    The frontend gap-report page was calling the student-owner-only
    GET /interview-sessions/{id} (403 for a teacher) by mistake; this is
    the teacher-scoped sibling it should have been calling all along.
    """
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    try:
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text
        session_id = start_resp.json()["session_id"]

        teacher_resp = await client.get(
            f"/api/v1/teacher/interview-sessions/{session_id}",
            headers=_auth(admin_bearer),
        )
        assert teacher_resp.status_code == 200, teacher_resp.text
        body = teacher_resp.json()
        assert body["session_id"] == session_id
        assert body["interview_config_id"] == str(config_id)

        # The student-owner-only endpoint remains off-limits to the teacher.
        denied = await client.get(
            f"/api/v1/interview-sessions/{session_id}",
            headers=_auth(admin_bearer),
        )
        assert denied.status_code == 403, denied.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def test_teacher_gap_report_endpoint_returns_200(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Regression: GET /teacher/interview-sessions/{id}/gap-report 500'd in
    prod because ``GapReportAuthoringRead.model_validate(report)`` was called
    directly on the ORM row, which has no ``generated_at`` attribute (it's
    ``created_at`` via TimestampMixin) — a required field on the schema.
    """
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
    assert start_resp.status_code == 201, start_resp.text
    session_id = start_resp.json()["session_id"]
    gap_report_id = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO gap_reports ("
                    "id, student_id, course_id, module_id, "
                    "source_interview_session_id, student_summary, teacher_summary, "
                    "report_json"
                    ") VALUES ("
                    ":id, :student_id, :course_id, :module_id, "
                    ":session_id, :student_summary, :teacher_summary, "
                    "CAST(:report_json AS jsonb)"
                    ")"
                ),
                {
                    "id": gap_report_id,
                    "student_id": seeded_users.student_id,
                    "course_id": scenario["course_id"],
                    "module_id": scenario["module_id"],
                    "session_id": session_id,
                    "student_summary": "Strong on theory, weak on application.",
                    "teacher_summary": "Push more hands-on practice.",
                    "report_json": json.dumps(
                        {
                            "study_plan": [
                                {
                                    "topic": "Recursion",
                                    "suggested_lesson_id": None,
                                    "suggested_resource_ids": [],
                                }
                            ],
                            "rubric_aggregated": {"technical_accuracy": 3.2},
                        }
                    ),
                },
            )

        resp = await client.get(
            f"/api/v1/teacher/interview-sessions/{session_id}/gap-report",
            headers=_auth(admin_bearer),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == str(gap_report_id)
        assert body["discrepancy_summary"] == "Strong on theory, weak on application."
        assert body["study_plan"][0]["topic"] == "Recursion"
        assert body["per_criterion_breakdown"] == {"technical_accuracy": 3.2}
        assert body["teacher_summary"] == "Push more hands-on practice."
        assert "generated_at" in body
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM gap_reports WHERE id = :id"), {"id": gap_report_id}
            )
            await conn.execute(
                text("DELETE FROM interview_sessions WHERE id = :id"), {"id": session_id}
            )


async def test_teacher_course_and_student_session_list_endpoints(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Regression coverage for the new course-wide + per-student
    interview-sessions endpoints (student-dashboard brainstorm, 2026-07-11).

    Neither endpoint existed before — teachers could only list sessions
    scoped to a single interview config, never across a whole course.
    """
    config_id = await _seed_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        actor_id=seeded_users.admin_id,
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    try:
        start_resp = await client.post(
            f"/api/v1/interview-configs/{config_id}/sessions",
            json={"input_mode": "text"},
            headers=_auth(student_token),
        )
        assert start_resp.status_code == 201, start_resp.text
        session_id = start_resp.json()["session_id"]

        course_resp = await client.get(
            f"/api/v1/teacher/courses/{scenario['course_id']}/interview-sessions",
            headers=_auth(admin_bearer),
        )
        assert course_resp.status_code == 200, course_resp.text
        course_rows = course_resp.json()
        assert len(course_rows) == 1
        row = course_rows[0]
        assert row["session_id"] == session_id
        assert row["interview_config_id"] == str(config_id)
        assert row["interview_config_title"]
        assert row["student_id"] == str(seeded_users.student_id)
        assert row["student_name"] is not None

        student_resp = await client.get(
            f"/api/v1/teacher/courses/{scenario['course_id']}/students/"
            f"{seeded_users.student_id}/interview-sessions",
            headers=_auth(admin_bearer),
        )
        assert student_resp.status_code == 200, student_resp.text
        student_rows = student_resp.json()
        assert len(student_rows) == 1
        assert student_rows[0]["session_id"] == session_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )
