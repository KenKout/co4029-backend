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
        # BR gate: quiz taking is a course-item flow — the student must be
        # enrolled for the learner reads to resolve (can_view_course_content).
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (course_id, student_id, status, source) "
                "VALUES (:c, :s, 'active', 'manager_bulk') "
                "ON CONFLICT (course_id, student_id) DO NOTHING"
            ),
            {"c": seeded_users.course_id, "s": seeded_users.student_id},
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
    assert "/teacher/courses/{course_id}/quiz-attempts" in paths
    assert "/teacher/courses/{course_id}/students/{student_id}/quiz-attempts" in paths


def test_learner_router_metadata() -> None:
    assert learner_router.prefix == ""
    paths = {route.path for route in learner_router.routes}  # type: ignore[attr-defined]
    assert "/quizzes/{quiz_id}" in paths
    assert "/quizzes/{quiz_id}/attempts" in paths
    assert "/attempts/{attempt_id}/answers" in paths
    assert "/attempts/{attempt_id}/submit" in paths
    assert "/attempts/{attempt_id}" in paths
    assert "/attempts/{attempt_id}/integrity-events" in paths
    assert "/me/quizzes/{quiz_id}/attempts" in paths


async def test_teacher_quiz_attempts_endpoints(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Regression coverage for the new course-wide + per-student
    quiz-attempts endpoints (student-dashboard brainstorm, 2026-07-11).

    Neither endpoint existed before — there was no teacher-facing way to
    list quiz attempts at all, only aggregate stats.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={
            "module_id": str(scenario["module_id"]),
            "title": "Attempts Quiz",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = create_resp.json()["id"]

    attempt_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts "
                "(id, quiz_id, student_id, attempt_number, status, score_percent, passed) "
                "VALUES (:id, :quiz, :student, 1, 'graded', 88.50, TRUE)"
            ),
            {"id": attempt_id, "quiz": quiz_id, "student": seeded_users.student_id},
        )

    course_resp = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quiz-attempts",
        headers=_auth(admin_bearer),
    )
    assert course_resp.status_code == 200, course_resp.text
    course_rows = course_resp.json()
    assert len(course_rows) == 1
    row = course_rows[0]
    assert row["id"] == str(attempt_id)
    assert row["quiz_id"] == quiz_id
    assert row["quiz_title"] == "Attempts Quiz"
    assert row["student_id"] == str(seeded_users.student_id)
    assert row["student_name"] is not None
    assert float(row["score_percent"]) == 88.50
    assert row["passed"] is True

    student_resp = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_id']}/students/"
        f"{seeded_users.student_id}/quiz-attempts",
        headers=_auth(admin_bearer),
    )
    assert student_resp.status_code == 200, student_resp.text
    student_rows = student_resp.json()
    assert len(student_rows) == 1
    assert student_rows[0]["id"] == str(attempt_id)


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
            "title": "Generated Quiz",
            "question_count": 5,
            "question_types": ["multiple_choice"],
            "generation_mode": "topic",
            "focus_topics": [],
            "source_lesson_ids": [],
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

    # Seed the approved question BEFORE publishing — the publish gate now
    # requires at least one approved question, so an empty quiz can't be
    # published.
    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status, "
                " expected_response_time_ms) "
                "VALUES (:id, :qz, 1, 'multiple_choice', 'What is 2+2?', 'approved', 30000)"
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

    pub_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert pub_resp.status_code == 200, pub_resp.text

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
    progress_body = attempt_resp.json()
    assert progress_body["attempt_id"]
    take_body = progress_body["take"]
    assert take_body["questions"]
    serialized = json.dumps(progress_body)
    assert "is_correct" not in serialized
    for question in take_body["questions"]:
        for option in question["options"]:
            assert "is_correct" not in option

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


# ---------------------------------------------------------------------------
# Source-grep guard (FIX-SEC-1 perimeter)
# ---------------------------------------------------------------------------


async def test_cross_org_quiz_not_readable_or_takable(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A published quiz from another organization is not reachable by id.

    Regression: ``GET /quizzes/{id}`` and ``POST /quizzes/{id}/attempts``
    used to trust ``status='published'`` alone (``del current_user``), so a
    student of org A could fetch org B's quiz — question count, and via the
    attempt flow the questions + options themselves — by UUID. Both entry
    points must now resolve to 404 for a caller who is neither a member of
    the owning org nor a course manager (no existence leak).
    """
    org2 = uuid.uuid4()
    course2 = uuid.uuid4()
    module2 = uuid.uuid4()
    quiz2 = uuid.uuid4()
    question2 = uuid.uuid4()
    suffix = org2.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {"id": org2, "slug": f"xorg-{suffix}", "name": "X Org"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'X Course', 'published')"
            ),
            {
                "id": course2,
                "org": org2,
                "owner": seeded_users.teacher_id,
                "slug": f"xcourse-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'X Module', 1, 'published')"
            ),
            {"m": module2, "c": course2},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:q, :c, :m, 'X Quiz', 'published')"
            ),
            {"q": quiz2, "c": course2, "m": module2},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions (id, quiz_id, position, question_type, "
                "prompt_text, review_status, expected_response_time_ms) "
                "VALUES (:q, :qz, 1, 'multiple_choice', 'X secret?', 'approved', 30000)"
            ),
            {"q": question2, "qz": quiz2},
        )

    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    try:
        quiz_resp = await client.get(
            f"/api/v1/quizzes/{quiz2}",
            headers=_auth(student_token),
        )
        assert quiz_resp.status_code == 404, quiz_resp.text
        attempt_resp = await client.post(
            f"/api/v1/quizzes/{quiz2}/attempts",
            json={"quiz_id": str(quiz2)},
            headers=_auth(student_token),
        )
        assert attempt_resp.status_code == 404, attempt_resp.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM quiz_question_options WHERE question_id = :q"),
                {"q": question2},
            )
            await conn.execute(text("DELETE FROM quiz_questions WHERE id = :q"), {"q": question2})
            await conn.execute(
                text("DELETE FROM quiz_source_lessons WHERE quiz_id = :q"),
                {"q": quiz2},
            )
            await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz2})
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module2})
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course2})
            await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org2})
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": student_sid},
            )


async def _seed_published_quiz_with_question(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    *,
    title: str,
) -> tuple[uuid.UUID, uuid.UUID, dict[str, uuid.UUID]]:
    """Create + publish a quiz with one MCQ question (4 options).

    Returns ``(quiz_id, question_id, option_ids_by_key)``. Options carry a
    real ``expected_response_time_ms`` so the SM-2 review path doesn't skip.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": title},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = uuid.UUID(create_resp.json()["id"])

    # Seed the approved question + options BEFORE publishing. The publish gate
    # requires at least one approved question (and a t_exp on each), so an
    # empty quiz can no longer be published — insert content first, then flip
    # to published.
    question_id = uuid.uuid4()
    option_ids: dict[str, uuid.UUID] = {}
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, "
                " review_status, expected_response_time_ms) "
                "VALUES (:id, :qz, 1, 'multiple_choice', 'What is 2+2?', "
                " 'approved', 30000)"
            ),
            {"id": question_id, "qz": quiz_id},
        )
        for option_key, option_text, is_correct, position in (
            ("A", "3", False, 1),
            ("B", "4", True, 2),
            ("C", "5", False, 3),
            ("D", "6", False, 4),
        ):
            oid = uuid.uuid4()
            option_ids[option_key] = oid
            await conn.execute(
                text(
                    "INSERT INTO quiz_question_options "
                    "(id, question_id, option_key, option_text, is_correct, position) "
                    "VALUES (:id, :q, :k, :t, :c, :p)"
                ),
                {
                    "id": oid,
                    "q": question_id,
                    "k": option_key,
                    "t": option_text,
                    "c": is_correct,
                    "p": position,
                },
            )

    pub_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert pub_resp.status_code == 200, pub_resp.text

    return quiz_id, question_id, option_ids


async def test_answer_upsert_allows_editing_without_conflict(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Re-saving a question after changing the answer must NOT 409.

    Regression: the write path used to blindly INSERT, so a second save
    for the same (attempt, question) violated
    uq_quiz_attempt_answers_question. The upsert edits in place instead.
    """
    quiz_id, question_id, option_ids = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Upsert Quiz"
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201, attempt_resp.text
    attempt_id = attempt_resp.json()["attempt_id"]

    # First save — wrong answer (C).
    first = await client.post(
        f"/api/v1/attempts/{attempt_id}/answers",
        json={
            "question_id": str(question_id),
            "selected_option_id": str(option_ids["C"]),
            "t_actual_ms": 5000,
        },
        headers=_auth(student_token),
    )
    assert first.status_code == 201, first.text

    # Second save — changed to correct answer (B). Must succeed (upsert),
    # not 409.
    second = await client.post(
        f"/api/v1/attempts/{attempt_id}/answers",
        json={
            "question_id": str(question_id),
            "selected_option_id": str(option_ids["B"]),
            "t_actual_ms": 8000,
        },
        headers=_auth(student_token),
    )
    assert second.status_code == 201, second.text

    # DB holds exactly ONE answer row, reflecting the latest edit.
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT selected_option_id, t_actual_ms, is_correct "
                    "FROM quiz_attempt_answers WHERE attempt_id = :a"
                ),
                {"a": attempt_id},
            )
        ).all()
    assert len(rows) == 1
    assert rows[0][0] == option_ids["B"]
    assert rows[0][1] == 8000
    assert rows[0][2] is True

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_attempt_progress_returns_saved_answers_no_leak(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """GET /attempts/{id}/progress rehydrates saved answers, no correctness leak."""
    quiz_id, question_id, option_ids = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Progress Quiz"
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201, attempt_resp.text
    attempts_resp = await client.get(
        f"/api/v1/me/quizzes/{quiz_id}/attempts",
        headers=_auth(student_token),
    )
    attempt_id = attempts_resp.json()[0]["id"]

    await client.post(
        f"/api/v1/attempts/{attempt_id}/answers",
        json={
            "question_id": str(question_id),
            "selected_option_id": str(option_ids["B"]),
            "t_actual_ms": 4200,
            "hint_used": True,
        },
        headers=_auth(student_token),
    )

    progress_resp = await client.get(
        f"/api/v1/attempts/{attempt_id}/progress",
        headers=_auth(student_token),
    )
    assert progress_resp.status_code == 200, progress_resp.text
    body = progress_resp.json()
    assert body["attempt_id"] == attempt_id
    assert body["quiz_id"] == str(quiz_id)
    assert body["status"] == "in_progress"
    assert "started_at" in body
    assert len(body["answers"]) == 1
    ans = body["answers"][0]
    assert ans["question_id"] == str(question_id)
    assert ans["selected_option_id"] == str(option_ids["B"])
    assert ans["t_actual_ms"] == 4200
    assert ans["hint_used"] is True
    # No-leak: correctness / points must not appear anywhere in the payload.
    serialized = json.dumps(body)
    assert "is_correct" not in serialized
    assert "points_awarded" not in serialized

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_attempt_progress_404_for_other_student(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A student cannot read another student's in-progress attempt."""
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Progress Owner Quiz"
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201
    attempts_resp = await client.get(
        f"/api/v1/me/quizzes/{quiz_id}/attempts",
        headers=_auth(student_token),
    )
    attempt_id = attempts_resp.json()[0]["id"]

    # Admin (a different user) tries to read the student's attempt → 404.
    other_resp = await client.get(
        f"/api/v1/attempts/{attempt_id}/progress",
        headers=_auth(admin_bearer),
    )
    assert other_resp.status_code == 404, other_resp.text

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


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


async def test_quiz_integrity_events_recorded_for_in_progress_attempt(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Owner can POST integrity events for a live attempt; rows land in DB."""
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Integrity Quiz"
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201, attempt_resp.text
    attempt_id = attempt_resp.json()["attempt_id"]

    resp = await client.post(
        f"/api/v1/attempts/{attempt_id}/integrity-events",
        json={
            "events": [
                {"event_type": "tab_switch", "severity": "warning"},
                {"event_type": "focus_lost", "severity": "info"},
            ]
        },
        headers=_auth(student_token),
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["accepted"] == 2

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT event_type, severity, assessment_kind, student_id "
                    "FROM assessment_integrity_events "
                    "WHERE quiz_attempt_id = :a ORDER BY event_type"
                ),
                {"a": attempt_id},
            )
        ).all()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"focus_lost", "tab_switch"}
    assert all(r[2] == "quiz" for r in rows)
    assert all(str(r[3]) == str(seeded_users.student_id) for r in rows)

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM assessment_integrity_events WHERE quiz_attempt_id = :a"),
            {"a": attempt_id},
        )
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_quiz_integrity_events_404_for_other_student(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A non-owner cannot post integrity events against someone's attempt."""
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Integrity Owner Quiz"
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201
    attempt_id = attempt_resp.json()["attempt_id"]

    # Admin (different user) attempts to post → 404 (no existence leak).
    other = await client.post(
        f"/api/v1/attempts/{attempt_id}/integrity-events",
        json={"events": [{"event_type": "tab_switch", "severity": "warning"}]},
        headers=_auth(admin_bearer),
    )
    assert other.status_code == 404, other.text

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_quiz_integrity_events_dropped_when_not_in_progress(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Events for a submitted/graded attempt are silently dropped (accepted=0)."""
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Integrity Closed Quiz"
    )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert attempt_resp.status_code == 201
    attempt_id = attempt_resp.json()["attempt_id"]

    # Force the attempt out of in_progress.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quiz_attempts SET status = 'graded' WHERE id = :a"),
            {"a": attempt_id},
        )

    resp = await client.post(
        f"/api/v1/attempts/{attempt_id}/integrity-events",
        json={"events": [{"event_type": "tab_switch", "severity": "warning"}]},
        headers=_auth(student_token),
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["accepted"] == 0

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text("SELECT COUNT(*) FROM assessment_integrity_events WHERE quiz_attempt_id = :a"),
                {"a": attempt_id},
            )
        ).scalar_one()
    assert rows == 0

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_start_attempt_blocked_before_available_from(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A quiz with available_from in the future rejects attempt start with 409."""
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Not Yet Open Quiz"
    )
    future = datetime.now(UTC) + timedelta(days=1)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET available_from = :ts WHERE id = :id"),
            {"ts": future, "id": quiz_id},
        )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["reason"] == "quiz_not_yet_open"

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_start_attempt_blocked_after_available_until(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A quiz whose available_until has passed rejects attempt start with 409."""
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Closed Quiz"
    )
    past = datetime.now(UTC) - timedelta(days=1)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET available_until = :ts WHERE id = :id"),
            {"ts": past, "id": quiz_id},
        )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["reason"] == "quiz_closed"

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_start_attempt_allowed_inside_window(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A quiz whose now sits inside [available_from, available_until] starts fine.

    ``due_at`` in the past must NOT block (it's a soft deadline).
    """
    quiz_id, _question_id, _opts = await _seed_published_quiz_with_question(
        client, admin_bearer, engine, scenario, title="Open Window Quiz"
    )
    now = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE quizzes SET available_from = :past, "
                "available_until = :future, due_at = :due WHERE id = :id"
            ),
            {
                "past": now - timedelta(hours=1),
                "future": now + timedelta(hours=1),
                "due": now - timedelta(minutes=5),  # past soft deadline — must not block
                "id": quiz_id,
            },
        )
    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)

    resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_token),
    )
    assert resp.status_code == 201, resp.text

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_update_quiz_persists_schedule_window(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """PATCH /teacher/quizzes/{id} accepts ISO window strings on a DRAFT quiz.

    A published quiz is frozen (see
    test_update_published_quiz_is_rejected), so schedule edits are only
    valid while the quiz is still a draft.
    """
    # Create a DRAFT quiz (do not publish) so the edit is allowed.
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Schedule PATCH Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = uuid.UUID(create_resp.json()["id"])

    open_ts = "2026-08-01T09:00:00+00:00"
    close_ts = "2026-08-08T09:00:00+00:00"
    due_ts = "2026-08-07T23:59:00+00:00"
    resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={
            "available_from": open_ts,
            "available_until": close_ts,
            "due_at": due_ts,
        },
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available_from"].startswith("2026-08-01T09:00:00")
    assert body["available_until"].startswith("2026-08-08T09:00:00")
    assert body["due_at"].startswith("2026-08-07T23:59:00")

    # Clearing a window field with null must persist NULL.
    clear_resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={"available_until": None},
        headers=_auth(admin_bearer),
    )
    assert clear_resp.status_code == 200, clear_resp.text
    assert clear_resp.json()["available_until"] is None


async def test_update_quiz_browser_security_boolean_coerced(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """PATCH accepts the client's boolean ``browser_security`` toggle.

    The column is a string enum ('none' | 'securewindow') guarded by
    ``ck_quizzes_browser_security``. The Settings tab models the field as a
    boolean toggle and sends a raw bool; the service must map it to the enum
    so it can't reach the column and trip the CHECK constraint (which
    previously surfaced as a 500).
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Browser-Security PATCH Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = uuid.UUID(create_resp.json()["id"])

    # true -> 'securewindow'
    on_resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={"browser_security": True},
        headers=_auth(admin_bearer),
    )
    assert on_resp.status_code == 200, on_resp.text
    async with engine.connect() as conn:
        stored = await conn.scalar(
            text("SELECT browser_security FROM quizzes WHERE id = :id"),
            {"id": str(quiz_id)},
        )
    assert stored == "securewindow"

    # false -> 'none'
    off_resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={"browser_security": False},
        headers=_auth(admin_bearer),
    )
    assert off_resp.status_code == 200, off_resp.text
    async with engine.connect() as conn:
        stored = await conn.scalar(
            text("SELECT browser_security FROM quizzes WHERE id = :id"),
            {"id": str(quiz_id)},
        )
    assert stored == "none"

    # A bogus string is a clean 400, not a 500 from the CHECK constraint.
    bad_resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={"browser_security": "lockdown"},
        headers=_auth(admin_bearer),
    )
    assert bad_resp.status_code == 400, bad_resp.text


async def test_published_quiz_allows_student_safe_settings(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """On a published quiz, student-safe settings stay editable.

    Title/description/schedule/reminders don't disrupt a student who is
    taking or has finished the quiz, so they may be PATCHed after publish.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Live Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = uuid.UUID(create_resp.json()["id"])
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET status = 'published' WHERE id = :id"),
            {"id": quiz_id},
        )

    resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={
            "title": "Renamed after publish",
            "description": "Deadline extended",
            "available_until": "2026-09-01T09:00:00+00:00",
            "reminders_enabled": True,
        },
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["title"] == "Renamed after publish"
    assert body["available_until"].startswith("2026-09-01T09:00:00")


async def test_published_quiz_rejects_frozen_settings(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """On a published quiz, scoring/timing/attempt settings are frozen.

    Changing these under a student who is mid-attempt (or already finished)
    would corrupt grading/presentation, so they must 409
    (quiz_published_setting_locked).
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Frozen Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = uuid.UUID(create_resp.json()["id"])
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET status = 'published' WHERE id = :id"),
            {"id": quiz_id},
        )

    # A frozen field alone → 409.
    resp = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={"passing_score_percent": 90},
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 409, resp.text
    assert "quiz_published_setting_locked" in resp.text

    # A frozen field mixed with a student-safe field is still rejected
    # wholesale (no partial application).
    mixed = await client.patch(
        f"/api/v1/teacher/quizzes/{quiz_id}",
        json={"title": "New title", "shuffle_questions": True},
        headers=_auth(admin_bearer),
    )
    assert mixed.status_code == 409, mixed.text
    assert "shuffle_questions" in mixed.text


# ---------------------------------------------------------------------------
# Quiz results analytics endpoint (T1.5 + T1.6)
# ---------------------------------------------------------------------------


def test_authoring_router_exposes_quiz_results_route() -> None:
    paths = {route.path for route in authoring_router.routes}  # type: ignore[attr-defined]
    assert "/teacher/quizzes/{quiz_id}/results" in paths


async def test_quiz_results_endpoint_assembles_payload(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """As the course teacher: 200 with summary/per_student/per_question.

    Two students with completed attempts → unique_students==2 and two
    per-student rows with the correct best/latest score distinction.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Results Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = create_resp.json()["id"]

    # One question so per_question is non-empty even with attempts.
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

    now = datetime.now(UTC)
    student_a = seeded_users.student_id
    student_b = seeded_users.hod_id
    # A: #1 60 (fail), #2 90 (pass) → best 90, latest 90
    # B: #1 100 (pass), #2 50 (fail) → best 100, latest 50
    attempts = [
        (student_a, 1, "graded", 60, False),
        (student_a, 2, "graded", 90, True),
        (student_b, 1, "graded", 100, True),
        (student_b, 2, "graded", 50, False),
    ]
    async with engine.begin() as conn:
        for sid, num, stat, score, passed in attempts:
            await conn.execute(
                text(
                    "INSERT INTO quiz_attempts "
                    "(id, quiz_id, student_id, attempt_number, status, "
                    " score_percent, passed, time_taken_seconds, started_at, submitted_at) "
                    "VALUES (:id, :q, :s, :n, :st, :sc, :p, 120, :start, :submit)"
                ),
                {
                    "id": uuid.uuid4(),
                    "q": quiz_id,
                    "s": sid,
                    "n": num,
                    "st": stat,
                    "sc": score,
                    "p": passed,
                    "start": now - timedelta(hours=2),
                    "submit": now - timedelta(hours=1),
                },
            )

    resp = await client.get(
        f"/api/v1/teacher/quizzes/{quiz_id}/results",
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quiz_id"] == quiz_id
    assert body["quiz_title"] == "Results Quiz"
    assert body["grading_method"] == "highest"
    assert body["passing_score_percent"] is not None
    assert "summary" in body
    assert "per_student" in body
    assert "per_question" in body
    assert body["summary"]["unique_students"] == 2
    assert body["summary"]["total_attempts"] == 4

    rows = {row["student_id"]: row for row in body["per_student"]}
    assert len(rows) == 2
    row_a = rows[str(student_a)]
    assert float(row_a["best_score_percent"]) == 90.0
    assert float(row_a["latest_score_percent"]) == 90.0
    assert row_a["attempts_count"] == 2
    assert row_a["passed"] is True
    row_b = rows[str(student_b)]
    assert float(row_b["best_score_percent"]) == 100.0
    assert float(row_b["latest_score_percent"]) == 50.0
    assert row_b["attempts_count"] == 2

    # per_question lists the one question.
    assert len(body["per_question"]) == 1
    assert body["per_question"][0]["question_id"] == str(question_id)


async def test_quiz_results_endpoint_403_for_unrelated_user(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A user without course.update on the course is rejected with 403."""
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Forbidden Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = create_resp.json()["id"]

    student_sid = await _seed_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=student_sid)
    resp = await client.get(
        f"/api/v1/teacher/quizzes/{quiz_id}/results",
        headers=_auth(student_token),
    )
    assert resp.status_code == 403, resp.text

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": student_sid})


async def test_quiz_results_endpoint_zero_attempts(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """Zero-attempts quiz → 200, zeroed summary, empty per_student,
    per_question still lists the questions."""
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={"module_id": str(scenario["module_id"]), "title": "Empty Results Quiz"},
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = create_resp.json()["id"]

    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status) "
                "VALUES (:id, :qz, 1, 'short_answer', 'Explain X.', 'approved')"
            ),
            {"id": question_id, "qz": quiz_id},
        )

    resp = await client.get(
        f"/api/v1/teacher/quizzes/{quiz_id}/results",
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["total_attempts"] == 0
    assert body["summary"]["unique_students"] == 0
    assert body["per_student"] == []
    assert len(body["per_question"]) == 1
    assert body["per_question"][0]["question_id"] == str(question_id)
