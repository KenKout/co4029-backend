"""Integration tests for the learner interview-progress endpoint (course-learn).

``GET /api/v1/courses/{course_id}/interview-progress`` returns per-interview
completion state for the calling student.

Interviews were graded per attempt long before this endpoint existed
(``interview_sessions.pass_verdict``, written by the ARQ evaluator), but the
verdict never reached the curriculum, so an interview item stayed pending
forever and a module holding one could never auto-collapse.

Rule (user decision, 2026-08-06): completed <=> at least one NON-PRACTICE
attempt has ``pass_verdict = TRUE``. Deliberately STRICTER than the quiz rule,
which also completes on "failed with every attempt consumed" -- here the tag is
meant to read as *passed*, so failing every attempt keeps the item pending.

Fixture scaffolding is copied per-file from test_quiz_learner_progress.py, as
the other integration tests in this directory do.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token
from abridgeai.features.interviews.routers import learner_router, learner_sessions_router


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
async def app(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.include_router(learner_sessions_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {"id": session_id, "uid": user_id, "h": generate_token(), "exp": expires_at},
        )
    return session_id


async def _seed_interview_item(
    conn: AsyncConnection,
    *,
    config_id: uuid.UUID,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    position: int,
    title: str,
) -> None:
    """One interview config + its module_items ordering row.

    ``course_id`` is NOT NULL on interview_configs (it denormalises the module's
    course), so it must be passed explicitly — unlike quizzes, where the test
    helper could get away with module_id alone.
    """
    await conn.execute(
        text(
            "INSERT INTO interview_configs "
            "(id, course_id, module_id, title, supported_modes, status) "
            "VALUES (:id, :c, :m, :t, 'hybrid', 'published')"
        ),
        {"id": config_id, "c": course_id, "m": module_id, "t": title},
    )
    await conn.execute(
        text(
            "INSERT INTO module_items (id, module_id, item_type, interview_config_id, position) "
            "VALUES (:id, :m, 'interview', :c, :pos)"
        ),
        {"id": uuid.uuid4(), "m": module_id, "c": config_id, "pos": position},
    )


async def _seed_attempt(
    conn: AsyncConnection,
    *,
    config_id: uuid.UUID,
    student_id: uuid.UUID,
    attempt_number: int,
    status: str = "completed",
    pass_verdict: bool | None = None,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO interview_sessions (id, interview_config_id, student_id, "
            "attempt_number, status, input_mode, pass_verdict) "
            "VALUES (:id, :c, :s, :n, :st, 'hybrid', :pv)"
        ),
        {
            "id": uuid.uuid4(),
            "c": config_id,
            "s": student_id,
            "n": attempt_number,
            "st": status,
            "pv": pass_verdict,
        },
    )


async def _enroll(conn: AsyncConnection, *, course_id: uuid.UUID, student_id: uuid.UUID) -> None:
    """Enrol the student, or the read 404s.

    ``interview-progress`` is a course-item read gated by
    ``can_view_course_content``, which requires an active/completed enrolment
    (or course-management rights). Same requirement as quiz-progress.
    """
    await conn.execute(
        text(
            "INSERT INTO course_enrollments (course_id, student_id, status, source) "
            "VALUES (:c, :s, 'active', 'manager_bulk') "
            "ON CONFLICT (course_id, student_id) DO NOTHING"
        ),
        {"c": course_id, "s": student_id},
    )


async def _cleanup(engine: AsyncEngine, module_id: uuid.UUID, session_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM interview_sessions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM module_items WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": session_id})


async def test_interview_progress_completion_matrix(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """The rule end-to-end, including where it diverges from quizzes."""
    module_id = uuid.uuid4()
    configs = {
        "passed": uuid.uuid4(),          # one TRUE verdict -> completed
        "mixed": uuid.uuid4(),           # fail, pass, fail -> completed
        "all_failed": uuid.uuid4(),      # 3 FALSE verdicts -> PENDING (vs quiz)
        "ungraded": uuid.uuid4(),        # verdict NULL (ARQ pending) -> pending
        "in_flight": uuid.uuid4(),       # live attempt -> pending
        "never": uuid.uuid4(),           # untouched -> pending
    }
    student = seeded_users.student_id

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Interview Progress Module', 1, 'published')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        for position, (key, config_id) in enumerate(configs.items(), start=1):
            await _seed_interview_item(
                conn,
                config_id=config_id,
                course_id=seeded_users.course_id,
                module_id=module_id,
                position=position,
                title=key,
            )

        await _seed_attempt(
            conn, config_id=configs["passed"], student_id=student,
            attempt_number=1, pass_verdict=True,
        )
        for n, verdict in ((1, False), (2, True), (3, False)):
            await _seed_attempt(
                conn, config_id=configs["mixed"], student_id=student,
                attempt_number=n, pass_verdict=verdict,
            )
        for n in (1, 2, 3):
            await _seed_attempt(
                conn, config_id=configs["all_failed"], student_id=student,
                attempt_number=n, pass_verdict=False,
            )
        await _seed_attempt(
            conn, config_id=configs["ungraded"], student_id=student,
            attempt_number=1, pass_verdict=None,
        )
        await _seed_attempt(
            conn, config_id=configs["in_flight"], student_id=student,
            attempt_number=1, status="in_progress", pass_verdict=None,
        )
        await _enroll(conn, course_id=seeded_users.course_id, student_id=student)

    sid = await _seed_session(engine, student)
    token = create_access_token(user_id=student, session_id=sid)
    try:
        resp = await client.get(
            f"/api/v1/courses/{seeded_users.course_id}/interview-progress",
            headers=_auth(token),
        )
        assert resp.status_code == 200, resp.text
        by_id = {row["interview_config_id"]: row for row in resp.json()}

        passed = by_id[str(configs["passed"])]
        assert passed["passed"] is True
        assert passed["completed"] is True

        # BOOL_OR over a mixed group: one success anywhere is enough.
        mixed = by_id[str(configs["mixed"])]
        assert mixed["completed"] is True
        assert mixed["attempts_used"] == 3

        # THE divergence from the quiz rule. A quiz would be "completed" here
        # once attempts ran out; an interview stays pending because the tag
        # means passed.
        all_failed = by_id[str(configs["all_failed"])]
        assert all_failed["attempts_used"] == 3
        assert all_failed["attempts_graded"] == 3
        assert all_failed["passed"] is False
        assert all_failed["completed"] is False

        # Evaluation is an ARQ job: a fresh submit has no verdict yet. That is
        # NOT a fail, and attempts_graded exposes the gap so the UI can say so.
        ungraded = by_id[str(configs["ungraded"])]
        assert ungraded["attempts_used"] == 1
        assert ungraded["attempts_graded"] == 0
        assert ungraded["completed"] is False

        in_flight = by_id[str(configs["in_flight"])]
        assert in_flight["attempts_in_flight"] == 1
        assert in_flight["completed"] is False

        never = by_id[str(configs["never"])]
        assert never["attempts_used"] == 0
        assert never["completed"] is False
    finally:
        await _cleanup(engine, module_id, sid)


async def test_another_students_pass_does_not_leak(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Progress is per-caller: a peer's success must not complete my item."""
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Peer Pass Module', 2, 'published')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await _seed_interview_item(
            conn, config_id=config_id, course_id=seeded_users.course_id,
            module_id=module_id, position=1, title="Peer",
        )
        # The TEACHER passes it; the student asking has done nothing.
        await _seed_attempt(
            conn, config_id=config_id, student_id=seeded_users.teacher_id,
            attempt_number=1, pass_verdict=True,
        )
        await _enroll(
            conn, course_id=seeded_users.course_id, student_id=seeded_users.student_id
        )

    sid = await _seed_session(engine, seeded_users.student_id)
    token = create_access_token(user_id=seeded_users.student_id, session_id=sid)
    try:
        resp = await client.get(
            f"/api/v1/courses/{seeded_users.course_id}/interview-progress",
            headers=_auth(token),
        )
        assert resp.status_code == 200, resp.text
        row = next(r for r in resp.json() if r["interview_config_id"] == str(config_id))
        assert row["attempts_used"] == 0
        assert row["completed"] is False
    finally:
        await _cleanup(engine, module_id, sid)


async def test_soft_deleted_item_leaves_the_population(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Population must match what the curriculum renders, or keys drift."""
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Deleted Item Module', 3, 'published')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await _seed_interview_item(
            conn, config_id=config_id, course_id=seeded_users.course_id,
            module_id=module_id, position=1, title="Gone",
        )
        await conn.execute(
            text("UPDATE module_items SET deleted_at = now() WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await _enroll(
            conn, course_id=seeded_users.course_id, student_id=seeded_users.student_id
        )

    sid = await _seed_session(engine, seeded_users.student_id)
    token = create_access_token(user_id=seeded_users.student_id, session_id=sid)
    try:
        resp = await client.get(
            f"/api/v1/courses/{seeded_users.course_id}/interview-progress",
            headers=_auth(token),
        )
        assert resp.status_code == 200, resp.text
        ids = [r["interview_config_id"] for r in resp.json()]
        assert str(config_id) not in ids
    finally:
        await _cleanup(engine, module_id, sid)


async def test_cross_org_course_is_404_not_empty(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """No existence leak: a course the caller cannot view must 404.

    An empty 200 would confirm the course id exists.
    """
    org2 = uuid.uuid4()
    course2 = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, name, slug) VALUES (:o, 'Other', :s)"),
            {"o": org2, "s": f"other-{org2.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, title, slug, status) "
                "VALUES (:c, :o, :owner, 'Other Course', :s, 'published')"
            ),
            {
                "c": course2,
                "o": org2,
                # NOT NULL. Reusing the seeded teacher is fine: they are in the
                # OTHER org, which is exactly the cross-tenant shape under test.
                "owner": seeded_users.teacher_id,
                "s": f"oc-{course2.hex[:8]}",
            },
        )

    sid = await _seed_session(engine, seeded_users.student_id)
    token = create_access_token(user_id=seeded_users.student_id, session_id=sid)
    try:
        resp = await client.get(
            f"/api/v1/courses/{course2}/interview-progress",
            headers=_auth(token),
        )
        assert resp.status_code == 404, resp.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course2})
            await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org2})
            await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})
