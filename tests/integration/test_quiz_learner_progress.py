"""Integration tests for the learner quiz-progress endpoint (course-learn).

``GET /api/v1/courses/{course_id}/quiz-progress`` returns per-quiz
completion state for the calling student. Completion follows the
teacher-configured milestone (``passing_score_percent`` via the
materialised gradebook): passed → completed; failed with every allowed
attempt consumed (and nothing in flight) → completed; otherwise pending.
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
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token
from abridgeai.features.quizzes.routers import learner_router

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
async def app(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
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


async def _seed_quiz(
    conn: AsyncConnection,
    *,
    quiz_id: uuid.UUID,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    position: int,
    title: str,
    allow_retakes: bool = True,
    max_attempts: int | None = 2,
) -> None:
    """One published quiz + its module_items ordering row (fresh UUIDs)."""
    item_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO quizzes (id, course_id, module_id, title, status, "
            "allow_retakes, max_attempts) "
            "VALUES (:id, :c, :m, :title, 'published', :ar, :ma)"
        ),
        {
            "id": quiz_id,
            "c": course_id,
            "m": module_id,
            "title": title,
            "ar": allow_retakes,
            "ma": max_attempts,
        },
    )
    await conn.execute(
        text(
            "INSERT INTO module_items (id, module_id, item_type, quiz_id, position) "
            "VALUES (:id, :m, 'quiz', :q, :pos)"
        ),
        {"id": item_id, "m": module_id, "q": quiz_id, "pos": position},
    )


async def _seed_attempt(
    conn: AsyncConnection,
    *,
    quiz_id: uuid.UUID,
    student_id: uuid.UUID,
    attempt_number: int,
    status: str,
    score_percent: float | None = None,
    passed: bool | None = None,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO quiz_attempts (id, quiz_id, student_id, attempt_number, status, "
            "score_percent, passed) "
            "VALUES (:id, :q, :s, :n, :st, :sc, :pa)"
        ),
        {
            "id": uuid.uuid4(),
            "q": quiz_id,
            "s": student_id,
            "n": attempt_number,
            "st": status,
            "sc": score_percent,
            "pa": passed,
        },
    )


async def _seed_grade(
    conn: AsyncConnection,
    *,
    quiz_id: uuid.UUID,
    student_id: uuid.UUID,
    grade_percent: float,
    passed: bool,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO quiz_grades (id, quiz_id, student_id, grade_percent, passed, "
            "grading_method, attempts_counted, computed_at) "
            "VALUES (:id, :q, :s, :g, :p, 'highest', 1, NOW())"
        ),
        {
            "id": uuid.uuid4(),
            "q": quiz_id,
            "s": student_id,
            "g": grade_percent,
            "p": passed,
        },
    )


# ---------------------------------------------------------------------------
# Completion matrix
# ---------------------------------------------------------------------------


async def test_quiz_progress_completion_matrix(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """The milestone rule end-to-end: passed, failed+exhausted, remaining
    attempts, in-flight, no-retakes clamp, unlimited, never attempted."""
    module_id = uuid.uuid4()
    quizzes = {
        "passed": uuid.uuid4(),          # 1 graded pass → completed
        "failed_exhausted": uuid.uuid4(),  # 2/2 used, failed → completed
        "failed_remaining": uuid.uuid4(),  # 2/3 used, failed → pending
        "in_flight": uuid.uuid4(),       # 1/1 used but in_progress → pending
        "no_retakes": uuid.uuid4(),      # allow_retakes=F, 1 failed → completed
        "unlimited": uuid.uuid4(),       # max_attempts NULL, 2 failed → pending
        "never": uuid.uuid4(),           # no attempts → pending
    }
    student = seeded_users.student_id

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Progress Module', 1, 'published')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["passed"], course_id=seeded_users.course_id,
            module_id=module_id, position=1, title="Passed",
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["failed_exhausted"], course_id=seeded_users.course_id,
            module_id=module_id, position=2, title="Failed Exhausted", max_attempts=2,
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["failed_remaining"], course_id=seeded_users.course_id,
            module_id=module_id, position=3, title="Failed Remaining", max_attempts=3,
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["in_flight"], course_id=seeded_users.course_id,
            module_id=module_id, position=4, title="In Flight", max_attempts=1,
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["no_retakes"], course_id=seeded_users.course_id,
            module_id=module_id, position=5, title="No Retakes",
            allow_retakes=False, max_attempts=5,
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["unlimited"], course_id=seeded_users.course_id,
            module_id=module_id, position=6, title="Unlimited", max_attempts=None,
        )
        await _seed_quiz(
            conn, quiz_id=quizzes["never"], course_id=seeded_users.course_id,
            module_id=module_id, position=7, title="Never",
        )

        # passed: 1 graded pass (grade-of-record passed).
        await _seed_attempt(
            conn, quiz_id=quizzes["passed"], student_id=student,
            attempt_number=1, status="graded", score_percent=88.0, passed=True,
        )
        await _seed_grade(
            conn, quiz_id=quizzes["passed"], student_id=student,
            grade_percent=88.0, passed=True,
        )
        # failed_exhausted: 2 graded fails, both attempts used.
        await _seed_attempt(
            conn, quiz_id=quizzes["failed_exhausted"], student_id=student,
            attempt_number=1, status="graded", score_percent=45.0, passed=False,
        )
        await _seed_attempt(
            conn, quiz_id=quizzes["failed_exhausted"], student_id=student,
            attempt_number=2, status="graded", score_percent=50.0, passed=False,
        )
        await _seed_grade(
            conn, quiz_id=quizzes["failed_exhausted"], student_id=student,
            grade_percent=50.0, passed=False,
        )
        # failed_remaining: 2 graded fails, but 3 attempts allowed.
        await _seed_attempt(
            conn, quiz_id=quizzes["failed_remaining"], student_id=student,
            attempt_number=1, status="graded", score_percent=40.0, passed=False,
        )
        await _seed_attempt(
            conn, quiz_id=quizzes["failed_remaining"], student_id=student,
            attempt_number=2, status="graded", score_percent=55.0, passed=False,
        )
        await _seed_grade(
            conn, quiz_id=quizzes["failed_remaining"], student_id=student,
            grade_percent=55.0, passed=False,
        )
        # in_flight: the single allowed slot is consumed but still open.
        await _seed_attempt(
            conn, quiz_id=quizzes["in_flight"], student_id=student,
            attempt_number=1, status="in_progress",
        )
        # no_retakes: allow_retakes=F clamps to 1 → the one graded fail is terminal.
        await _seed_attempt(
            conn, quiz_id=quizzes["no_retakes"], student_id=student,
            attempt_number=1, status="graded", score_percent=30.0, passed=False,
        )
        await _seed_grade(
            conn, quiz_id=quizzes["no_retakes"], student_id=student,
            grade_percent=30.0, passed=False,
        )
        # unlimited: two fails, but max_attempts NULL → never exhausted.
        await _seed_attempt(
            conn, quiz_id=quizzes["unlimited"], student_id=student,
            attempt_number=1, status="graded", score_percent=35.0, passed=False,
        )
        await _seed_attempt(
            conn, quiz_id=quizzes["unlimited"], student_id=student,
            attempt_number=2, status="graded", score_percent=42.0, passed=False,
        )
        await _seed_grade(
            conn, quiz_id=quizzes["unlimited"], student_id=student,
            grade_percent=42.0, passed=False,
        )
        # never: no rows at all.
        # BR gate: quiz-progress is a course-item read — the student must
        # be enrolled for it to resolve (can_view_course_content).
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (course_id, student_id, status, source) "
                "VALUES (:c, :s, 'active', 'manager_bulk') "
                "ON CONFLICT (course_id, student_id) DO NOTHING"
            ),
            {"c": seeded_users.course_id, "s": student},
        )

    sid = await _seed_session(engine, student)
    token = create_access_token(user_id=student, session_id=sid)
    try:
        resp = await client.get(
            f"/api/v1/courses/{seeded_users.course_id}/quiz-progress",
            headers=_auth(token),
        )
        assert resp.status_code == 200, resp.text
        rows = {r["quiz_id"]: r for r in resp.json()}
        assert set(rows) == {str(q) for q in quizzes.values()}, resp.text

        p = rows[str(quizzes["passed"])]
        assert p["completed"] is True
        assert p["passed"] is True
        assert p["attempts_used"] == 1
        assert p["attempts_remaining"] == 1

        p = rows[str(quizzes["failed_exhausted"])]
        assert p["completed"] is True
        assert p["passed"] is False
        assert p["attempts_used"] == 2
        assert p["attempts_remaining"] == 0

        p = rows[str(quizzes["failed_remaining"])]
        assert p["completed"] is False
        assert p["passed"] is False
        assert p["attempts_used"] == 2
        assert p["attempts_remaining"] == 1

        p = rows[str(quizzes["in_flight"])]
        assert p["completed"] is False
        assert p["passed"] is None
        assert p["attempts_used"] == 1
        assert p["attempts_remaining"] == 0

        p = rows[str(quizzes["no_retakes"])]
        assert p["completed"] is True
        assert p["passed"] is False
        assert p["max_attempts"] == 1  # clamped despite max_attempts=5
        assert p["attempts_used"] == 1
        assert p["attempts_remaining"] == 0

        p = rows[str(quizzes["unlimited"])]
        assert p["completed"] is False
        assert p["passed"] is False
        assert p["max_attempts"] is None
        assert p["attempts_remaining"] is None
        assert p["attempts_used"] == 2

        p = rows[str(quizzes["never"])]
        assert p["completed"] is False
        assert p["passed"] is None
        assert p["attempts_used"] == 0
        assert p["attempts_remaining"] == 2
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM quiz_grades WHERE quiz_id = ANY(:ids)"),
                {"ids": list(quizzes.values())},
            )
            await conn.execute(
                text("DELETE FROM quiz_attempts WHERE quiz_id = ANY(:ids)"),
                {"ids": list(quizzes.values())},
            )
            await conn.execute(
                text("DELETE FROM module_items WHERE module_id = :m"),
                {"m": module_id},
            )
            await conn.execute(
                text("DELETE FROM quizzes WHERE id = ANY(:ids)"),
                {"ids": list(quizzes.values())},
            )
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": sid},
            )


async def test_quiz_progress_cross_org_returns_404(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A student outside the course's organization gets 404 — no existence
    leak on the new endpoint (same perimeter as /quizzes/{id})."""
    org2 = uuid.uuid4()
    course2 = uuid.uuid4()
    module2 = uuid.uuid4()
    quiz2 = uuid.uuid4()
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
        await _seed_quiz(
            conn, quiz_id=quiz2, course_id=course2, module_id=module2,
            position=1, title="X Quiz",
        )

    sid = await _seed_session(engine, seeded_users.student_id)
    token = create_access_token(user_id=seeded_users.student_id, session_id=sid)
    try:
        resp = await client.get(
            f"/api/v1/courses/{course2}/quiz-progress",
            headers=_auth(token),
        )
        assert resp.status_code == 404, resp.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM module_items WHERE module_id = :m"),
                {"m": module2},
            )
            await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz2})
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module2})
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course2})
            await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org2})
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": sid},
            )
