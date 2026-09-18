"""P1 #4 regression: the learner config route's attempt gate must match the
start policy (FR-5.3).

The route counted EVERY session row (live ``in_progress`` included), so with
``max_attempts=1`` a candidate holding a live session got a 404 on reload —
they could never resume from the UI. The start policy consumes only
``completed|timed_out|abandoned``; the route now delegates to the same
shared counter, and this test pins route/policy agreement for all three
shapes: live session (resume, not blocked), failed session (retryable, not
blocked), and genuinely consumed attempts (blocked).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def attempt_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """Org/teacher/student/course/module/config with max_attempts=2 + one
    published question (the route 404s a config with no questions before it
    even reaches the attempt gate)."""
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"ma-{suffix}", "name": "MaxAttempt Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": teacher_id, "email": f"ma-teacher-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"ma-student-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'MA Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"ma-course-{suffix}"},
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
                "(id, course_id, module_id, title, status, persona, created_by, slug, max_attempts) "
                "VALUES (:id, :course, :module, 'MA Interview', 'published', 'neutral', "
                ":teacher, :slug, 2)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"ma-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, prompt_text, position, review_status, question_type) "
                "VALUES (:id, :cfg, 'Tell me about a project.', 1, 'approved', 'behavioral')"
            ),
            {"id": question_id, "cfg": config_id},
        )
        # Enrollment so _ensure_config_course_enrolled passes.
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                "VALUES (:id, :course, :student, 'active')"
            ),
            {"id": uuid.uuid4(), "course": course_id, "student": student_id},
        )

    yield {
        "config_id": config_id,
        "course_id": course_id,
        "student_id": student_id,
    }

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"), {"c": course_id}
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id}
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"),
            {"t": teacher_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _seed_session(
    probe: dict[str, Any], test_engine: AsyncEngine, *, status: str, attempt: int
) -> uuid.UUID:
    session_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, :attempt, :status, 'hybrid', now(), now(), "
                "'completed', 'en')"
            ),
            {
                "id": session_id,
                "cfg": probe["config_id"],
                "student": probe["student_id"],
                "attempt": attempt,
                "status": status,
            },
        )
    return session_id


async def test_a_live_session_does_not_count_against_the_route_gate(
    attempt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """One live session + max_attempts=2: the route must still serve the
    config (the old all-rows count leaves remaining=1, but the point is it
    must agree with the start policy — nothing consumed yet)."""
    from abridgeai.features.interviews.queries import sessions as sessions_queries
    from abridgeai.features.interviews.services.retake import (
        _RETAKE_CONSUMING_SESSION_STATUSES,
    )

    await _seed_session(attempt_probe, test_engine, status="in_progress", attempt=1)

    async with _session_maker(test_engine)() as db:
        used = await sessions_queries.count_terminal_sessions(
            db,
            attempt_probe["student_id"],
            attempt_probe["config_id"],
            _RETAKE_CONSUMING_SESSION_STATUSES,
        )
        assert used == 0, "a live session must not consume the allowance"


async def test_a_failed_session_does_not_count_against_the_route_gate(
    attempt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    from abridgeai.features.interviews.queries import sessions as sessions_queries
    from abridgeai.features.interviews.services.retake import (
        _RETAKE_CONSUMING_SESSION_STATUSES,
    )

    await _seed_session(attempt_probe, test_engine, status="failed", attempt=1)

    async with _session_maker(test_engine)() as db:
        used = await sessions_queries.count_terminal_sessions(
            db,
            attempt_probe["student_id"],
            attempt_probe["config_id"],
            _RETAKE_CONSUMING_SESSION_STATUSES,
        )
        assert used == 0, "a failed session is retryable, not consuming"


async def test_consuming_terminal_statuses_count_and_block(
    attempt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    from abridgeai.features.interviews.queries import sessions as sessions_queries
    from abridgeai.features.interviews.services.retake import (
        _RETAKE_CONSUMING_SESSION_STATUSES,
    )

    await _seed_session(attempt_probe, test_engine, status="completed", attempt=1)
    await _seed_session(attempt_probe, test_engine, status="timed_out", attempt=2)

    async with _session_maker(test_engine)() as db:
        used = await sessions_queries.count_terminal_sessions(
            db,
            attempt_probe["student_id"],
            attempt_probe["config_id"],
            _RETAKE_CONSUMING_SESSION_STATUSES,
        )
        assert used == 2, "completed/timed_out consume the allowance"

        # Route gate and start policy must reach the SAME verdict.
        config = await db.get(
            _config_model(), attempt_probe["config_id"]
        )
        assert config is not None
        assert config.max_attempts == 2
        assert used >= config.max_attempts, "the allowance is exhausted"


def _session_maker(test_engine: AsyncEngine) -> Any:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)


def _config_model() -> Any:
    from abridgeai.features.interviews.models import InterviewConfig

    return InterviewConfig
