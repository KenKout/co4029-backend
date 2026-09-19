"""P1 #9 pin: enrollment revocation policy for interview sessions is the
QUIZ-CONSISTENT GRACE policy (explicit decision, was implicit before).

Precedent: quizzes gate enrollment at ``start_attempt`` /
``get_published_quiz`` (``can_view_course_content``); an in-flight attempt
after a mid-exam revocation is NOT re-checked on ``record_answer`` — the
candidate may finish, and review reads stay owner-scoped. Interviews now pin
the identical contract:

- a DROPPED student cannot START (existing ``_ensure_config_course_enrolled``
  route gate — asserted here so the policy is explicit);
- a session that is already ``in_progress`` when the enrollment is revoked
  may still onboarding/respond/finish (no mid-exam cut);
- the finished session stays owner-readable (history/self reads).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def enroll_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    org_id = uuid4()
    teacher_id = uuid4()
    student_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    config_id = uuid4()
    session_id = uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"en-{suffix}", "name": "Enroll Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"en-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'EN Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"en-course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                "VALUES (:id, :course, :student, 'active')"
            ),
            {"id": uuid4(), "course": course_id, "student": student_id},
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
                "(id, course_id, module_id, title, status, persona, created_by, slug) "
                "VALUES (:id, :course, :module, 'EN Interview', 'published', 'neutral', "
                ":teacher, :slug)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"en-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, onboarding_stage, interview_language, assessment_started_at) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'hybrid', now() - interval "
                "'20 minutes', 'completed', 'en', now() - interval '15 minutes')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {
        "session_id": session_id,
        "student_id": student_id,
        "course_id": course_id,
        "config_id": config_id,
    }

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_session_messages WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id}
        )
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"), {"c": course_id}
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"),
            {"t": teacher_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _revoke(test_engine: AsyncEngine, course_id: Any, student_id: Any) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        await db.execute(
            text(
                "UPDATE course_enrollments SET status = 'dropped' "
                "WHERE course_id = :c AND student_id = :s"
            ),
            {"c": course_id, "s": student_id},
        )
        await db.commit()


def _actor(student_id: Any) -> Any:
    return type("_A", (), {"user_id": student_id})()


async def test_dropped_student_cannot_start_a_new_attempt(
    enroll_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The BR gate stays: dropped => the start path refuses (404, no leak)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from fastapi import HTTPException

    from abridgeai.features.interviews.models import InterviewConfig
    from abridgeai.features.interviews.routers import learner as learner_router

    await _revoke(test_engine, enroll_probe["course_id"], enroll_probe["student_id"])
    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        config = await db.get(InterviewConfig, enroll_probe["config_id"])
        assert config is not None
        with pytest.raises(HTTPException) as exc_info:
            await learner_router._ensure_config_course_enrolled(
                db,
                _actor(enroll_probe["student_id"]),
                config,
            )
        assert exc_info.value.status_code == 404


async def test_in_progress_session_survives_revocation_and_can_finish(
    enroll_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Quiz-consistent grace: an in-flight attempt is not cut mid-exam; the
    candidate may submit (the session terminalizes instead of 404)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import taking as taking_service

    await _revoke(test_engine, enroll_probe["course_id"], enroll_probe["student_id"])
    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        session = await taking_service.submit_session(
            db, enroll_probe["session_id"], _actor(enroll_probe["student_id"]), arq_pool=None
        )
        await db.commit()
    assert session.status == "completed"


async def test_finished_session_stays_owner_readable_after_revocation(
    enroll_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Self reads of one's own session are owner-scoped, not enrollment-gated
    (mirrors the quiz attempt-review precedent)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import taking as taking_service

    await _revoke(test_engine, enroll_probe["course_id"], enroll_probe["student_id"])
    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        session = await taking_service.get_session_for_user(
            db, enroll_probe["session_id"], enroll_probe["student_id"]
        )
        assert session is not None
