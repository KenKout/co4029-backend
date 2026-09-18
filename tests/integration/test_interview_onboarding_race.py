"""P1 #14 regression: concurrent DISTINCT onboarding actions at one stage
must advance the stage exactly once.

Two tabs (or a double-click racing a rerender) send two actions with
different turn keys for the same stage. The dedupe only catches the SAME
key; both transactions passed the stage equality check and each wrote its
own transition, skipping a step (or starting the assessment early). The
stage transition is now a conditional UPDATE against the stage the
transaction READ — the second writer's update matches zero rows, the
transaction restarts from the winner's canonical stage, and exactly one
ceremony/advance exists.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def onboard_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
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
            {"id": org_id, "slug": f"ob-{suffix}", "name": "Onboard Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"ob-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'OB Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"ob-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'OB Interview', 'published', 'neutral', "
                ":teacher, :slug)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"ob-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'hybrid', now(), "
                "'identity_check', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "student_id": student_id}

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_session_messages WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id}
        )
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"),
            {"t": teacher_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


def _actor(student_id: uuid.UUID) -> Any:
    return type("_A", (), {"user_id": student_id})()


async def _stage(test_engine: AsyncEngine, session_id: uuid.UUID) -> str:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.models import InterviewSession

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        session = await db.get(InterviewSession, session_id)
        assert session is not None
        return str(session.onboarding_stage)


async def _turn_count(test_engine: AsyncEngine, session_id: uuid.UUID) -> int:
    async with test_engine.begin() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT count(*) FROM interview_session_messages "
                    "WHERE session_id = :s AND metadata_json->>'kind' = 'onboarding'"
                ),
                {"s": session_id},
            )
        ).scalar_one()


async def test_distinct_key_duplicate_action_advances_once(
    onboard_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Same stage, same action, TWO turn keys (two tabs / double-click): the
    stage must advance identity_check → audio_check exactly ONCE and the
    second caller must be served the canonical (winner) stage."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import onboarding as onboarding_service

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    actor = _actor(onboard_probe["student_id"])

    barrier = asyncio.Event()
    first_started = asyncio.Event()

    async def call_a() -> Any:
        async with maker() as db:
            result = await onboarding_service.respond(
                db,
                session_id=onboard_probe["session_id"],
                actor=actor,
                stage="identity_check",
                response_text=None,
                action="confirm_identity",
                language="en",
                turn_key=f"ob-dup-a-{uuid4().hex[:8]}",
                _before_stage_write=lambda: _park(
                    first_started, barrier
                ),
            )
            await db.commit()
            return result

    async def call_b() -> Any:
        async with maker() as db:
            result = await onboarding_service.respond(
                db,
                session_id=onboard_probe["session_id"],
                actor=actor,
                stage="identity_check",
                response_text=None,
                action="confirm_identity",
                language="en",
                turn_key=f"ob-dup-b-{uuid4().hex[:8]}",
            )
            await db.commit()
            return result

    task_a = asyncio.create_task(call_a())
    await first_started.wait()
    task_b = asyncio.create_task(call_b())
    await asyncio.sleep(0.05)  # B passes its stage check while A is parked
    barrier.set()
    await asyncio.gather(task_a, task_b)

    assert await _stage(test_engine, onboard_probe["session_id"]) == "audio_check", (
        "exactly one transition: identity_check → audio_check"
    )
    # The persisted user turns: BOTH attempts record an answer (each key is a
    # real candidate action), but only ONE stage transition/ceremony exists.
    ceremony_rows = await _ceremony_count(test_engine, onboard_probe["session_id"])
    assert ceremony_rows == 1, "exactly one audio_check ceremony row"


async def _park(started: asyncio.Event, barrier: asyncio.Event) -> None:
    started.set()
    await barrier.wait()


async def _ceremony_count(test_engine: AsyncEngine, session_id: uuid.UUID) -> int:
    async with test_engine.begin() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT count(*) FROM interview_session_messages "
                    "WHERE session_id = :s AND role = 'ai' "
                    "AND metadata_json->>'ceremony_key' = 'audio_check'"
                ),
                {"s": session_id},
            )
        ).scalar_one()
