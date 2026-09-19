"""P1 #6 regression: destructive config lifecycle transitions are refused
while sessions are live.

The realtime runtime and the evaluator read the LIVE config/outcomes/
questions — a teacher unpublishing/archiving/deleting mid-interview could
re-point or remove the data an in-flight candidate (or a pending evaluation
recovery) is still using. ``in_progress`` is the only live status; terminal
sessions keep reading by id and are never re-graded against new content.
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
async def lifecycle_probe(
    test_engine: AsyncEngine,
) -> AsyncIterator[dict[str, Any]]:
    org_id = uuid4()
    teacher_id = uuid4()
    student_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    config_id = uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"lc-{suffix}", "name": "LC Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"lc-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'LC Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"lc-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'LC Interview', 'published', 'neutral', "
                ":teacher, :slug)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"lc-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'hybrid', now(), "
                "'completed', 'en')"
            ),
            {"id": uuid4(), "cfg": config_id, "student": student_id},
        )

    yield {"config_id": config_id, "teacher_id": teacher_id}

    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM interview_sessions WHERE interview_config_id = :c",
            ),
            {"c": config_id},
        )
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"),
            {"t": teacher_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


def _actor(user_id: Any) -> Any:
    return type("_A", (), {"user_id": user_id})()


@pytest.mark.parametrize(
    "transition",
    ["archive", "unpublish", "delete"],
)
async def test_destructive_transitions_refused_while_live(
    lifecycle_probe: dict[str, Any], test_engine: AsyncEngine, transition: str
) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import authoring

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        async def call_transition() -> None:
            actor = _actor(lifecycle_probe["teacher_id"])
            config_id = lifecycle_probe["config_id"]
            if transition == "archive":
                await authoring.archive_interview_config(db, config_id, actor)
            elif transition == "unpublish":
                await authoring.unpublish_interview_config(db, config_id, actor)
            else:
                await authoring.delete_interview_config(db, config_id, actor)

        with pytest.raises(Exception) as exc_info:
            await call_transition()
        await db.rollback()
    assert "in progress" in str(exc_info.value)


async def test_non_destructive_unarchive_stays_allowed(
    lifecycle_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Unarchive only restores visibility — no guard, and it raises the
    expected state error (config is 'published', not 'archived'), proving the
    guard did not fire first."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import authoring

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        with pytest.raises(Exception) as exc_info:
            await authoring.unarchive_interview_config(
                db, lifecycle_probe["config_id"], _actor(lifecycle_probe["teacher_id"])
            )
        await db.rollback()
    assert "expected 'archived'" in str(exc_info.value)
