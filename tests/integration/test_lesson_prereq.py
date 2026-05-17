"""Integration test for the ``lesson_prerequisites`` self-FK table (T3.1).

Per plan §3941-3947 / Reconciliation §A4 the lesson-after-lesson
gating is a required user decision. This test exercises the full
self-FK round-trip against the live docker postgres:

* INSERT two lessons in the same module.
* INSERT a ``lesson_prerequisites`` row linking lesson_2 → lesson_1.
* SELECT the row back.
* Verify the ``ck_lesson_prerequisites_not_self`` CHECK rejects a
  self-prereq cycle.

The fixture builds an isolated org / owner / course / module so the
test does not interact with the seeded ``test_*`` rows from
``conftest.py``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings


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
async def engine() -> AsyncEngine:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def two_lessons(engine: AsyncEngine):
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_1_id = uuid.uuid4()
    lesson_2_id = uuid.uuid4()

    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"lp-{suffix}", "name": "LessonPrereq Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"lp-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{suffix}",
                "title": "Lesson Prereq Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, :position)"
            ),
            {
                "id": module_id,
                "course": course_id,
                "title": "Module 1",
                "position": 1,
            },
        )
        for lesson_id, slug, title in [
            (lesson_1_id, "lesson-1", "Lesson 1"),
            (lesson_2_id, "lesson-2", "Lesson 2"),
        ]:
            await conn.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title) "
                    "VALUES (:id, :module, :slug, :title)"
                ),
                {"id": lesson_id, "module": module_id, "slug": slug, "title": title},
            )

    yield {
        "lesson_1_id": lesson_1_id,
        "lesson_2_id": lesson_2_id,
        "module_id": module_id,
        "course_id": course_id,
        "owner_id": owner_id,
        "org_id": org_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_prerequisites WHERE lesson_id IN (:l1, :l2)"),
            {"l1": lesson_1_id, "l2": lesson_2_id},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_lesson_prereq_self_fk(engine: AsyncEngine, two_lessons) -> None:
    lesson_1_id = two_lessons["lesson_1_id"]
    lesson_2_id = two_lessons["lesson_2_id"]

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lesson_prerequisites (lesson_id, prereq_lesson_id) "
                "VALUES (:lesson, :prereq)"
            ),
            {"lesson": lesson_2_id, "prereq": lesson_1_id},
        )

    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT prereq_lesson_id FROM lesson_prerequisites WHERE lesson_id = :lesson"),
            {"lesson": lesson_2_id},
        )
        rows = result.fetchall()

    assert len(rows) == 1
    assert rows[0][0] == lesson_1_id


async def test_lesson_prereq_self_cycle_rejected(engine: AsyncEngine, two_lessons) -> None:
    lesson_1_id = two_lessons["lesson_1_id"]

    with pytest.raises(IntegrityError, match=r"ck_lesson_prerequisites_not_self|check"):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO lesson_prerequisites (lesson_id, prereq_lesson_id) VALUES (:l, :l)"
                ),
                {"l": lesson_1_id},
            )


async def test_module_item_xor_violation_rejected(engine: AsyncEngine, two_lessons) -> None:
    module_id = two_lessons["module_id"]

    with pytest.raises(IntegrityError, match=r"ck_module_items_item_type|check"):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO module_items "
                    "(module_id, item_type, lesson_id, position) "
                    "VALUES (:m, 'lesson', NULL, 1)"
                ),
                {"m": module_id},
            )
