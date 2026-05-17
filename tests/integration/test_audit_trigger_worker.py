"""T3: worker without HTTP context falls back to ``SYSTEM_ACTOR_ID``.

Background workers (Celery, scheduled tasks, alembic data migrations)
run outside any FastAPI request lifecycle, so ``current_actor_var`` is
unset and the ``after_begin`` listener short-circuits without binding
``app.actor_id``. The Postgres trigger's ``COALESCE(NULLIF(..., '')::uuid,
system_actor)`` clause must then stamp the system sentinel
(``00000000-0000-0000-0000-000000000001``, seeded by migration 0004).

Worker semantics: this test does NOT call ``current_actor_var.set()``
and does NOT exercise the HTTP layer; it goes directly through the
engine, mirroring how a real worker writes rows.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import abridgeai.features.identity.models  # noqa: F401  -- register FK targets
from abridgeai.core.config import get_settings
from abridgeai.core.db import register_app_actor_listener

SYSTEM_ACTOR_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


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
    register_app_actor_listener(eng)
    yield eng
    await eng.dispose()


async def test_worker_no_actor_stamps_system(engine: AsyncEngine) -> None:
    org_id = uuid.uuid4()
    course_id = uuid.uuid4()
    suffix = course_id.hex[:8]
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO organizations (id, slug, name, status) "
                    "VALUES (:id, :slug, :name, 'active')"
                ),
                {"id": org_id, "slug": f"worker-org-{suffix}", "name": "Worker Test Org"},
            )
            await conn.execute(
                text(
                    "INSERT INTO courses "
                    "(id, organization_id, owner_user_id, slug, title, status) "
                    "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
                ),
                {
                    "id": course_id,
                    "org": org_id,
                    "owner": SYSTEM_ACTOR_ID,
                    "slug": f"worker-course-{suffix}",
                    "title": "Worker-inserted course",
                },
            )

        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT created_by, updated_by FROM courses WHERE id = :id"),
                    {"id": course_id},
                )
            ).one()

        assert row.created_by == SYSTEM_ACTOR_ID, (
            f"expected created_by={SYSTEM_ACTOR_ID}, got {row.created_by}"
        )
        assert row.updated_by == SYSTEM_ACTOR_ID, (
            f"expected updated_by={SYSTEM_ACTOR_ID}, got {row.updated_by}"
        )

        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE courses SET title = :t WHERE id = :id"),
                {"t": "Worker-updated course", "id": course_id},
            )

        async with engine.begin() as conn:
            row2 = (
                await conn.execute(
                    text("SELECT created_by, updated_by, title FROM courses WHERE id = :id"),
                    {"id": course_id},
                )
            ).one()

        assert row2.title == "Worker-updated course"
        assert row2.created_by == SYSTEM_ACTOR_ID
        assert row2.updated_by == SYSTEM_ACTOR_ID
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
            await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})
