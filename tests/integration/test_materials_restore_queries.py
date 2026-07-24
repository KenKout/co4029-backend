"""Query-level tests for material soft-delete recovery (restore + list-deleted).

DB-only (no S3/moto): exercises the queries that back the teacher-facing
"Recently deleted" + Restore recovery feature directly against Postgres,
mirroring the fixture pattern in ``test_materials_queries.py``.

Covers:
* ``restore_soft_deleted_material`` lifts the tombstone on the material AND
  its cascaded versions (the inverse of ``soft_delete_cascade``).
* ``list_deleted_materials`` surfaces only tombstoned rows, newest-first,
  respecting the retention ``since`` window.
* ``get_material_including_deleted`` bypasses the T0.7 soft-delete filter.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import the full app so EVERY ORM model is registered before soft_delete_cascade
# walks relationships (it recurses ModuleItem -> Quiz/Interview, which live in
# sibling features). A partial model import leaves those mappers unconfigured
# and mapper initialization fails.
import abridgeai.api  # noqa: F401  -- force full model registry via router imports
import abridgeai.features.courses.models  # noqa: F401  -- register lessons / modules / courses FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users / organizations / storage_objects FK targets
import abridgeai.features.materials.models  # noqa: F401  -- register learning_material_* FK targets
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.materials.models import LearningMaterial
from abridgeai.features.materials.queries import (
    get_material_for_authoring,
    get_material_including_deleted,
    list_all_materials,
    list_deleted_materials,
    restore_soft_deleted_material,
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
async def fixture_data(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_id = uuid.uuid4()

    mat_live = uuid.uuid4()
    mat_deleted = uuid.uuid4()

    ver_live = uuid.uuid4()
    ver_deleted = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Restore Org', 'active')"
            ),
            {"id": org_id, "slug": f"rorg-{org_id.hex[:8]}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": owner, "email": f"owner-{owner.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:c, :o, :u, :s, 'Restore Course', 'published')"
            ),
            {"c": course_id, "o": org_id, "u": owner, "s": f"course-{course_id.hex[:6]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Module', 1, 'published')"
            ),
            {"m": module_id, "c": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l, :m, 'lesson-1', 'Lesson', 'published')"
            ),
            {"l": lesson_id, "m": module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type, size_bytes) "
                "VALUES (:id, 'test-bucket', 'materials/restore.pdf', 'application/pdf', 1024)"
            ),
            {"id": storage_id},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type, "
                "ai_processing_enabled, visible_to_students) VALUES "
                "(:m1, :l, 'Live Doc', 'pdf', TRUE, TRUE), "
                "(:m2, :l, 'Deleted Doc', 'pdf', TRUE, FALSE)"
            ),
            {"m1": mat_live, "m2": mat_deleted, "l": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, is_current, processing_status) "
                "VALUES "
                "(:v1, :m1, :s, 1, TRUE, 'ready'), "
                "(:v2, :m2, :s, 1, TRUE, 'ready')"
            ),
            {"v1": ver_live, "v2": ver_deleted, "m1": mat_live, "m2": mat_deleted, "s": storage_id},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = :v WHERE id = :m"),
            {"v": ver_live, "m": mat_live},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = :v WHERE id = :m"),
            {"v": ver_deleted, "m": mat_deleted},
        )

    yield {
        "lesson_id": lesson_id,
        "owner": owner,
        "mat_live": mat_live,
        "mat_deleted": mat_deleted,
        "ver_deleted": ver_deleted,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE material_id IN (:m1, :m2)"),
            {"m1": mat_live, "m2": mat_deleted},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id IN (:m1, :m2)"),
            {"m1": mat_live, "m2": mat_deleted},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM storage_objects WHERE id = :s"), {"s": storage_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": owner})
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _soft_delete(
    session_factory: async_sessionmaker[AsyncSession],
    material_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> None:
    async with session_factory() as session:
        material = await session.get(LearningMaterial, material_id)
        assert material is not None
        await soft_delete_cascade(session, material, actor_id=actor_id)
        await session.commit()


async def test_soft_delete_hides_then_restore_brings_back(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    lesson_id = fixture_data["lesson_id"]
    mat = fixture_data["mat_deleted"]

    await _soft_delete(session_factory, mat, fixture_data["owner"])

    # Hidden from the active list (T0.7 filter), and get_material_for_authoring
    # (db.get, listener-gated) returns None.
    async with session_factory() as session:
        active = await list_all_materials(session, lesson_id)
        assert all(m.id != mat for m in active)
        assert await get_material_for_authoring(session, mat) is None
        # But the include-deleted lookup finds it.
        including = await get_material_including_deleted(session, mat)
        assert including is not None
        assert including.deleted_at is not None

    # Restore lifts the tombstone.
    async with session_factory() as session:
        restored = await restore_soft_deleted_material(session, mat)
        assert restored is True
        await session.commit()

    async with session_factory() as session:
        active = await list_all_materials(session, lesson_id)
        assert any(m.id == mat for m in active), "restored material must reappear in active list"
        again = await get_material_for_authoring(session, mat)
        assert again is not None
        assert again.deleted_at is None


async def test_restore_lifts_version_tombstone(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    mat = fixture_data["mat_deleted"]
    ver = fixture_data["ver_deleted"]

    await _soft_delete(session_factory, mat, fixture_data["owner"])

    # The cascade tombstoned the version too.
    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    "SELECT deleted_at FROM learning_material_versions "
                    "WHERE id = :v"
                ),
                {"v": ver},
            )
        ).one()
        assert row.deleted_at is not None, "cascade should tombstone the version"

    async with session_factory() as session:
        await restore_soft_deleted_material(session, mat)
        await session.commit()

    # Restore lifted it on the version as well.
    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    "SELECT deleted_at FROM learning_material_versions "
                    "WHERE id = :v"
                ),
                {"v": ver},
            )
        ).one()
        assert row.deleted_at is None, "restore must lift the version tombstone"


async def test_restore_active_material_returns_false(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    # mat_live was never deleted → nothing to restore.
    async with session_factory() as session:
        result = await restore_soft_deleted_material(session, fixture_data["mat_live"])
        assert result is False


async def test_list_deleted_only_tombstoned_and_respects_since(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    lesson_id = fixture_data["lesson_id"]
    mat = fixture_data["mat_deleted"]

    # Before deletion: nothing in the trash.
    async with session_factory() as session:
        assert await list_deleted_materials(session, lesson_id) == []

    await _soft_delete(session_factory, mat, fixture_data["owner"])

    async with session_factory() as session:
        deleted = await list_deleted_materials(session, lesson_id)
        assert [m.id for m in deleted] == [mat]
        # The live material is NOT in the deleted list.
        assert all(m.id != fixture_data["mat_live"] for m in deleted)

    # A retention window in the future (since = now + 1 day) excludes it.
    async with session_factory() as session:
        future = datetime.now(tz=UTC) + timedelta(days=1)
        assert await list_deleted_materials(session, lesson_id, since=future) == []

    # A window reaching into the past includes it.
    async with session_factory() as session:
        past = datetime.now(tz=UTC) - timedelta(days=30)
        deleted = await list_deleted_materials(session, lesson_id, since=past)
        assert [m.id for m in deleted] == [mat]
