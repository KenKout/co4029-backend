from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.courses.models  # noqa: F401  -- register lessons / modules / courses FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users / organizations / storage_objects FK targets
import abridgeai.features.materials.models  # noqa: F401  -- register learning_material_* FK targets
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.materials.models import (
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def lesson_chain(engine: AsyncEngine):
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_object_id = uuid.uuid4()
    suffix = org_id.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"mat-{suffix}", "name": "Materials Cascade Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"mat-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"mat-course-{suffix}",
                "title": "Materials Cascade Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, :title, 1, 'draft')"
            ),
            {"id": module_id, "cid": course_id, "title": "Materials Cascade Module"},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:id, :mid, :slug, :title, 'draft')"
            ),
            {
                "id": lesson_id,
                "mid": module_id,
                "slug": f"mat-lesson-{suffix}",
                "title": "Materials Cascade Lesson",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, :bucket, :key, :mime)"
            ),
            {
                "id": storage_object_id,
                "bucket": "test",
                "key": f"mat-{suffix}/source.pdf",
                "mime": "application/pdf",
            },
        )
    yield {
        "org_id": org_id,
        "owner_id": owner_id,
        "course_id": course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "storage_object_id": storage_object_id,
    }
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM document_chunks WHERE material_version_id IN ("
                "  SELECT id FROM learning_material_versions WHERE material_id IN ("
                "    SELECT id FROM learning_materials WHERE lesson_id = :l"
                "  )"
                ")"
            ),
            {"l": lesson_id},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = NULL WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(
            text(
                "DELETE FROM learning_material_versions WHERE material_id IN ("
                "  SELECT id FROM learning_materials WHERE lesson_id = :l"
                ")"
            ),
            {"l": lesson_id},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": storage_object_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_soft_delete_cascade_walks_versions(
    session_factory: async_sessionmaker[AsyncSession], lesson_chain
) -> None:
    owner_id = lesson_chain["owner_id"]

    async with session_factory() as session:
        material = LearningMaterial(
            lesson_id=lesson_chain["lesson_id"],
            title="Cascade Test Material",
            material_type="pdf",
        )
        session.add(material)
        await session.flush()

        versions = [
            LearningMaterialVersion(
                material_id=material.id,
                storage_object_id=lesson_chain["storage_object_id"],
                version_no=i,
            )
            for i in range(1, 3)
        ]
        for version in versions:
            session.add(version)
        await session.flush()

        chunks = [
            DocumentChunk(
                course_id=lesson_chain["course_id"],
                module_id=lesson_chain["module_id"],
                lesson_id=lesson_chain["lesson_id"],
                material_version_id=versions[i].id,
                chunk_index=0,
                chunk_type="pdf",
                content=f"chunk content {i}",
                content_hash=f"{'a' * 63}{i}",
            )
            for i in range(2)
        ]
        for chunk in chunks:
            session.add(chunk)
        await session.flush()

        material_id = material.id
        version_ids = [v.id for v in versions]
        chunk_ids = [c.id for c in chunks]
        await session.commit()

    async with session_factory() as session:
        material = await session.get(LearningMaterial, material_id)
        assert material is not None
        result = await soft_delete_cascade(session, material, actor_id=owner_id)
        await session.commit()

    affected_tables = {tbl for (tbl, _id) in result.affected}
    assert affected_tables == {"learning_materials", "learning_material_versions"}
    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert material_id in affected_ids
    assert all(vid in affected_ids for vid in version_ids)
    assert result.count == 3

    async with session_factory() as session:
        deleted_material = (
            await session.execute(
                select(LearningMaterial)
                .where(LearningMaterial.id == material_id)
                .execution_options(include_deleted=True)
            )
        ).scalar_one()
        assert deleted_material.deleted_at is not None
        assert deleted_material.deleted_by == owner_id

        deleted_versions = (
            (
                await session.execute(
                    select(LearningMaterialVersion)
                    .where(LearningMaterialVersion.material_id == material_id)
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(deleted_versions) == 2
        assert all(v.deleted_at is not None for v in deleted_versions)
        assert all(v.deleted_by == owner_id for v in deleted_versions)

        active_material = (
            await session.execute(
                select(LearningMaterial).where(LearningMaterial.id == material_id)
            )
        ).scalar_one_or_none()
        assert active_material is None

        surviving_chunks = (
            (await session.execute(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))))
            .scalars()
            .all()
        )
        assert len(surviving_chunks) == 2
