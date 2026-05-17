"""Raw-SQL writers for ``storage_objects`` and version-state mutations.

Lives in the ``authoring`` package (T4.5) — split from the monolithic
``authoring.py`` for the 800 LOC review cap. The identity feature owns
the ``StorageObject`` ORM; the import-linter "Features are independent"
contract forbids us from importing it. Raw SQL is the documented escape
hatch for the writes that need to land alongside the version row in the
same transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create_storage_object(
    db: AsyncSession,
    *,
    bucket: str,
    object_key: str | None,
    size_bytes: int,
    content_type: str,
    original_filename: str,
    uploaded_by: UUID,
) -> UUID:
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    new_id = uuid4()
    placeholder_key = object_key or f"materials/_pending/{new_id}/{original_filename}"
    await db.execute(
        text(
            """
            INSERT INTO storage_objects
                (id, bucket, object_key, original_filename, mime_type,
                 size_bytes, uploaded_by, uploaded_at, created_at, updated_at)
            VALUES
                (:id, :bucket, :object_key, :original_filename, :mime_type,
                 :size_bytes, :uploaded_by, NOW(), NOW(), NOW())
            """
        ),
        {
            "id": new_id,
            "bucket": bucket,
            "object_key": placeholder_key,
            "original_filename": original_filename,
            "mime_type": content_type,
            "size_bytes": size_bytes,
            "uploaded_by": uploaded_by,
        },
    )
    return new_id


async def set_storage_object_key(
    db: AsyncSession, storage_object_id: UUID, object_key: str
) -> None:
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    await db.execute(
        text("UPDATE storage_objects SET object_key = :k, updated_at = NOW() WHERE id = :id"),
        {"k": object_key, "id": storage_object_id},
    )


async def read_storage_object_declared_size(db: AsyncSession, storage_object_id: UUID) -> int:
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    row = (
        await db.execute(
            text("SELECT size_bytes FROM storage_objects WHERE id = :id"),
            {"id": storage_object_id},
        )
    ).first()
    if row is None or row.size_bytes is None:
        return 0
    return int(row.size_bytes)


async def update_storage_object_metadata(
    db: AsyncSession,
    storage_object_id: UUID,
    *,
    size_bytes: int,
    etag: str | None,
    content_type: str | None,
    checksum: str | None,
) -> None:
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    await db.execute(
        text(
            """
            UPDATE storage_objects
            SET size_bytes = :size,
                checksum_sha256 = COALESCE(:checksum, checksum_sha256),
                mime_type = COALESCE(:ct, mime_type),
                updated_at = NOW()
            WHERE id = :id
            """
        ),
        {
            "size": size_bytes,
            "checksum": checksum,
            "ct": content_type,
            "id": storage_object_id,
        },
    )
    _ = etag  # ETag carried in head_object; baseline ``storage_objects`` has no etag column.


async def reset_other_versions_current(
    db: AsyncSession, material_id: UUID, keep_version_id: UUID
) -> None:
    from sqlalchemy import select  # noqa: PLC0415  -- localised import

    from abridgeai.features.materials.models import (  # noqa: PLC0415  -- localised import
        LearningMaterialVersion,
    )

    rows = (
        (
            await db.execute(
                select(LearningMaterialVersion).where(
                    LearningMaterialVersion.material_id == material_id,
                    LearningMaterialVersion.id != keep_version_id,
                    LearningMaterialVersion.is_current.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    for version in rows:
        version.is_current = False


async def purge_chunks_for_version(db: AsyncSession, version_id: UUID) -> None:
    """Hard-delete every ``document_chunks`` row for ``version_id``.

    ``DocumentChunk`` is intentionally NOT a soft-delete table (T4.1 +
    Reconciliation §C11 — derived data, deleted-and-rebuilt on
    re-ingest). Plain ``DELETE`` is correct here.
    """
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    await db.execute(
        text("DELETE FROM document_chunks WHERE material_version_id = :vid"),
        {"vid": version_id},
    )
