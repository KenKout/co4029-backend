"""Visibility-checked learner-side query helpers (T4.6 surface).

Two helpers, both intentionally local to the materials feature:

* :func:`list_chunks_preview` — first ``N`` ``DocumentChunk`` rows for the
  current version, used by the quiz-UI source-attribution preview.
* :func:`get_stream_target_for_material` — ``(bucket, object_key, title)``
  for the current ready version's storage object, used by the catalog
  service to mint a presigned GET URL without crossing into the identity
  feature's ORM (the import-linter "Features are independent" contract
  forbids ``materials.* -> identity.models``; the lookup belongs in the
  materials data layer regardless).

Visibility gating: :func:`get_stream_target_for_material` enforces
``visible_to_students = TRUE`` inline and resolves the version through
:mod:`._version_scope` (the current one when ready, else the newest ready
one, so a re-upload in flight does not take the material away from
students); :func:`list_chunks_preview` is fronted by a service-layer
visibility check (per the routers→services→queries discipline) and resolves
the same version, so a preview always describes the file the learner is
actually served. The soft-delete listener (T0.7) auto-applies
``deleted_at IS NULL`` to ORM SELECTs; the raw-SQL ``stream-target`` query
inlines the same predicate itself because the listener only watches ORM
execution.

Soft-delete on chunks: ``DocumentChunk`` is intentionally not a
soft-delete table (T4.1 — "deleted-and-rebuilt on re-ingest"), so its
own ``deleted_at`` column does not exist.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.materials.models import (
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
)
from abridgeai.features.materials.queries._version_scope import learner_version_id


@dataclass
class MaterialStreamTarget:
    """Minimal view used by the service layer to mint a presigned URL.

    Mutable (not ``frozen``) on purpose — :class:`abridgeai.infrastructure.s3.StorageObject`
    is declared as a Protocol with settable members, and a frozen
    dataclass would fail the ``isinstance`` / Protocol fit check at type
    time. The service does not actually mutate the fields.
    """

    bucket: str
    object_key: str
    title: str
    material_version_id: uuid.UUID | None = None


# The readiness and is_current predicates that used to sit in the WHERE clause
# now live in the version subquery, which PICKS the servable version instead of
# asserting that the current one is servable. Spelled out rather than
# interpolated from a shared constant: an f-string here trips S608, and
# silencing an injection lint to save six lines of SQL is a bad trade. The
# ORM twin is `_version_scope.learner_version_id` — the two must agree, and
# `test_materials_queries` exercises both.
_STREAM_TARGET_SQL = text(
    """
    SELECT so.bucket             AS bucket,
           so.object_key         AS object_key,
           lm.title              AS title,
           lmv.id                AS material_version_id
    FROM learning_materials lm
    JOIN learning_material_versions lmv
      ON lmv.id = (
          SELECT v.id
          FROM learning_material_versions v
          WHERE v.material_id = lm.id
            AND v.processing_status = 'ready'
            AND v.deleted_at IS NULL
          ORDER BY v.is_current DESC, v.version_no DESC
          LIMIT 1
      )
    JOIN storage_objects so
      ON so.id = lmv.storage_object_id
    WHERE lm.id = :material_id
      AND lm.deleted_at IS NULL
      AND so.deleted_at IS NULL
      AND lm.visible_to_students = TRUE
    """
)


_AUTHORING_STREAM_TARGET_SQL = text(
    """
    SELECT so.bucket       AS bucket,
           so.object_key   AS object_key,
           lm.title        AS title
    FROM learning_materials lm
    JOIN learning_material_versions lmv
      ON lmv.id = lm.current_version_id
    JOIN storage_objects so
      ON so.id = lmv.storage_object_id
    WHERE lm.id = :material_id
      AND lm.deleted_at IS NULL
      AND lmv.deleted_at IS NULL
      AND so.deleted_at IS NULL
    """
)


async def list_chunks_preview(
    db: AsyncSession, material_id: UUID, *, limit: int
) -> list[DocumentChunk]:
    """Return the first ``limit`` chunks (by ``chunk_index`` ASC) for ``material_id``.

    Resolves the version through :func:`~._version_scope.learner_version_id`
    rather than ``current_version_id``, so the preview describes the same
    version the learner is streamed: historical or pre-rebuild rows for the
    same material id never leak, and a re-upload in flight shows the last
    ready version's chunks instead of nothing. Returns ``[]`` when
    ``limit <= 0``. Visibility gating is the caller's job.
    """
    if limit <= 0:
        return []
    stmt = (
        select(DocumentChunk)
        .join(
            LearningMaterialVersion,
            DocumentChunk.material_version_id == LearningMaterialVersion.id,
        )
        .join(
            LearningMaterial,
            LearningMaterialVersion.id == learner_version_id(LearningMaterial.id),
        )
        .where(LearningMaterial.id == material_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_stream_target_for_material(
    db: AsyncSession, material_id: UUID
) -> MaterialStreamTarget | None:
    """Bucket + object_key + title for a learner-streamable material.

    Returns ``None`` (router maps to 404) when the material is missing,
    soft-deleted, invisible, has never finished processing a single version,
    or that version has no resolvable storage object. A material whose newest
    version is mid-pipeline (or failed) still streams its last ready version.
    Existence MUST NOT leak — the query silently returns ``None`` for every
    disqualifying state.
    """
    result = await db.execute(_STREAM_TARGET_SQL, {"material_id": material_id})
    row = result.one_or_none()
    if row is None:
        return None
    return MaterialStreamTarget(
        bucket=row.bucket,
        object_key=row.object_key,
        title=row.title,
        material_version_id=row.material_version_id,
    )


async def get_authoring_stream_target_for_material(
    db: AsyncSession, material_id: UUID
) -> MaterialStreamTarget | None:
    """Authoring sibling of :func:`get_stream_target_for_material`.

    Drops the learner gates (``visible_to_students=TRUE`` and
    ``processing_status='ready'``) so a teacher can preview hidden /
    mid-pipeline materials during course assembly. Soft-delete and
    ``is_current`` are NOT enforced here — versioning is handled at the
    service layer (the teacher always streams the current version).
    """
    result = await db.execute(_AUTHORING_STREAM_TARGET_SQL, {"material_id": material_id})
    row = result.one_or_none()
    if row is None:
        return None
    return MaterialStreamTarget(bucket=row.bucket, object_key=row.object_key, title=row.title)


__all__ = [
    "MaterialStreamTarget",
    "get_authoring_stream_target_for_material",
    "get_stream_target_for_material",
    "list_chunks_preview",
]
