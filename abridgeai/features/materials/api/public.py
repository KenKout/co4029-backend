"""Typed cross-feature read surface for the materials feature.

Module-level async functions only; cap 800 LOC. Returned values are
Pydantic DTOs from :mod:`._dto` — never ORM models — so consumers in
sibling features (``spaced_repetition.services.remediation``,
``progress``, ``admin``, etc.) cannot mutate state nor lazy-load
attributes in async context.

Cross-feature read aggregates wrapped here
------------------------------------------
* Storage-object lookups for a given material / version. The
  ``storage_objects`` table is owned by ``features.identity``; the
  import-linter "Features are independent" contract forbids importing
  the identity ``StorageObject`` ORM. Consumers asked materials "what
  bucket+key+size does this material's blob live at?", not "give me
  the storage row" — the wrapper shape reflects that.
* Chunk → material context resolution for the SR remediation
  notification card (Wave 5 T30): batch lookup of ``document_chunks``
  rows joined to ``learning_material_versions → learning_materials →
  lessons → modules → courses``. Replaces the raw-SQL block in
  ``features/spaced_repetition/services/remediation.py:266-294``.
* Material + lesson-context for progress queries (course slug, lesson
  title, module id) without granting the consumer access to the
  internal ``LearningMaterial`` ORM.
* ``DocumentChunk`` listing for a material (current version only) —
  formerly raw SQL inside the quizzes coverage pipeline; here the
  query goes through the ORM so the soft-delete listener fires for the
  joined version row.
* ``ProcessingJob`` status by job id — used by the admin processing
  dashboard.

Soft-delete behaviour
---------------------
The T0.7 ``with_loader_criteria`` listener auto-applies
``WHERE deleted_at IS NULL`` to every ORM ``SELECT`` against a
``SoftDeleteMixin`` model. ``LearningMaterial`` and
``LearningMaterialVersion`` are soft-delete tables; ``DocumentChunk``,
``ProcessingJob``, and ``storage_objects`` are not. Functions therefore
do **not** manually re-add the predicate for the materials-feature
soft-delete tables; the listener handles it. Raw SQL on
``storage_objects`` and ``lessons``/``modules``/``courses`` (foreign
features) inlines the predicate explicitly because the listener only
watches ORM execution.

What is NOT here
----------------
* Wrappers around the legitimate raw ``DELETE`` on ``document_chunks``
  used by ``services/authoring/_storage.py:purge_chunks_for_version``.
  ``DocumentChunk`` is intentionally not soft-delete (T4.1, "deleted-
  and-rebuilt on re-ingest"); that delete stays internal to the
  feature's services layer.
* Direct ``StorageObject`` row exposure. Consumers that thought they
  needed it actually want bucket+key+size — :class:`StorageBlobInfoDTO`
  ships exactly that.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import bindparam, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.materials.api._dto import (
    DocumentChunkDTO,
    MaterialContextDTO,
    ProcessingJobDTO,
    StorageBlobInfoDTO,
)
from abridgeai.features.materials.models import (
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
    ProcessingJob,
)


_STORAGE_BLOB_BY_VERSION_SQL = text(
    """
    SELECT so.bucket       AS bucket,
           so.object_key   AS object_key,
           so.size_bytes   AS size_bytes
    FROM learning_material_versions lmv
    JOIN storage_objects so ON so.id = lmv.storage_object_id
    WHERE lmv.id = :version_id
      AND lmv.deleted_at IS NULL
      AND so.deleted_at IS NULL
    """
)


_STORAGE_BLOB_BY_MATERIAL_SQL = text(
    """
    SELECT so.bucket       AS bucket,
           so.object_key   AS object_key,
           so.size_bytes   AS size_bytes
    FROM learning_materials lm
    JOIN learning_material_versions lmv
      ON lmv.id = lm.current_version_id
    JOIN storage_objects so ON so.id = lmv.storage_object_id
    WHERE lm.id = :material_id
      AND lm.deleted_at IS NULL
      AND lmv.deleted_at IS NULL
      AND so.deleted_at IS NULL
    """
)


_MATERIAL_CONTEXT_SQL = text(
    """
    SELECT lm.id              AS material_id,
           lm.title           AS material_title,
           lm.material_type   AS material_type,
           lm.current_version_id AS current_version_id,
           l.id               AS lesson_id,
           l.title            AS lesson_title,
           m.id               AS module_id,
           c.id               AS course_id,
           c.slug             AS course_slug
    FROM learning_materials lm
    JOIN lessons l ON l.id = lm.lesson_id
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE lm.id = :material_id
      AND lm.deleted_at IS NULL
      AND l.deleted_at IS NULL
      AND m.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)


_RESOLVE_CHUNKS_SQL = text(
    """
    SELECT DISTINCT ON (lm.id)
           lm.id              AS material_id,
           lm.title           AS material_title,
           lm.material_type   AS material_type,
           lm.current_version_id AS current_version_id,
           l.id               AS lesson_id,
           l.title            AS lesson_title,
           m.id               AS module_id,
           c.id               AS course_id,
           c.slug             AS course_slug,
           dc.id              AS chunk_id
    FROM document_chunks dc
    JOIN learning_material_versions lmv ON lmv.id = dc.material_version_id
    JOIN learning_materials lm ON lm.id = lmv.material_id
    JOIN lessons l ON l.id = lm.lesson_id
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE dc.id = ANY(CAST(:chunk_ids AS uuid[]))
      AND lm.deleted_at IS NULL
      AND lmv.deleted_at IS NULL
      AND l.deleted_at IS NULL
      AND m.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
).bindparams(bindparam("chunk_ids", type_=ARRAY(PG_UUID(as_uuid=True))))


async def get_storage_blob_for_version(
    db: AsyncSession, version_id: UUID
) -> StorageBlobInfoDTO | None:
    """Return ``(bucket, object_key, size_bytes)`` for a version's blob.

    Joins ``learning_material_versions → storage_objects`` and inlines
    ``deleted_at IS NULL`` on both tables (raw SQL bypasses the ORM
    soft-delete listener; ``storage_objects`` is identity-owned and not
    importable from materials regardless). Returns ``None`` when the
    version is missing, soft-deleted, or its storage row is gone.
    """
    row = (await db.execute(_STORAGE_BLOB_BY_VERSION_SQL, {"version_id": version_id})).first()
    if row is None:
        return None
    return StorageBlobInfoDTO(
        bucket=row.bucket, object_key=row.object_key, size_bytes=int(row.size_bytes)
    )


async def get_storage_size_and_key(
    db: AsyncSession, *, material_id: UUID
) -> tuple[int, str] | None:
    """Return ``(size_bytes, object_key)`` for a material's current blob.

    Wraps the cross-feature aggregate so consumers never reach the
    identity-owned ``storage_objects`` table directly. Walks
    ``learning_materials.current_version_id → learning_material_versions
    → storage_objects``. Returns ``None`` for missing / soft-deleted /
    versionless materials. Tuple ordering matches the legacy callers'
    ``size, key`` unpack convention.
    """
    row = (await db.execute(_STORAGE_BLOB_BY_MATERIAL_SQL, {"material_id": material_id})).first()
    if row is None:
        return None
    return int(row.size_bytes), row.object_key


async def get_material_with_lesson_context(
    db: AsyncSession, material_id: UUID
) -> MaterialContextDTO | None:
    """Material + lesson + module + course context for ``material_id``.

    Used by progress queries that need the lesson title and course slug
    alongside the material id. Returns ``None`` when any link in the
    chain is missing or soft-deleted (no existence-leak distinction
    between "no such material" and "deleted material" — both are
    ``None``).
    """
    row = (await db.execute(_MATERIAL_CONTEXT_SQL, {"material_id": material_id})).first()
    if row is None:
        return None
    return MaterialContextDTO(
        material_id=_as_uuid(row.material_id),
        material_title=row.material_title,
        material_type=row.material_type,
        current_version_id=_as_uuid_or_none(row.current_version_id),
        lesson_id=_as_uuid(row.lesson_id),
        lesson_title=row.lesson_title,
        module_id=_as_uuid(row.module_id),
        course_id=_as_uuid(row.course_id),
        course_slug=row.course_slug,
    )


async def resolve_chunks_to_materials(
    db: AsyncSession, chunk_ids: Iterable[UUID]
) -> dict[UUID, MaterialContextDTO]:
    """Resolve chunk UUIDs → owning :class:`MaterialContextDTO` per material.

    Replaces the raw-SQL fan-out in
    ``features/spaced_repetition/services/remediation.py:266-294`` with
    a single typed call. The result is keyed by **material_id**, not
    chunk_id, because the SR remediation notification card de-dupes one
    deep-link per material — many wrong-answer chunks map to the same
    material. ``DISTINCT ON (lm.id)`` enforces the de-dup server-side.

    Empty input → empty dict (cheap fast-path that avoids the round-trip
    entirely).
    """
    chunk_id_list = [str(cid) for cid in chunk_ids]
    if not chunk_id_list:
        return {}
    rows = (
        await db.execute(_RESOLVE_CHUNKS_SQL, {"chunk_ids": chunk_id_list})
    ).mappings().all()
    out: dict[UUID, MaterialContextDTO] = {}
    for row in rows:
        material_id = _as_uuid(row["material_id"])
        out[material_id] = MaterialContextDTO(
            material_id=material_id,
            material_title=row["material_title"],
            material_type=row["material_type"],
            current_version_id=_as_uuid_or_none(row["current_version_id"]),
            lesson_id=_as_uuid(row["lesson_id"]),
            lesson_title=row["lesson_title"],
            module_id=_as_uuid(row["module_id"]),
            course_id=_as_uuid(row["course_id"]),
            course_slug=row["course_slug"],
        )
    return out


async def get_document_chunks_by_material(
    db: AsyncSession, material_id: UUID
) -> list[DocumentChunkDTO]:
    """Return every :class:`DocumentChunk` row for ``material_id``'s current version.

    Joins through ``LearningMaterial.current_version_id`` so historical /
    pre-rebuild chunk rows for the same material do not leak. Sorted by
    ``chunk_index`` ascending. Empty list when the material is missing,
    soft-deleted, has no current version, or has no chunks yet.

    The query goes through the ORM so the T0.7 soft-delete listener auto-
    filters the joined ``LearningMaterial`` / ``LearningMaterialVersion``
    rows. ``DocumentChunk`` itself is not a soft-delete table.
    """
    stmt = (
        select(DocumentChunk)
        .join(
            LearningMaterialVersion,
            DocumentChunk.material_version_id == LearningMaterialVersion.id,
        )
        .join(
            LearningMaterial,
            LearningMaterial.current_version_id == LearningMaterialVersion.id,
        )
        .where(LearningMaterial.id == material_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        DocumentChunkDTO(
            chunk_id=row.id,
            material_version_id=row.material_version_id,
            course_id=row.course_id,
            module_id=row.module_id,
            lesson_id=row.lesson_id,
            chunk_index=row.chunk_index,
            chunk_type=row.chunk_type,
            content=row.content,
            content_hash=row.content_hash,
            metadata=dict(row.metadata_json or {}),
        )
        for row in rows
    ]


async def get_processing_job_status(
    db: AsyncSession, job_id: UUID
) -> ProcessingJobDTO | None:
    """Return the :class:`ProcessingJobDTO` for ``job_id`` or ``None``.

    Used by the admin processing dashboard. ``ProcessingJob`` is hard-
    deleted (no ``deleted_at`` column / mixin), so the soft-delete
    listener does not apply.
    """
    job = await db.get(ProcessingJob, job_id)
    if job is None:
        return None
    return ProcessingJobDTO(
        job_id=job.id,
        entity_type=job.entity_type,
        entity_id=job.entity_id,
        job_type=job.job_type,
        status=job.status,
        progress_percent=job.progress_percent,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        retry_count=job.retry_count,
    )


def _as_uuid(value: object) -> UUID:
    """Coerce ``str | UUID`` from raw-SQL row mappings to ``UUID``.

    psycopg returns ``uuid.UUID`` for ``uuid``-typed columns; some
    drivers / cast paths produce ``str``. The DTO field is typed
    ``UUID``, so coerce uniformly.
    """
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _as_uuid_or_none(value: object) -> UUID | None:
    if value is None:
        return None
    return _as_uuid(value)


__all__ = [
    "get_document_chunks_by_material",
    "get_material_with_lesson_context",
    "get_processing_job_status",
    "get_storage_blob_for_version",
    "get_storage_size_and_key",
    "resolve_chunks_to_materials",
]
