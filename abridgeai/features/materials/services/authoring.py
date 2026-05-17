"""Teacher-side authoring service for the materials feature (T4.5).

Owns the direct-upload lifecycle (single-shot + multipart), version
register-on-complete with mandatory ``head_object`` verification (the
phantom-complete attack guard from plan §4934 + Reconciliation §C9),
soft-delete that preserves the underlying S3 object (plan §4946 +
§4954 — recovery requires retention), and reprocess with a chunk
purge + concurrency check (Reconciliation §C13).

Architectural rules honoured:

* ``services -> sqlalchemy`` import-linter contract (T0.4) — this
  module imports ``AsyncSession`` only under :data:`TYPE_CHECKING`.
  Runtime data access goes through ``queries.*``; the rare write
  path that needs raw ORM (``DocumentChunk`` purge on reprocess)
  uses :func:`abridgeai.core.db.recursive_delete.soft_delete_cascade`
  and the queries module.
* Routers→services boundary — every router endpoint composes a
  helper here; no direct ``queries.*`` access from
  :mod:`features.materials.routers.authoring`.
* Service layer flushes; the router commits. ARQ jobs are enqueued
  AFTER the flush so the version + processing_job rows are visible
  in DB once the job dequeues.

§C9 redesign — direct-upload flow rewrite
------------------------------------------
* :func:`init_upload` chooses single vs multipart based on
  ``size_bytes`` (>100MB → multipart). Multipart pre-mints the first
  ``min(part_count, 100)`` part URLs; the router exposes
  :func:`fetch_multipart_parts` for the next batch.
* :func:`complete_upload` calls
  :func:`abridgeai.infrastructure.s3.head_object` to verify the bytes
  actually landed (404 if missing, 400 if zero-byte), then flips
  ``processing_status='pending'`` (the ingest worker's entry state),
  marks the version current, and enqueues the ARQ task.

§C13 reprocess invariants
--------------------------
* :func:`reprocess_material` returns ``RuntimeError("concurrent_reprocess")``
  when ``get_latest_processing_job`` reports ``pending`` / ``running``;
  the router maps to HTTP 409.
* On reprocess, every ``document_chunks`` row for the version is
  DELETEd outright (chunks are derived data, no soft-delete per
  T4.1 / Reconciliation §C11) before the new ARQ job lands.

FIX-SEC-1 supporting helper
---------------------------
The router uses T3.7's ``require_lesson_authoring_access`` for
lesson-scoped endpoints. For material-scoped endpoints the router
walks ``material_id → lesson_id → course_id`` and calls
``require_course_permission("course.update")``; this service exposes
:func:`resolve_course_id_for_material` so the router doesn't reach
into the queries package for that single read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.materials.models import (
    LearningMaterial,
    LearningMaterialVersion,
    ProcessingJob,
)
from abridgeai.features.materials.queries import (
    get_latest_processing_job,
    get_material_for_authoring,
    list_all_materials,
)
from abridgeai.features.materials.schemas import (
    MaterialAuthoring,
    MaterialUpdate,
    MaterialUploadComplete,
    MaterialUploadInit,
    MaterialVersionAuthoring,
    ProcessingProgress,
)
from abridgeai.features.materials.schemas.public import MaterialTypeLiteral
from abridgeai.infrastructure.s3 import (
    CompletedPart as S3CompletedPart,
)
from abridgeai.infrastructure.s3 import (
    abort_multipart_upload,
    complete_multipart_upload,
    create_multipart_upload,
    create_upload_url,
    head_object,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Plan §4927 — single-shot threshold. Multipart for anything bigger.
_MULTIPART_THRESHOLD_BYTES: int = 100 * 1024 * 1024
_MULTIPART_PART_BYTES: int = 10 * 1024 * 1024  # 10 MB part size (S3 minimum is 5 MB).
_MULTIPART_FIRST_BATCH_CAP: int = 100  # mint up to N URLs in init; client paginates.

# Plan §4935 — zero-byte uploads are rejected.
_MIN_ACCEPTABLE_BYTES: int = 1
# Plan §4935 — accept ±1% slack vs the size declared at init. Network
# round-tripping + S3's accounting can produce a few-byte delta on the
# tail block; declare the tolerance up front so reviewers don't argue
# over magic numbers.
_SIZE_TOLERANCE_FRACTION: float = 0.01

# MIME → material_type fallback when the client omits material_type at init.
_MIME_TO_TYPE: dict[str, MaterialTypeLiteral] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "text/plain": "text",
    "text/markdown": "text",
}
_MIME_PREFIX_TO_TYPE: tuple[tuple[str, MaterialTypeLiteral], ...] = (
    ("video/", "video"),
    ("audio/", "audio"),
    ("image/", "image"),
)


# ---------------------------------------------------------------------------
# Response DTOs (router serialises these directly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MultipartPart:
    part_number: int
    url: str


@dataclass(frozen=True)
class MaterialUploadInitResponse:
    """Discriminated response returned by :func:`init_upload`.

    ``mode='single'`` carries ``upload_url``; ``mode='multipart'``
    carries ``upload_id`` + the first batch of presigned part URLs.
    The router serialises this dataclass directly — Pydantic was kept
    out of the service layer to avoid a circular import dance with
    schemas (request schemas already import from public schemas).
    """

    material_id: UUID
    version_id: UUID
    storage_object_id: UUID
    mode: Literal["single", "multipart"]
    expires_at: datetime
    upload_url: str | None = None
    upload_id: str | None = None
    part_count: int | None = None
    part_size_bytes: int | None = None
    parts: tuple[_MultipartPart, ...] | None = None


@dataclass(frozen=True)
class MultipartPartsResponse:
    parts: tuple[_MultipartPart, ...]
    expires_at: datetime


@dataclass(frozen=True)
class CompletedPartIn:
    part_number: int
    etag: str


@dataclass(frozen=True)
class UploadCompleteResponse:
    material_id: UUID
    version_id: UUID
    processing_job_id: UUID
    pipeline_run_id: UUID


@dataclass(frozen=True)
class ReprocessResponse:
    material_id: UUID
    version_id: UUID
    processing_job_id: UUID
    pipeline_run_id: UUID


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_material_type(
    declared: MaterialTypeLiteral | None, content_type: str
) -> MaterialTypeLiteral:
    """Return ``declared`` if present, otherwise infer from MIME.

    Falls back to ``'text'`` when no rule matches; the ingest pipeline
    still runs for unknown types (it picks the appropriate extractor
    or fails loudly) so the catch-all is safer than a hard reject at
    init time.
    """
    if declared is not None:
        return declared
    lowered = content_type.lower()
    if lowered in _MIME_TO_TYPE:
        return _MIME_TO_TYPE[lowered]
    for prefix, mtype in _MIME_PREFIX_TO_TYPE:
        if lowered.startswith(prefix):
            return mtype
    return "text"


def _compute_part_count(size_bytes: int) -> int:
    return max(1, math.ceil(size_bytes / _MULTIPART_PART_BYTES))


def _present(version: LearningMaterialVersion) -> MaterialVersionAuthoring:
    return MaterialVersionAuthoring.model_validate(version)


async def _require_material(db: AsyncSession, material_id: UUID) -> LearningMaterial:
    material = await get_material_for_authoring(db, material_id)
    if material is None:
        raise NotFoundError(f"Material {material_id} not found")
    return material


async def _require_version(db: AsyncSession, version_id: UUID) -> LearningMaterialVersion:
    # Avoid the soft-delete listener gating on a write path: the router
    # explicitly wants the row even if soft-deleted hasn't been purged
    # yet. ``db.get`` participates in the listener anyway, so we use it
    # for read-side guard only; reprocess / complete flows reach
    # additionally-fresh rows.
    version = await db.get(LearningMaterialVersion, version_id)
    if version is None:
        raise NotFoundError(f"Material version {version_id} not found")
    return version


async def _resolve_storage_view(db: AsyncSession, version: LearningMaterialVersion) -> _StorageView:
    """Return ``(bucket, object_key)`` for the version's storage row.

    Raw SQL keeps us off the identity feature's ORM (the import-linter
    "Features are independent" contract) — only ``bucket`` /
    ``object_key`` are needed for the S3 helpers.
    """
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    row = (
        await db.execute(
            text(
                "SELECT bucket, object_key FROM storage_objects "
                "WHERE id = :id AND deleted_at IS NULL"
            ),
            {"id": version.storage_object_id},
        )
    ).first()
    if row is None:
        raise NotFoundError(
            f"Storage object {version.storage_object_id} for version {version.id} not found"
        )
    return _StorageView(bucket=row.bucket, object_key=row.object_key)


@dataclass
class _StorageView:
    """Duck-typed S3 ``StorageObject`` Protocol fit (only bucket+object_key).

    Non-frozen so the writable Protocol on
    :class:`abridgeai.infrastructure.s3.StorageObject` accepts the type
    (matches the pattern from
    :class:`abridgeai.features.materials.queries.chunks.MaterialStreamTarget`
    and :class:`abridgeai.features.materials.workers.ingest._StorageView`).
    The service never mutates the fields.
    """

    bucket: str
    object_key: str


async def resolve_course_id_for_material(db: AsyncSession, material_id: UUID) -> UUID | None:
    """Walk ``material → lesson → module → course``.

    Used by the router to derive the ``course_id`` for the
    ``require_course_permission("course.update")`` guard on
    material-scoped endpoints (FIX-SEC-1 perimeter).
    Returns ``None`` if the chain is broken or any row is soft-deleted.
    """
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    row = (
        await db.execute(
            text(
                """
                SELECT m.course_id AS course_id
                FROM learning_materials lm
                JOIN lessons l ON l.id = lm.lesson_id
                JOIN modules m ON m.id = l.module_id
                JOIN courses c ON c.id = m.course_id
                WHERE lm.id = :id
                  AND lm.deleted_at IS NULL
                  AND l.deleted_at IS NULL
                  AND m.deleted_at IS NULL
                  AND c.deleted_at IS NULL
                """
            ),
            {"id": material_id},
        )
    ).first()
    return None if row is None else row.course_id


# ---------------------------------------------------------------------------
# Read helpers (composed by router GETs)
# ---------------------------------------------------------------------------


async def list_authoring_materials(
    db: AsyncSession, lesson_id: UUID, *, include_archived: bool = False
) -> list[MaterialAuthoring]:
    materials = await list_all_materials(db, lesson_id, include_archived=include_archived)
    return [await _present_material(db, m) for m in materials]


async def get_authoring_material(db: AsyncSession, material_id: UUID) -> MaterialAuthoring | None:
    material = await get_material_for_authoring(db, material_id)
    if material is None:
        return None
    return await _present_material(db, material)


async def _present_material(db: AsyncSession, material: LearningMaterial) -> MaterialAuthoring:
    """Hydrate :class:`MaterialAuthoring` (latest version + counts)."""
    from sqlalchemy import func, select  # noqa: PLC0415  -- localised raw escape hatch

    version_count = (
        await db.execute(
            select(func.count(LearningMaterialVersion.id)).where(
                LearningMaterialVersion.material_id == material.id
            )
        )
    ).scalar_one()
    latest_version: LearningMaterialVersion | None = None
    if material.current_version_id is not None:
        latest_version = await db.get(LearningMaterialVersion, material.current_version_id)

    return MaterialAuthoring.model_validate(material).model_copy(
        update={
            "version_count": int(version_count),
            "latest_version": _present(latest_version) if latest_version is not None else None,
        }
    )


async def get_processing_progress(db: AsyncSession, material_id: UUID) -> ProcessingProgress | None:
    """Return the latest version's progress slice (or ``None``)."""
    material = await get_material_for_authoring(db, material_id)
    if material is None or material.current_version_id is None:
        return None
    version = await db.get(LearningMaterialVersion, material.current_version_id)
    if version is None:
        return None
    job = await get_latest_processing_job(db, version.id)
    return ProcessingProgress(
        material_id=material.id,
        version_id=version.id,
        processing_status=version.processing_status,
        progress_percent=int(job.progress_percent) if job is not None else 0,
        latest_log_line=None,
        error_message=version.processing_error,
    )


async def update_material(
    db: AsyncSession,
    material_id: UUID,
    payload: MaterialUpdate,
    actor: CurrentUser,
) -> MaterialAuthoring:
    del actor  # audit listener writes updated_by
    material = await _require_material(db, material_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(material, key, value)
    await db.flush()
    await db.refresh(material)
    return await _present_material(db, material)


# ---------------------------------------------------------------------------
# Upload init (single + multipart) — plan §4925-4929
# ---------------------------------------------------------------------------


async def init_upload(
    db: AsyncSession,
    lesson_id: UUID,
    payload: MaterialUploadInit,
    actor: CurrentUser,
) -> MaterialUploadInitResponse:
    """Create material + version (pending) + storage row, mint upload URL(s)."""
    settings = get_settings()
    material_type = _resolve_material_type(payload.material_type, payload.content_type)

    material = LearningMaterial(
        lesson_id=lesson_id,
        title=payload.title,
        material_type=material_type,
        ai_processing_enabled=True,
        visible_to_students=False,
    )
    db.add(material)
    await db.flush()
    await db.refresh(material)

    storage_object_id = await _create_storage_object(
        db,
        bucket=settings.s3_bucket_name,
        # ``materials/{material_id}/{version_id_placeholder}/{filename}`` —
        # version_id is back-filled below once the version row has an id.
        object_key=None,
        size_bytes=max(0, int(payload.size_bytes)),
        content_type=payload.content_type,
        original_filename=payload.filename,
        uploaded_by=actor.user_id,
    )

    version = LearningMaterialVersion(
        material_id=material.id,
        storage_object_id=storage_object_id,
        version_no=1,
        is_current=False,
        processing_status="pending",
        uploaded_by=actor.user_id,
        uploaded_at=datetime.now(tz=UTC),
    )
    db.add(version)
    await db.flush()
    await db.refresh(version)

    object_key = f"materials/{material.id}/{version.id}/{payload.filename}"
    await _set_storage_object_key(db, storage_object_id, object_key)

    storage_view = _StorageView(bucket=settings.s3_bucket_name, object_key=object_key)

    if payload.size_bytes > _MULTIPART_THRESHOLD_BYTES:
        part_count = _compute_part_count(payload.size_bytes)
        first_batch = min(part_count, _MULTIPART_FIRST_BATCH_CAP)
        init = await create_multipart_upload(
            storage_view, part_count=first_batch, content_type=payload.content_type
        )
        expires_at = datetime.now(tz=UTC) + timedelta(seconds=settings.s3_url_ttl_seconds)
        return MaterialUploadInitResponse(
            material_id=material.id,
            version_id=version.id,
            storage_object_id=storage_object_id,
            mode="multipart",
            expires_at=expires_at,
            upload_id=init.upload_id,
            part_count=part_count,
            part_size_bytes=_MULTIPART_PART_BYTES,
            parts=tuple(_MultipartPart(part_number=p.part_number, url=p.url) for p in init.parts),
        )

    upload_url, expires_at = await create_upload_url(
        storage_view, content_type=payload.content_type
    )
    return MaterialUploadInitResponse(
        material_id=material.id,
        version_id=version.id,
        storage_object_id=storage_object_id,
        mode="single",
        expires_at=expires_at,
        upload_url=upload_url,
    )


async def fetch_multipart_parts(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    upload_id: str,
    *,
    part_from: int,
    part_count: int,
) -> MultipartPartsResponse:
    """Mint a follow-up batch of presigned part URLs.

    Defensive validation only — caller must already have passed the
    course-perm check at the router. ``part_from`` is 1-indexed (S3
    convention); ``part_count`` is capped at
    :data:`_MULTIPART_FIRST_BATCH_CAP`.
    """
    if part_from < 1:
        raise AppError("part_from must be >= 1")
    if part_count < 1 or part_count > _MULTIPART_FIRST_BATCH_CAP:
        raise AppError(f"part_count must be in [1, {_MULTIPART_FIRST_BATCH_CAP}]")

    version = await _require_version(db, version_id)
    if version.material_id != material_id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    storage_view = await _resolve_storage_view(db, version)

    # Re-issue PRESIGN-only calls; the upload_id was minted by init_upload.
    # ``create_multipart_upload`` re-creates the upload — instead we presign
    # directly via aioboto3 with the existing upload_id.
    parts = await _presign_existing_multipart(
        storage_view, upload_id=upload_id, start=part_from, count=part_count
    )
    settings = get_settings()
    expires_at = datetime.now(tz=UTC) + timedelta(seconds=settings.s3_url_ttl_seconds)
    return MultipartPartsResponse(
        parts=tuple(_MultipartPart(part_number=pn, url=u) for pn, u in parts),
        expires_at=expires_at,
    )


async def _presign_existing_multipart(
    storage_view: _StorageView,
    *,
    upload_id: str,
    start: int,
    count: int,
) -> list[tuple[int, str]]:
    """Presign ``count`` consecutive part URLs starting at ``start`` for ``upload_id``.

    T2.3's ``s3.py`` only exposes ``create_multipart_upload`` (which
    initialises a new upload). For the second batch we presign
    ``upload_part`` directly via aioboto3 — a one-off escape hatch
    documented for a follow-up ``s3.presign_multipart_parts`` helper.
    """
    import aioboto3  # noqa: PLC0415  -- localised at the call site

    settings = get_settings()
    endpoint = settings.aws_public_endpoint_url or settings.aws_endpoint_url
    access_key = settings.aws_access_key_id
    secret_key = settings.aws_secret_access_key
    if access_key is None or secret_key is None:
        from abridgeai.infrastructure.errors import S3NotConfiguredError  # noqa: PLC0415

        raise S3NotConfiguredError("S3 credentials are not configured")
    expires_in = settings.s3_url_ttl_seconds

    session = aioboto3.Session()
    parts: list[tuple[int, str]] = []
    async with session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key.get_secret_value(),
        aws_secret_access_key=secret_key.get_secret_value(),
        region_name=settings.aws_region,
    ) as client:
        for offset in range(count):
            part_number = start + offset
            url = await client.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": storage_view.bucket,
                    "Key": storage_view.object_key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=expires_in,
            )
            parts.append((part_number, url))
    return parts


# ---------------------------------------------------------------------------
# Multipart finalisation
# ---------------------------------------------------------------------------


async def complete_multipart(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    upload_id: str,
    completed_parts: list[CompletedPartIn],
) -> None:
    """Wrap S3 ``CompleteMultipartUpload``. Caller follows up with :func:`complete_upload`."""
    version = await _require_version(db, version_id)
    if version.material_id != material_id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    storage_view = await _resolve_storage_view(db, version)
    s3_parts = [S3CompletedPart(part_number=p.part_number, etag=p.etag) for p in completed_parts]
    await complete_multipart_upload(storage_view, upload_id, s3_parts)


async def abort_multipart(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    upload_id: str,
) -> None:
    """Cancel a multipart upload + mark the version cancelled."""
    version = await _require_version(db, version_id)
    if version.material_id != material_id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    storage_view = await _resolve_storage_view(db, version)
    await abort_multipart_upload(storage_view, upload_id)
    version.processing_status = "cancelled"
    await db.flush()


# ---------------------------------------------------------------------------
# /complete — register version, head_object verify, enqueue ARQ
# ---------------------------------------------------------------------------


class HeadVerificationError(AppError):
    """``head_object`` rejected the upload (missing / zero-byte / size mismatch)."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


async def complete_upload(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    payload: MaterialUploadComplete,
    actor: CurrentUser,
    *,
    arq_pool: object | None,
) -> UploadCompleteResponse:
    """Verify the upload landed in S3 and enqueue the ingest task.

    Steps (plan §4933-4939, security-critical):

    1. Fetch the version + the storage row and call
       :func:`abridgeai.infrastructure.s3.head_object`. ``None`` →
       :class:`HeadVerificationError` with status 404 (phantom-complete
       attack guard).
    2. Reject zero-byte uploads (``size < 1``) → 400.
    3. Reject obvious size lies (>1% delta vs declared at init).
    4. Update storage row (size, etag, content_type) and the version
       row (processing_status='pending', mime, size, checksum). Mark
       the version current; set
       :attr:`LearningMaterial.current_version_id`.
    5. Generate ``pipeline_run_id``, create a :class:`ProcessingJob`,
       enqueue ``ingest_material_version_task``.
    """
    material = await _require_material(db, material_id)
    version = await _require_version(db, version_id)
    if version.material_id != material.id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")

    storage_view = await _resolve_storage_view(db, version)
    meta = await head_object(storage_view)
    if meta is None:
        raise HeadVerificationError(
            "Object not found in storage; upload may have failed",
            status_code=404,
        )
    if meta.size < _MIN_ACCEPTABLE_BYTES:
        raise HeadVerificationError("Empty file rejected", status_code=400)

    declared = await _read_storage_object_declared_size(db, version.storage_object_id)
    if declared > 0:
        slack = max(1, int(declared * _SIZE_TOLERANCE_FRACTION))
        if abs(meta.size - declared) > slack:
            raise HeadVerificationError(
                f"Size mismatch: declared={declared}, observed={meta.size}",
                status_code=400,
            )

    await _update_storage_object_metadata(
        db,
        version.storage_object_id,
        size_bytes=meta.size,
        etag=meta.etag,
        content_type=meta.content_type,
        checksum=payload.checksum_sha256,
    )

    # Reset prior ``is_current`` rows for this material before flipping
    # the current version (Reconciliation §C15 dual-source invariant).
    await _reset_other_versions_current(db, material.id, version.id)
    version.is_current = True
    version.processing_status = "pending"
    version.processed_at = None
    material.current_version_id = version.id
    await db.flush()

    pipeline_run_id = uuid4()
    job = ProcessingJob(
        entity_type="material_version",
        entity_id=version.id,
        job_type="full_pipeline",
        status="pending",
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            "ingest_material_version_task",
            actor.user_id,
            version.id,
            pipeline_run_id,
        )

    return UploadCompleteResponse(
        material_id=material.id,
        version_id=version.id,
        processing_job_id=job.id,
        pipeline_run_id=pipeline_run_id,
    )


# ---------------------------------------------------------------------------
# Reprocess — Reconciliation §C13
# ---------------------------------------------------------------------------


class ConcurrentReprocessError(AppError):
    """A previous ingest job for this version is still in flight (409)."""


async def reprocess_material(
    db: AsyncSession,
    material_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None,
) -> ReprocessResponse:
    """Clear chunks + enqueue a fresh ingest for the current version.

    Refuses (raises :class:`ConcurrentReprocessError`) when the latest
    :class:`ProcessingJob` is still ``pending`` / ``running``. The
    router maps the exception to HTTP 409.
    """
    material = await _require_material(db, material_id)
    if material.current_version_id is None:
        raise NotFoundError(f"Material {material_id} has no current version to reprocess")
    version = await _require_version(db, material.current_version_id)

    latest_job = await get_latest_processing_job(db, version.id)
    if latest_job is not None and latest_job.status in {"pending", "running"}:
        raise ConcurrentReprocessError(
            f"Material {material_id} has an in-progress reprocess job ({latest_job.id})"
        )

    await _purge_chunks_for_version(db, version.id)

    version.processing_status = "pending"
    version.processing_error = None
    version.processed_at = None
    await db.flush()

    pipeline_run_id = uuid4()
    job = ProcessingJob(
        entity_type="material_version",
        entity_id=version.id,
        job_type="full_pipeline",
        status="pending",
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            "ingest_material_version_task",
            actor.user_id,
            version.id,
            pipeline_run_id,
        )

    return ReprocessResponse(
        material_id=material.id,
        version_id=version.id,
        processing_job_id=job.id,
        pipeline_run_id=pipeline_run_id,
    )


# ---------------------------------------------------------------------------
# Soft-delete — preserves S3 object (plan §4946 + §4954)
# ---------------------------------------------------------------------------


async def soft_delete_material(db: AsyncSession, material_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete the material + cascade to versions. Does NOT touch S3."""
    material = await _require_material(db, material_id)
    await soft_delete_cascade(db, material, actor_id=actor.user_id)


# ---------------------------------------------------------------------------
# Storage-object writer helpers (raw SQL — features.identity ORM is
# off-limits per the import-linter "Features are independent" contract).
# ---------------------------------------------------------------------------


async def _create_storage_object(
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


async def _set_storage_object_key(
    db: AsyncSession, storage_object_id: UUID, object_key: str
) -> None:
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    await db.execute(
        text("UPDATE storage_objects SET object_key = :k, updated_at = NOW() WHERE id = :id"),
        {"k": object_key, "id": storage_object_id},
    )


async def _read_storage_object_declared_size(db: AsyncSession, storage_object_id: UUID) -> int:
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


async def _update_storage_object_metadata(
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


async def _reset_other_versions_current(
    db: AsyncSession, material_id: UUID, keep_version_id: UUID
) -> None:
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    await db.execute(
        text(
            """
            UPDATE learning_material_versions
            SET is_current = FALSE
            WHERE material_id = :mid AND id <> :vid
            """
        ),
        {"mid": material_id, "vid": keep_version_id},
    )


async def _purge_chunks_for_version(db: AsyncSession, version_id: UUID) -> None:
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


__all__ = [
    "CompletedPartIn",
    "ConcurrentReprocessError",
    "HeadVerificationError",
    "MaterialUploadInitResponse",
    "MultipartPartsResponse",
    "ReprocessResponse",
    "UploadCompleteResponse",
    "abort_multipart",
    "complete_multipart",
    "complete_upload",
    "fetch_multipart_parts",
    "get_authoring_material",
    "get_processing_progress",
    "init_upload",
    "list_authoring_materials",
    "reprocess_material",
    "resolve_course_id_for_material",
    "soft_delete_material",
    "update_material",
]
