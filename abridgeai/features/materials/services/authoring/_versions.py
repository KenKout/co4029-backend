"""Version finalisation, reprocess, and soft-delete.

Splits the version-state mutators out of the monolithic ``authoring.py``
for the 800 LOC review cap. Covers:

* :func:`complete_upload` — head_object verification (the
  phantom-complete attack guard from plan §4934 + Reconciliation §C9),
  storage-metadata write, current-version flip, ARQ enqueue.
* :func:`reprocess_material` — Reconciliation §C13: refuse-on-busy,
  chunk purge, fresh ingest job + ARQ enqueue.
* :func:`soft_delete_material` — preserves S3 (plan §4946 + §4954).

Service-layer exceptions (:class:`HeadVerificationError`,
:class:`ConcurrentReprocessError`) live in :mod:`._common`; routers
map them to HTTP status codes themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from abridgeai.ai.models import ProcessingJob
from abridgeai.core.db.conflict_mapper import flush_or_conflict, register_conflict_mappings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.materials.queries import (
    get_latest_processing_job,
    get_material_with_versions,
    restore_soft_deleted_material,
)
from abridgeai.features.materials.schemas import (
    MaterialUploadComplete,
    MaterialVersionAuthoring,
)
from abridgeai.features.materials.services.authoring._common import (
    MIN_ACCEPTABLE_BYTES,
    SIZE_TOLERANCE_FRACTION,
    ConcurrentReprocessError,
    HeadVerificationError,
    ReprocessResponse,
    UploadCompleteResponse,
    require_material,
    require_version,
    resolve_storage_view,
)
from abridgeai.features.materials.services.authoring._storage import (
    purge_chunks_for_version,
    read_storage_object_declared_size,
    reset_other_versions_current,
    update_storage_object_metadata,
)
from abridgeai.features.materials.workers.enqueue import enqueue_material_ingest
from abridgeai.infrastructure.s3 import head_object

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# §C15 backstop (migration 0017): concurrent current-version flips
# (complete_upload vs rollback) hit the partial unique index instead of
# silently leaving two is_current rows; flush_or_conflict maps it to 409.
register_conflict_mappings(
    {
        "uq_learning_material_versions_one_current": (
            "Another operation is changing this material's current version; retry"
        ),
    }
)


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
    material = await require_material(db, material_id)
    version = await require_version(db, version_id)
    if version.material_id != material.id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")

    storage_view = await resolve_storage_view(db, version)
    meta = await head_object(storage_view)
    if meta is None:
        raise HeadVerificationError(
            "Object not found in storage; upload may have failed",
            status_code=404,
        )
    if meta.size < MIN_ACCEPTABLE_BYTES:
        raise HeadVerificationError("Empty file rejected", status_code=400)

    declared = await read_storage_object_declared_size(db, version.storage_object_id)
    if declared > 0:
        slack = max(1, int(declared * SIZE_TOLERANCE_FRACTION))
        if abs(meta.size - declared) > slack:
            raise HeadVerificationError(
                f"Size mismatch: declared={declared}, observed={meta.size}",
                status_code=400,
            )

    await update_storage_object_metadata(
        db,
        version.storage_object_id,
        size_bytes=meta.size,
        etag=meta.etag,
        content_type=meta.content_type,
        checksum=payload.checksum_sha256,
    )

    # Reset prior ``is_current`` rows for this material before flipping
    # the current version (Reconciliation §C15 dual-source invariant).
    await reset_other_versions_current(db, material.id, version.id)
    version.is_current = True
    version.processing_status = "pending"
    version.processed_at = None
    material.current_version_id = version.id
    await flush_or_conflict(db)

    pipeline_run_id = uuid4()
    job = ProcessingJob(
        entity_type="material_version",
        entity_id=version.id,
        job_type="full_pipeline",
        status="pending",
    )
    db.add(job)
    await flush_or_conflict(db)
    await db.refresh(job)

    await enqueue_material_ingest(
        arq_pool,
        actor_id=actor.user_id,
        material_version_id=version.id,
        pipeline_run_id=pipeline_run_id,
    )

    return UploadCompleteResponse(
        material_id=material.id,
        version_id=version.id,
        processing_job_id=job.id,
        pipeline_run_id=pipeline_run_id,
    )


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
    material = await require_material(db, material_id)
    if material.current_version_id is None:
        raise NotFoundError(f"Material {material_id} has no current version to reprocess")
    version = await require_version(db, material.current_version_id)

    latest_job = await get_latest_processing_job(db, version.id)
    if latest_job is not None and latest_job.status in {"pending", "running"}:
        raise ConcurrentReprocessError(
            f"Material {material_id} has an in-progress reprocess job ({latest_job.id})"
        )

    await purge_chunks_for_version(db, version.id)

    version.processing_status = "pending"
    version.processing_error = None
    version.processed_at = None
    await flush_or_conflict(db)

    pipeline_run_id = uuid4()
    job = ProcessingJob(
        entity_type="material_version",
        entity_id=version.id,
        job_type="full_pipeline",
        status="pending",
    )
    db.add(job)
    await flush_or_conflict(db)
    await db.refresh(job)

    await enqueue_material_ingest(
        arq_pool,
        actor_id=actor.user_id,
        material_version_id=version.id,
        pipeline_run_id=pipeline_run_id,
    )

    return ReprocessResponse(
        material_id=material.id,
        version_id=version.id,
        processing_job_id=job.id,
        pipeline_run_id=pipeline_run_id,
    )


async def soft_delete_material(db: AsyncSession, material_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete the material + cascade to versions. Does NOT touch S3."""
    material = await require_material(db, material_id)
    await soft_delete_cascade(db, material, actor_id=actor.user_id)


async def restore_material(db: AsyncSession, material_id: UUID, actor: CurrentUser) -> None:
    """Lift the soft-delete tombstone on a material + its versions.

    The inverse of :func:`soft_delete_material`. Because the delete never
    touched S3 (plan §4946/§4954) and version-scoped chunks/embeddings were
    only tombstoned (not purged), a restored material comes back with its
    processed AI state intact — no reprocess needed. Raises
    :class:`NotFoundError` when there is no soft-deleted material to restore
    (already-active rows included — nothing to do).
    """
    restored = await restore_soft_deleted_material(db, material_id)
    if not restored:
        raise NotFoundError(f"No soft-deleted material {material_id} to restore")
    material = await require_material(db, material_id)
    material.updated_by = actor.user_id
    await flush_or_conflict(db)


async def list_material_versions(
    db: AsyncSession, material_id: UUID
) -> list[MaterialVersionAuthoring]:
    """Full version history (newest first) with processing state (FR-3.4/3.5)."""
    loaded = await get_material_with_versions(db, material_id)
    if loaded is None:
        raise NotFoundError(f"Material {material_id} not found")
    _, versions = loaded
    return [MaterialVersionAuthoring.model_validate(v) for v in versions]


async def rollback_material_version(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    actor: CurrentUser,
) -> MaterialVersionAuthoring:
    """Point the material's current version back at a prior ``ready`` one.

    Pure pointer swap (FR-3.4): ``document_chunks`` are version-scoped
    (``material_version_id``), so a previously-``ready`` version still
    has its chunks/embeddings — no reprocess is needed. Learner reads
    (presigned stream URLs, chunk previews) follow
    ``LearningMaterial.current_version_id`` immediately.

    Raises :class:`ConflictError` (router → 409) when the target is
    already current or its pipeline never reached ``ready``.
    """
    material = await require_material(db, material_id)
    version = await require_version(db, version_id)
    if version.material_id != material.id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    if version.is_current:
        raise ConflictError(f"Version {version_id} is already the current version")
    if version.processing_status != "ready":
        raise ConflictError(
            f"Version {version_id} is not ready "
            f"(processing_status={version.processing_status!r}); "
            "only fully-processed versions can be rolled back to"
        )

    # Same dual-source invariant as complete_upload (Reconciliation §C15).
    await reset_other_versions_current(db, material.id, version.id)
    version.is_current = True
    version.updated_by = actor.user_id
    material.current_version_id = version.id
    material.updated_by = actor.user_id
    await flush_or_conflict(db)
    return MaterialVersionAuthoring.model_validate(version)
