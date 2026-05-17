"""ARQ task wrapping the materials ingestion pipeline (T4.7).

Bandwidth-saving architecture:

* The worker pulls source bytes directly from S3 (internal endpoint via
  ``download_to_temp``); the backend HTTP API is never used as a proxy.
* The file lands in a ``tempfile.TemporaryDirectory`` whose context exit
  auto-cleans on success, failure, or cancellation.
* The pipeline (T4.4) is invoked with the local path; it streams from
  disk for memory discipline.

Convention (Phase 0.8 / plan §5107-5108):

* Signature is ``async def task(ctx, actor_id: UUID, ...)`` — ``actor_id``
  is the FIRST argument after ``ctx``. ``set_worker_actor`` installs it
  into the audit context so SQLAlchemy ``before_flush`` populates
  ``created_by`` / ``updated_by`` automatically.
* Structured-log context (``request_id``-equivalent) is bound via
  ``bind_request_context`` and torn down in ``finally`` so neighbouring
  tasks in the worker pool never see leaked state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.audit import current_actor_var
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.materials.ingestion import run_material_ingest
from abridgeai.features.materials.models import LearningMaterialVersion
from abridgeai.infrastructure.s3 import download_to_temp
from abridgeai.workers.actor import set_worker_actor

_logger = get_logger(__name__)


@dataclass
class _StorageView:
    """Duck-typed view of a ``storage_objects`` row.

    Mirrors the pattern from ``features.materials.ingestion.pipeline``:
    avoids importing the identity feature's ORM ``StorageObject`` (which
    would break the import-linter "Features are independent" contract).
    The S3 helper ``download_to_temp`` only needs ``bucket`` and
    ``object_key``.
    """

    bucket: str
    object_key: str


async def _load_storage_view(
    db: AsyncSession,
    storage_object_id: UUID,
) -> _StorageView | None:
    row = (
        await db.execute(
            text("SELECT bucket, object_key FROM storage_objects WHERE id = :id"),
            {"id": storage_object_id},
        )
    ).first()
    if row is None:
        return None
    return _StorageView(bucket=row.bucket, object_key=row.object_key)


async def ingest_material_version_task(
    ctx: dict[str, Any],
    actor_id: UUID,
    material_version_id: UUID,
    pipeline_run_id: UUID,
) -> None:
    """ARQ task: run ingestion for one ``LearningMaterialVersion``.

    The worker downloads the source file from S3 into a per-task temp
    directory, hands the local path to the T4.4 pipeline, then commits.
    On any unhandled exception the pipeline's ``_capture_failure`` writes
    have already flushed (status='failed', error_message populated); the
    worker re-raises so ARQ records the failure and applies the
    configured retry policy.

    Parameters
    ----------
    ctx
        ARQ task context (unused here; reserved for ARQ internals).
    actor_id
        UUID of the user (or system actor) that initiated the ingest;
        propagated to audit columns via ``set_worker_actor``.
    material_version_id
        FK into ``learning_material_versions``.
    pipeline_run_id
        Audit-grouping UUID stamped onto every ``ai_model_calls`` row
        produced by this run (see T2.4 / T4.4).
    """
    _ = ctx
    set_worker_actor(actor_id)
    bind_request_context(
        material_version_id=str(material_version_id),
        pipeline_run_id=str(pipeline_run_id),
        actor_id=str(actor_id),
    )
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            try:
                version = await db.get(LearningMaterialVersion, material_version_id)
                if version is None:
                    _logger.warning(
                        "materials_ingest_task_missing_version",
                        material_version_id=str(material_version_id),
                    )
                    return
                storage_view = await _load_storage_view(db, version.storage_object_id)
                if storage_view is None:
                    _logger.warning(
                        "materials_ingest_task_missing_storage_object",
                        material_version_id=str(material_version_id),
                        storage_object_id=str(version.storage_object_id),
                    )
                    return

                with TemporaryDirectory(prefix="abridgeai-worker-") as temp_dir:
                    local_path = await download_to_temp(storage_view, dest_dir=Path(temp_dir))
                    await run_material_ingest(
                        db,
                        material_version_id,
                        pipeline_run_id,
                        source_path=local_path,
                    )
                await db.commit()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _logger.exception(
                    "materials_ingest_task_failed",
                    material_version_id=str(material_version_id),
                    pipeline_run_id=str(pipeline_run_id),
                )
                # Persist the pipeline's ``_capture_failure`` audit rows
                # (processing_status='failed', error_message populated)
                # before propagating to ARQ so the failure survives retry.
                await db.commit()
                raise
    finally:
        current_actor_var.set(None)
        clear_request_context()


__all__ = ["ingest_material_version_task"]
