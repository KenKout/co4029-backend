"""Processing service -- queue depth, job listing, retry orchestration (T7.5).

Retry semantics: the service updates the failed job in-place (status='pending',
retry_count++, error cleared) and best-effort re-enqueues to ARQ if a pool is
provided. Per the plan: "production may need a separate retry queue" -- we do
the simple thing today (re-enqueue against the original ``job_type``) and let
the operator escalate via task management if a worker still cannot drain it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.admin.queries import processing as processing_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_RESET_FAILED_JOB_SQL = text(
    """
    UPDATE processing_jobs
       SET status = 'pending',
           retry_count = retry_count + 1,
           error_message = NULL,
           started_at = NULL,
           finished_at = NULL,
           progress_percent = 0,
           updated_at = NOW()
     WHERE id = :job_id
       AND status = 'failed'
    RETURNING id, entity_type, entity_id, job_type, status, retry_count
    """
)


async def queue_depth(db: AsyncSession) -> dict[str, int]:
    return await processing_queries.queue_depth(db)


async def status_counts_since(db: AsyncSession, *, since: datetime) -> dict[str, int]:
    """Per-status counts over the same ``since`` window as :func:`list_jobs`."""
    return await processing_queries.status_counts_since(db, since=since)


async def list_jobs(
    db: AsyncSession,
    *,
    status: str | None,
    since: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    return await processing_queries.list_jobs(db, status=status, since=since, limit=limit)


async def get_job(db: AsyncSession, *, job_id: UUID) -> dict[str, Any]:
    job = await processing_queries.get_job(db, job_id=job_id)
    if job is None:
        raise NotFoundError(f"processing_job {job_id} not found")
    return job


async def retry_failed_job(
    db: AsyncSession,
    *,
    job_id: UUID,
    arq_pool: object | None,
) -> dict[str, Any]:
    """Reset a failed job to ``pending`` and best-effort re-enqueue.

    Returns the refreshed job row. Raises :class:`NotFoundError` if the job
    does not exist OR is not in the ``failed`` state (idempotent guard --
    re-running on an already-running job is a no-op).
    """
    row = (await db.execute(_RESET_FAILED_JOB_SQL, {"job_id": job_id})).mappings().one_or_none()
    if row is None:
        raise NotFoundError(f"processing_job {job_id} not found or not in 'failed' state")
    await db.commit()

    if arq_pool is not None:
        enqueue = getattr(arq_pool, "enqueue_job", None)
        if enqueue is not None:
            await enqueue(
                row["job_type"],
                str(row["id"]),
                str(row["entity_id"]),
            )

    refreshed = await processing_queries.get_job(db, job_id=job_id)
    if refreshed is None:
        raise NotFoundError(f"processing_job {job_id} vanished after retry")
    return refreshed


__all__ = ["get_job", "list_jobs", "queue_depth", "retry_failed_job", "status_counts_since"]
