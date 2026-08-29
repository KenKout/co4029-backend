"""Processing-job queries for the admin dashboard (T7.5)."""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

_SQL_DIR = resources.files("abridgeai.features.admin.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_QUEUE_DEPTH_SQL = _load("processing/queue_depth.sql")
_SUMMARY_SQL = _load("processing/summary.sql")
_LIST_JOBS_SQL = _load("processing/list_jobs.sql")
_GET_JOB_SQL = _load("processing/get_job.sql")
_JOB_CONTEXT_SQL = _load("processing/job_context.sql")
_JOB_AI_STAGES_SQL = _load("processing/job_ai_stages.sql")
_JOB_AI_CALLS_SQL = _load("processing/job_ai_calls.sql")


async def queue_depth(db: AsyncSession) -> dict[str, int]:
    row = (await db.execute(_QUEUE_DEPTH_SQL)).mappings().one()
    return {
        "pending": int(row["pending_count"] or 0),
        "running": int(row["running_count"] or 0),
        "failed": int(row["failed_count"] or 0),
        "completed": int(row["completed_count"] or 0),
        "cancelled": int(row["cancelled_count"] or 0),
        "total": int(row["total_count"] or 0),
    }


async def status_counts_since(
    db: AsyncSession, *, since: datetime, until: datetime | None = None
) -> dict[str, int]:
    """Per-status counts over the same ``since`` window as :func:`list_jobs`.

    The admin processing page's tab badges read from here — deriving them
    client-side from the status-filtered jobs list made every other tab's
    count collapse to zero the moment one status was selected.
    """
    row = (
        await db.execute(_SUMMARY_SQL, {"since": since, "until": until})
    ).mappings().one()
    return {
        "pending": int(row["pending_count"] or 0),
        "running": int(row["running_count"] or 0),
        "failed": int(row["failed_count"] or 0),
        "completed": int(row["completed_count"] or 0),
        "cancelled": int(row["cancelled_count"] or 0),
        "total": int(row["total_count"] or 0),
    }


async def list_jobs(
    db: AsyncSession,
    *,
    status: str | None,
    since: datetime,
    until: datetime | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _LIST_JOBS_SQL,
            {"status": status, "since": since, "until": until, "limit": limit},
        )
    ).mappings()
    return [dict(r) for r in rows]


async def get_job(db: AsyncSession, *, job_id: UUID) -> dict[str, Any] | None:
    row = (await db.execute(_GET_JOB_SQL, {"job_id": job_id})).mappings().one_or_none()
    return dict(row) if row is not None else None


async def job_context(
    db: AsyncSession, *, job_id: UUID, now: datetime
) -> dict[str, Any] | None:
    """Owner + timings for one job (``sql/processing/job_context.sql``).

    ``None`` when the id is not a ``processing_jobs`` row -- generation runs
    share the id space on the detail endpoint but have their own ownership,
    so the caller decides what to do rather than getting a row of nulls.
    """
    row = (
        await db.execute(_JOB_CONTEXT_SQL, {"job_id": str(job_id), "now": now})
    ).mappings().one_or_none()
    return dict(row) if row else None


async def job_ai_stages(db: AsyncSession, *, job_id: UUID) -> list[dict[str, Any]]:
    """Per-stage AI rollup for one job."""
    rows = (
        await db.execute(_JOB_AI_STAGES_SQL, {"job_id": str(job_id)})
    ).mappings()
    return [dict(r) for r in rows]


async def job_ai_calls(
    db: AsyncSession, *, job_id: UUID, limit: int
) -> list[dict[str, Any]]:
    """Individual AI calls for one job, newest first."""
    rows = (
        await db.execute(
            _JOB_AI_CALLS_SQL, {"job_id": str(job_id), "limit": limit}
        )
    ).mappings()
    return [dict(r) for r in rows]


__all__ = [
    "get_job",
    "job_ai_calls",
    "job_ai_stages",
    "job_context",
    "list_jobs",
    "queue_depth",
    "status_counts_since",
]
