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
_LIST_JOBS_SQL = _load("processing/list_jobs.sql")
_GET_JOB_SQL = _load("processing/get_job.sql")


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


async def list_jobs(
    db: AsyncSession,
    *,
    status: str | None,
    since: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _LIST_JOBS_SQL,
            {"status": status, "since": since, "limit": limit},
        )
    ).mappings()
    return [dict(r) for r in rows]


async def get_job(db: AsyncSession, *, job_id: UUID) -> dict[str, Any] | None:
    row = (await db.execute(_GET_JOB_SQL, {"job_id": job_id})).mappings().one_or_none()
    return dict(row) if row is not None else None


__all__ = ["get_job", "list_jobs", "queue_depth"]
