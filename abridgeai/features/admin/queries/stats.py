"""Stats aggregation queries (T7.5)."""

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


_OVERVIEW_SQL = _load("stats/overview.sql")
_ACTIVE_USERS_SQL = _load("stats/active_users.sql")
_CONTENT_SQL = _load("stats/content.sql")
_HEALTH_SQL = _load("stats/health.sql")


async def overview_counts(db: AsyncSession, *, organization_id: UUID | None) -> dict[str, int]:
    row = (await db.execute(_OVERVIEW_SQL, {"organization_id": organization_id})).mappings().one()
    return {
        "total_users": int(row["total_users"] or 0),
        "total_courses": int(row["total_courses"] or 0),
        "total_enrollments": int(row["total_enrollments"] or 0),
        "total_materials": int(row["total_materials"] or 0),
        "total_quiz_attempts": int(row["total_quiz_attempts"] or 0),
    }


async def active_users(
    db: AsyncSession, *, organization_id: UUID | None, now: datetime
) -> dict[str, int]:
    row = (
        (
            await db.execute(
                _ACTIVE_USERS_SQL,
                {"organization_id": organization_id, "now": now},
            )
        )
        .mappings()
        .one()
    )
    return {
        "dau": int(row["dau"] or 0),
        "wau": int(row["wau"] or 0),
        "mau": int(row["mau"] or 0),
    }


async def content_breakdown(
    db: AsyncSession, *, organization_id: UUID | None
) -> dict[str, list[dict[str, Any]]]:
    row = (await db.execute(_CONTENT_SQL, {"organization_id": organization_id})).mappings().one()
    return {
        "courses_by_status": list(row["courses_by_status"] or []),
        "materials_by_type": list(row["materials_by_type"] or []),
        "processing_jobs_by_status": list(row["processing_jobs_by_status"] or []),
    }


async def health_snapshot(db: AsyncSession, *, since: datetime) -> dict[str, int]:
    row = (await db.execute(_HEALTH_SQL, {"since": since})).mappings().one()
    return {
        "failed_jobs_count": int(row["failed_jobs_count"] or 0),
        "in_flight_jobs_count": int(row["in_flight_jobs_count"] or 0),
        "failed_ai_calls_count": int(row["failed_ai_calls_count"] or 0),
    }


__all__ = [
    "active_users",
    "content_breakdown",
    "health_snapshot",
    "overview_counts",
]
