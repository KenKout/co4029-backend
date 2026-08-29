"""Stats aggregation queries (T7.5)."""

from __future__ import annotations

from datetime import date, datetime
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
_ACTIVE_USERS_TREND_SQL = _load("stats/active_users_trend.sql")
_CONTENT_SQL = _load("stats/content.sql")
_DASHBOARD_SQL = _load("stats/dashboard.sql")
_API_RELIABILITY_SQL = _load("stats/api_reliability.sql")
_API_LATENCY_TREND_SQL = _load("stats/api_latency_trend.sql")


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


async def active_users_trend(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[date, int]]:
    rows = (
        await db.execute(
            _ACTIVE_USERS_TREND_SQL,
            {
                "organization_id": organization_id,
                "window_start": window_start,
                "window_end": window_end,
            },
        )
    ).mappings()
    return [(row["day"], int(row["count"] or 0)) for row in rows]


async def api_latency_trend(
    db: AsyncSession,
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[date, int, float | None, float | None]]:
    """Daily latency percentiles + volume; raw floats, callers round."""
    rows = (
        await db.execute(
            _API_LATENCY_TREND_SQL,
            {"window_start": window_start, "window_end": window_end},
        )
    ).mappings()
    return [
        (
            row["day"],
            int(row["requests_total"] or 0),
            row["p50_latency_ms"],
            row["p95_latency_ms"],
        )
        for row in rows
    ]


async def content_breakdown(db: AsyncSession, *, organization_id: UUID | None) -> dict[str, Any]:
    row = (await db.execute(_CONTENT_SQL, {"organization_id": organization_id})).mappings().one()
    return {
        "courses_by_status": list(row["courses_by_status"] or []),
        "materials_by_type": list(row["materials_by_type"] or []),
        "courses_created_7d": int(row["courses_created_7d"] or 0),
        "materials_created_7d": int(row["materials_created_7d"] or 0),
    }


async def operator_dashboard(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    as_of: datetime,
    window_start: datetime,
    window_end: datetime,
    previous_start: datetime,
) -> dict[str, Any]:
    """Cost / usage / tenant half of the dashboard (``sql/stats/dashboard.sql``).

    Job metrics are NOT here -- they come from
    :mod:`abridgeai.features.admin.services.job_metrics`, the definition shared
    with the processing surface (PRD ADM-004). The window bounds are explicit
    so the service can hand the SAME span to every component of the rollup.
    """
    row = (
        (
            await db.execute(
                _DASHBOARD_SQL,
                {
                    "organization_id": organization_id,
                    "as_of": as_of,
                    "window_start": window_start,
                    "window_end": window_end,
                    "previous_start": previous_start,
                },
            )
        )
        .mappings()
        .one()
    )
    return {
        "as_of": row["as_of"],
        "spend_window_usd": float(row["spend_window_usd"] or 0),
        "spend_prev_window_usd": float(row["spend_prev_window_usd"] or 0),
        "tokens_window": int(row["tokens_window"] or 0),
        "failed_ai_calls_window": int(row["failed_ai_calls_window"] or 0),
        "ai_calls_window": int(row["ai_calls_window"] or 0),
        "projected_month_end_usd": float(row["projected_month_end_usd"] or 0),
        "top_cost_driver": row["top_cost_driver"],
        "top_cost_driver_usd": float(row["top_cost_driver_usd"] or 0),
        "slowest_model": row["slowest_model"],
        "slowest_model_p95_ms": int(row["slowest_model_p95_ms"] or 0),
        "active_users_today": int(row["active_users_today"] or 0),
        "active_users_window": int(row["active_users_window"] or 0),
        "total_users": int(row["total_users"] or 0),
        "materials_ingested_window": int(row["materials_ingested_window"] or 0),
        "orgs_total": int(row["orgs_total"] or 0),
    }


async def api_reliability(
    db: AsyncSession,
    *,
    as_of: datetime,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    """Request error rate + latency percentiles (``sql/stats/api_reliability.sql``).

    Percentiles and the totals come back raw; the service turns them into
    rates so the empty-window case stays a single explicit branch.
    """
    row = (
        (
            await db.execute(
                _API_RELIABILITY_SQL,
                {
                    "as_of": as_of,
                    "window_start": window_start,
                    "window_end": window_end,
                },
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


__all__ = [
    "active_users",
    "api_reliability",
    "content_breakdown",
    "operator_dashboard",
    "overview_counts",
]
