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
_HEALTH_SQL = _load("stats/health.sql")
_DASHBOARD_SQL = _load("stats/dashboard.sql")


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
    days: int,
    now: datetime,
) -> list[tuple[date, int]]:
    rows = (
        await db.execute(
            _ACTIVE_USERS_TREND_SQL,
            {"organization_id": organization_id, "days": days, "now": now},
        )
    ).mappings()
    return [(row["day"], int(row["count"] or 0)) for row in rows]


async def content_breakdown(db: AsyncSession, *, organization_id: UUID | None) -> dict[str, Any]:
    row = (await db.execute(_CONTENT_SQL, {"organization_id": organization_id})).mappings().one()
    return {
        "courses_by_status": list(row["courses_by_status"] or []),
        "materials_by_type": list(row["materials_by_type"] or []),
        "processing_jobs_by_status": list(row["processing_jobs_by_status"] or []),
        "courses_created_7d": int(row["courses_created_7d"] or 0),
        "materials_created_7d": int(row["materials_created_7d"] or 0),
        "processing_jobs_created_today": int(row["processing_jobs_created_today"] or 0),
    }


async def health_snapshot(db: AsyncSession, *, since: datetime) -> dict[str, int]:
    row = (await db.execute(_HEALTH_SQL, {"since": since})).mappings().one()
    return {
        "failed_jobs_count": int(row["failed_jobs_count"] or 0),
        "in_flight_jobs_count": int(row["in_flight_jobs_count"] or 0),
        "failed_ai_calls_count": int(row["failed_ai_calls_count"] or 0),
    }


async def operator_dashboard(
    db: AsyncSession, *, organization_id: UUID | None, now: datetime
) -> dict[str, Any]:
    """One-row operator dashboard rollup (see ``sql/stats/dashboard.sql``).

    ``job_failure_rate_pct`` is derived here rather than in SQL so the
    zero-jobs case is a single explicit branch.
    """
    row = (
        (
            await db.execute(
                _DASHBOARD_SQL,
                {"organization_id": organization_id, "now": now},
            )
        )
        .mappings()
        .one()
    )
    jobs_failed = int(row["jobs_failed_7d"] or 0)
    jobs_total = int(row["jobs_total_7d"] or 0)
    failure_rate = round(100.0 * jobs_failed / jobs_total, 2) if jobs_total else 0.0
    return {
        "job_failure_rate_pct": failure_rate,
        "jobs_failed_7d": jobs_failed,
        "jobs_total_7d": jobs_total,
        "jobs_failed_prev_7d": int(row["jobs_failed_prev_7d"] or 0),
        "jobs_total_prev_7d": int(row["jobs_total_prev_7d"] or 0),
        "queue_depth": int(row["queue_depth"] or 0),
        "failed_ai_calls_30d": int(row["failed_ai_calls_30d"] or 0),
        "spend_7d_usd": float(row["spend_7d_usd"] or 0),
        "spend_prev_7d_usd": float(row["spend_prev_7d_usd"] or 0),
        "projected_month_end_usd": float(row["projected_month_end_usd"] or 0),
        "top_cost_driver": row["top_cost_driver"],
        "top_cost_driver_usd": float(row["top_cost_driver_usd"] or 0),
        "slowest_model": row["slowest_model"],
        "slowest_model_p95_ms": int(row["slowest_model_p95_ms"] or 0),
        "active_users_today": int(row["active_users_today"] or 0),
        "active_users_7d": int(row["active_users_7d"] or 0),
        "total_users": int(row["total_users"] or 0),
        "quiz_sessions_completed_7d": int(row["quiz_sessions_completed_7d"] or 0),
        "interview_sessions_7d": int(row["interview_sessions_7d"] or 0),
        "interview_pass_rate_pct": float(row["interview_pass_rate_pct"] or 0),
        "interview_evaluated_7d": int(row["interview_evaluated_7d"] or 0),
        "interview_students_7d": int(row["interview_students_7d"] or 0),
        "materials_ingested_7d": int(row["materials_ingested_7d"] or 0),
        "materials_stuck_processing": int(row["materials_stuck_processing"] or 0),
        "published_quizzes_missing_texp": int(row["published_quizzes_missing_texp"] or 0),
        "interview_configs_no_reviewed_questions": int(
            row["interview_configs_no_reviewed_questions"] or 0
        ),
        "orgs_inactive_30d": int(row["orgs_inactive_30d"] or 0),
    }


__all__ = [
    "active_users",
    "content_breakdown",
    "health_snapshot",
    "operator_dashboard",
    "overview_counts",
]
