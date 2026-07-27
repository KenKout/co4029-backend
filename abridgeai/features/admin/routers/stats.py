"""Stats router -- ``/admin/stats`` (T7.5).

All endpoints require ``system.stats.read`` OR ``system.administer``.
Org-scoping is resolved via :func:`resolve_admin_scope`.

* ``/overview`` and ``/content`` are bounded scans (top-level COUNT(*) over
  finite tables) -- no ``since`` parameter needed.
* ``/active-users`` uses fixed 24h / 7d / 30d windows.
* ``/health`` requires ``since`` to keep the failed-jobs / failed-AI-calls
  scans bounded.
* ``/dashboard`` is the operator rollup: fixed 1h / 24h / 7d / 14d / 30d and
  month-to-date windows, all evaluated server-side against ``now``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.admin.routers._scope import resolve_admin_scope
from abridgeai.features.admin.services import stats as stats_service

router = APIRouter(prefix="/admin/stats", tags=["admin", "stats"])

_REQUIRE_STATS = require_any_permission("system.stats.read", "system.administer")


class OverviewOut(BaseModel):
    total_users: int
    total_courses: int
    total_enrollments: int
    total_materials: int
    total_quiz_attempts: int


class ActiveUsersOut(BaseModel):
    dau: int
    wau: int
    mau: int


class HealthOut(BaseModel):
    failed_jobs_count: int
    in_flight_jobs_count: int
    failed_ai_calls_count: int


class ContentOut(BaseModel):
    courses_by_status: list[dict[str, Any]]
    materials_by_type: list[dict[str, Any]]
    processing_jobs_by_status: list[dict[str, Any]]


class DashboardOut(BaseModel):
    """Operator dashboard rollup.

    ``processing_jobs`` and ``ai_model_calls`` carry no organization edge in
    the schema, so the job / cost / latency fields are always global even for
    an org-scoped caller (documented in ``sql/stats/dashboard.sql``).
    """

    # needs action
    job_failure_rate_pct: float
    jobs_failed_7d: int
    jobs_total_7d: int
    jobs_failed_prev_7d: int
    jobs_total_prev_7d: int
    queue_depth: int
    failed_ai_calls_30d: int
    # cost snapshot
    spend_7d_usd: float
    spend_prev_7d_usd: float
    projected_month_end_usd: float
    top_cost_driver: str | None
    top_cost_driver_usd: float
    slowest_model: str | None
    slowest_model_p95_ms: int
    # activity
    active_users_today: int
    active_users_7d: int
    total_users: int
    quiz_sessions_completed_7d: int
    interview_sessions_7d: int
    interview_pass_rate_pct: float
    # Sample size behind the pass rate — a low rate over a couple of students is
    # a testing artifact, not a platform signal.
    interview_evaluated_7d: int
    interview_students_7d: int
    materials_ingested_7d: int
    # needs attention (checklist)
    materials_stuck_processing: int
    published_quizzes_missing_texp: int
    interview_configs_no_reviewed_questions: int
    orgs_inactive_30d: int


@router.get("/overview", response_model=OverviewOut)
async def get_overview(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OverviewOut:
    org_id = await resolve_admin_scope(db, user)
    counts = await stats_service.overview(db, organization_id=org_id)
    return OverviewOut(**counts)


@router.get("/active-users", response_model=ActiveUsersOut)
async def get_active_users(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ActiveUsersOut:
    org_id = await resolve_admin_scope(db, user)
    counts = await stats_service.active_users(db, organization_id=org_id)
    return ActiveUsersOut(**counts)


@router.get("/content", response_model=ContentOut)
async def get_content_breakdown(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ContentOut:
    org_id = await resolve_admin_scope(db, user)
    breakdown = await stats_service.content_breakdown(db, organization_id=org_id)
    return ContentOut(**breakdown)


@router.get("/health", response_model=HealthOut)
async def get_health(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[
        datetime,
        Query(description="Lower bound on event time (required to bound scan)."),
    ],
) -> HealthOut:
    del user
    snapshot = await stats_service.health(db, since=since)
    return HealthOut(**snapshot)


@router.get("/dashboard", response_model=DashboardOut)
async def get_dashboard(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DashboardOut:
    org_id = await resolve_admin_scope(db, user)
    metrics = await stats_service.operator_dashboard(db, organization_id=org_id)
    return DashboardOut(**metrics)


__all__ = ["router"]
