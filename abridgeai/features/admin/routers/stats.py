"""Stats router -- ``/admin/stats`` (T7.5).

All endpoints require ``system.stats.read`` OR ``system.administer``.
Org-scoping is resolved via :func:`resolve_admin_scope`.

* ``/overview`` and ``/content`` are bounded scans (top-level COUNT(*) over
  finite tables) -- no ``since`` parameter needed.
* ``/active-users`` uses fixed 24h / 7d / 30d windows.
* ``/health`` is GONE. It counted failed and in-flight jobs over
  ``processing_jobs`` alone, window-filtered on ``updated_at`` -- a fourth
  incompatible definition of "how many jobs failed", and one that under-
  reported the queue (a job pending since last week is still in flight now but
  fell outside a 24h window). The dashboard's reliability fields answer the
  same question off the shared job contract in ``services/job_metrics.py``.
* ``/dashboard`` is the operator rollup. Its window is caller-selectable
  (``window_days``, default 7) and every windowed metric in the response uses
  it, with the preceding window of the same length alongside for direction.
  The response carries ``as_of``, ``window_days`` and a ``*_scope`` per metric
  family so no tile is ambiguous about what it measures (PRD ADM-004).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
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


class ActiveUsersTrendPoint(BaseModel):
    date: date
    count: int


class ActiveUsersTrendOut(BaseModel):
    points: list[ActiveUsersTrendPoint]


class LatencyTrendPoint(BaseModel):
    day: date
    requests_total: int
    # NULL on a zero-traffic day — no fabricated 0ms line.
    p50_latency_ms: int | None = None
    p95_latency_ms: int | None = None


class LatencyTrendOut(BaseModel):
    points: list[LatencyTrendPoint]


class ContentOut(BaseModel):
    """Content inventory. Processing-job status is deliberately absent: jobs
    live in the Operations surface only (PRD ADM-004 / section 2 IA rule), so
    there is exactly one place that answers "how are jobs doing"."""

    courses_by_status: list[dict[str, Any]]
    materials_by_type: list[dict[str, Any]]
    # Analytics deltas for the content page's summary cards, both org-scoped.
    courses_created_7d: int = 0
    materials_created_7d: int = 0


class DashboardOut(BaseModel):
    """Operator dashboard rollup, ordered the way the dashboard reads it.

    Two rules run through the whole payload:

    * **Every rate is nullable.** ``None`` means the denominator was empty and
      the client must render "No data". A fabricated 0% makes a quiet platform
      indistinguishable from a healthy one (PRD section 5).
    * **Every family declares its scope.** ``processing_jobs``,
      ``ai_model_calls`` and ``http_audit_log`` carry no organization edge, so
      an org-scoped caller gets global numbers there. The ``*_scope`` fields
      say so outright instead of letting the figure imply a tenant filter it
      never had (PRD ADM-004).
    """

    # -- envelope --------------------------------------------------------
    as_of: datetime
    window_days: int
    # Exact date range (inclusive) the window covered, when the caller asked
    # for one instead of the relative ``window_days`` shape. The UI echoes
    # these verbatim so the label it prints == the rows counted.
    window_from: date | None = None
    window_to: date | None = None
    organization_id: UUID | None = None
    usage_scope: str
    tenant_scope: str
    job_scope: str
    cost_scope: str
    api_scope: str

    # -- reliability & throughput ---------------------------------------
    job_failure_rate_pct: float | None
    job_failure_rate_prev_pct: float | None
    jobs_terminal_window: int
    jobs_failed_window: int
    jobs_terminal_prev_window: int
    jobs_failed_prev_window: int
    queue_depth: int
    queue_pending: int
    queue_running: int
    queue_oldest_age_seconds: int | None
    requests_window: int
    requests_5xx_window: int
    requests_4xx_window: int
    api_error_rate_pct: float | None
    api_client_error_rate_pct: float | None
    api_p50_latency_ms: int | None
    api_p95_latency_ms: int | None

    # -- cost & capacity -------------------------------------------------
    spend_window_usd: float
    spend_prev_window_usd: float
    projected_month_end_usd: float
    tokens_window: int
    ai_calls_window: int
    failed_ai_calls_window: int
    ai_failure_rate_pct: float | None
    top_cost_driver: str | None
    top_cost_driver_usd: float
    slowest_model: str | None
    slowest_model_p95_ms: int

    # -- usage ------------------------------------------------------------
    active_users_today: int
    active_users_window: int
    total_users: int
    materials_ingested_window: int

    # -- tenant anomalies --------------------------------------------------
    orgs_total: int
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


@router.get("/active-users/trend", response_model=ActiveUsersTrendOut)
async def get_active_users_trend(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> ActiveUsersTrendOut:
    """Daily active users over the lookback window (distinct logins/day).

    Drives the trend chart on the Active Users tab, mirroring the AI-cost
    trend. ``days`` defaults to 30; every day in the window is returned
    (zero-activity days included) so the chart is continuous.
    """
    org_id = await resolve_admin_scope(db, user)
    raw = await stats_service.active_users_trend(db, organization_id=org_id, days=days)
    points = [ActiveUsersTrendPoint(date=d, count=n) for d, n in raw]
    return ActiveUsersTrendOut(points=points)


@router.get("/latency/trend", response_model=LatencyTrendOut)
async def get_api_latency_trend(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> LatencyTrendOut:
    """Daily p50/p95 API latency over the lookback window.

    Drives the latency chart on the stats overview, mirroring the
    active-users trend. GLOBAL scope — ``http_audit_log`` does not carry
    organization (see ``api_reliability.sql``). Every day in the window is
    returned (zero-traffic days included) so the chart is continuous.
    """
    points = await stats_service.api_latency_trend(db, days=days)
    return LatencyTrendOut(
        points=[
            LatencyTrendPoint(
                day=d,
                requests_total=r,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
            )
            for d, r, p50, p95 in points
        ]
    )


@router.get("/content", response_model=ContentOut)
async def get_content_breakdown(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ContentOut:
    org_id = await resolve_admin_scope(db, user)
    breakdown = await stats_service.content_breakdown(db, organization_id=org_id)
    return ContentOut(**breakdown)


@router.get("/dashboard", response_model=DashboardOut)
async def get_dashboard(
    user: Annotated[CurrentUser, Depends(_REQUIRE_STATS)],
    db: Annotated[AsyncSession, Depends(get_db)],
    window_days: Annotated[
        int,
        Query(
            ge=1,
            le=90,
            description="Length of every windowed metric, in days. All tiles "
            "move together so they stay comparable.",
        ),
    ] = stats_service.DEFAULT_WINDOW_DAYS,
    window_from: Annotated[
        date | None,
        Query(
            alias="from",
            description="Exact window start date (inclusive). Must pair with "
            "'to'; overrides window_days with a fixed calendar range so a "
            "custom picker range (e.g. Aug 1 - Aug 29) counts exactly the "
            "rows in it.",
        ),
    ] = None,
    window_to: Annotated[
        date | None,
        Query(
            alias="to",
            description="Exact window end date (inclusive). Must pair with "
            "'from', and must not be after today's date on the server.",
        ),
    ] = None,
    organization_id: Annotated[
        UUID | None,
        Query(
            description="Narrow org-traceable metrics to one tenant. Honoured "
            "only for callers holding system.administer -- everyone else is "
            "already pinned to their own organization."
        ),
    ] = None,
) -> DashboardOut:
    """Operator rollup for the admin dashboard.

    Scope resolution (PRD ADM-005): a Manager / HOD is always pinned to their
    own organization and the ``organization_id`` parameter is ignored for them
    -- accepting it would be a cross-tenant read. An IT Admin defaults to the
    global view and may narrow to one tenant with it.
    """
    if (window_from is None) != (window_to is None):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_error",
                "message": "'from' and 'to' must be provided together",
            },
        )
    if window_from is not None and window_to is not None:
        if window_from > window_to:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "'from' must not be after 'to'",
                },
            )
        if window_to > datetime.now(tz=UTC).date():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "'to' must not be after today",
                },
            )
        if (window_to - window_from).days + 1 > 366:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "range must not exceed 366 days",
                },
            )
    scope = await resolve_admin_scope(db, user)
    if scope is None and organization_id is not None:
        # Only reachable with system.administer (resolve_admin_scope returns
        # None exclusively for that permission).
        scope = organization_id
    metrics = await stats_service.operator_dashboard(
        db,
        organization_id=scope,
        window_days=window_days,
        window_from=window_from,
        window_to=window_to,
    )
    return DashboardOut(**metrics)


__all__ = ["router"]
