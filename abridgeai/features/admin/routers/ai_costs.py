"""AI cost dashboard router (T0.27) — ``/admin/ai/costs/*``.

Read-only observability surface over ``ai_model_calls``. Per the user
decision, this dashboard does **not** provide any hard rate-limit endpoint:
all routes are ``GET`` and never refuse based on cost.

Authorization: requires ``ai.processing.read`` or ``system.administer``
(matches T7.5 admin/processing precedent).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.admin.services import ai_costs as ai_costs_service

router = APIRouter(prefix="/admin/ai/costs", tags=["admin", "ai", "costs"])

_REQUIRE_READ = require_any_permission("ai.processing.read", "system.administer")


class CostTotals(BaseModel):
    tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    usd: float
    call_count: int


class FailedSpend(BaseModel):
    call_count: int = 0
    usd: float = 0.0


class RoleBreakdown(BaseModel):
    role: str
    tokens: int
    usd: float


class StageBreakdown(BaseModel):
    stage_name: str
    tokens: int
    usd: float


class TimeBucket(BaseModel):
    bucket_start_ts: datetime
    tokens: int
    usd: float


class SummaryOut(BaseModel):
    totals: CostTotals
    failed: FailedSpend = FailedSpend()
    by_role: list[RoleBreakdown]
    by_stage: list[StageBreakdown]
    buckets: list[TimeBucket]


class UserSpendOut(BaseModel):
    user_id: UUID
    display_name: str
    call_count: int
    total_tokens: int
    total_usd: float


class PipelineStage(BaseModel):
    stage_name: str
    tokens: int
    usd: float


class PipelineSpendOut(BaseModel):
    pipeline_run_id: UUID
    generation_type: str | None = None
    course_id: UUID | None = None
    started_at: datetime | None = None
    call_count: int
    total_tokens: int
    total_usd: float
    stages_breakdown: list[PipelineStage]


class RecentCallOut(BaseModel):
    id: UUID
    role: str | None = None
    tier: str | None = None
    stage_name: str | None = None
    model: str | None = None
    tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    usd: float
    latency_ms: int | None = None
    status: str | None = None
    created_at: datetime
    pipeline_run_id: UUID | None = None


class CategorySpendOut(BaseModel):
    dimension_value: str
    call_count: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_usd: float


class ModelEfficiencyOut(BaseModel):
    model_name: str
    call_count: int
    total_tokens: int
    total_usd: float
    latency_p50_ms: int
    latency_p95_ms: int
    usd_per_1m_tokens: float


def _parse_since(raw: str | None, default_days: int) -> datetime:
    return _parse_window(raw, None, default_days=default_days)[0]


def _parse_window(
    raw_since: str | None, raw_until: str | None, *, default_days: int
) -> tuple[datetime, datetime | None]:
    """Resolve the ``since``/``until`` pair to UTC datetimes.

    ``until`` is the window END as an EXCLUSIVE bound — the client sends local
    midnight after its last labelled day — or ``None`` for the legacy
    open-ended window ``[since, NOW())``. Omitting ``until`` keeps the old
    behaviour bit-for-bit: existing callers (and the integration tests that
    seed rows "since 5 days ago") keep their full open-ended span.
    """
    if raw_since is None:
        return ai_costs_service.default_since(default_days), None
    try:
        if "T" in raw_since or " " in raw_since:
            since = datetime.fromisoformat(raw_since.replace("Z", "+00:00"))
        else:
            since = datetime.strptime(raw_since, "%Y-%m-%d").replace(tzinfo=UTC)
        if raw_until is None:
            until_dt = None
        elif "T" in raw_until or " " in raw_until:
            until_dt = datetime.fromisoformat(raw_until.replace("Z", "+00:00"))
        else:
            until_dt = datetime.strptime(raw_until, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_since",
                "message": "since/until must be YYYY-MM-DD or ISO-8601 datetime",
            },
        ) from exc
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    if until_dt is not None and until_dt.tzinfo is None:
        until_dt = until_dt.replace(tzinfo=UTC)
    if until_dt is not None and until_dt <= since:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_window",
                "message": "until must not be on or before since",
            },
        )
    return since, until_dt


@router.get("/summary", response_model=SummaryOut)
async def get_summary(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    period: Annotated[
        str,
        Query(
            description="Bucket granularity: 'day', 'week', or 'month'.",
            pattern="^(day|week|month)$",
        ),
    ] = "day",
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    until: Annotated[
        str | None,
        Query(
            description=(
                "Exclusive window end (ISO date or datetime); omit for an "
                "open-ended window ending at NOW()."
            )
        ),
    ] = None,
    model: Annotated[
        str | None, Query(description="Filter to one model_name.")
    ] = None,
    role: Annotated[str | None, Query(description="Filter to one role.")] = None,
    operation: Annotated[
        str | None,
        Query(description="Filter to one operation (chat_completion|embedding)."),
    ] = None,
    call_status: Annotated[
        str | None,
        Query(alias="status", description="Filter to one call status."),
    ] = None,
) -> SummaryOut:
    since_dt, until_dt = _parse_window(since, until, default_days=30)
    payload: dict[str, Any] = await ai_costs_service.summary(
        db,
        since=since_dt,
        until=until_dt,
        period=period,
        model=model,
        role=role,
        operation=operation,
        status=call_status,
    )
    return SummaryOut.model_validate(payload)


@router.get("/by-user", response_model=list[UserSpendOut])
async def get_by_user(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    until: Annotated[
        str | None,
        Query(
            description=(
                "Exclusive window end (ISO date or datetime); omit for an "
                "open-ended window ending at NOW()."
            )
        ),
    ] = None,
    top_n: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[UserSpendOut]:
    since_dt, until_dt = _parse_window(since, until, default_days=30)
    # The soft-spend warning is a TODAY-window behaviour; a closed historical
    # window is not "today" regardless of what since resolves to.
    warn = until_dt is None and ai_costs_service.is_today_window(since_dt)
    rows = await ai_costs_service.by_user(
        db,
        since=since_dt,
        top_n=top_n,
        is_today_window=warn,
    )
    return [UserSpendOut.model_validate(r) for r in rows]


class OrganizationSpendOut(BaseModel):
    """Spend for one tenant.

    ``organization_id`` is NULL for the unattributed bucket -- calls with no
    derivable tenant, which are real spend and are kept so the rows still sum
    to the platform total.
    """

    organization_id: UUID | None = None
    organization_name: str = ""
    call_count: int
    failed_count: int
    tokens: int
    spend_usd: float


class OrganizationSpendPage(BaseModel):
    """Per-tenant spend plus how much of the bill it explains.

    ``coverage_pct`` travels with the rows on purpose: ``ai_model_calls`` has
    no tenant column, so this view derives ownership through optional parents
    and cannot reach every call. A breakdown that explains part of the bill
    without saying which part invites chargeback decisions the data does not
    support. ``None`` means the window had no spend at all -- not 0%.
    """

    items: list[OrganizationSpendOut]
    total_spend_usd: float
    attributed_spend_usd: float
    coverage_pct: float | None = None


@router.get("/by-organization", response_model=OrganizationSpendPage)
async def get_by_organization(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    until: Annotated[
        str | None,
        Query(
            description=(
                "Exclusive window end (ISO date or datetime); defaults to "
                "NOW() -- kept open by default so freshly-written calls stay "
                "attributable until the caller sends a bounded window."
            )
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> OrganizationSpendPage:
    """AI spend attributed to organizations (PRD ADM-040)."""
    since_dt, until_dt = _parse_window(since, until, default_days=30)
    result = await ai_costs_service.by_organization(
        db,
        since=since_dt,
        until=until_dt or datetime.now(tz=UTC),
        limit=limit,
    )
    return OrganizationSpendPage(
        items=[OrganizationSpendOut.model_validate(r) for r in result["items"]],
        total_spend_usd=result["total_spend_usd"],
        attributed_spend_usd=result["attributed_spend_usd"],
        coverage_pct=result["coverage_pct"],
    )


@router.get("/by-pipeline", response_model=list[PipelineSpendOut])
async def get_by_pipeline(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    until: Annotated[
        str | None,
        Query(
            description=(
                "Exclusive window end (ISO date or datetime); omit for an "
                "open-ended window ending at NOW()."
            )
        ),
    ] = None,
    top_n: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[PipelineSpendOut]:
    since_dt, until_dt = _parse_window(since, until, default_days=30)
    rows = await ai_costs_service.by_pipeline(
        db, since=since_dt, until=until_dt, top_n=top_n
    )
    return [PipelineSpendOut.model_validate(r) for r in rows]


@router.get("/by-category", response_model=list[CategorySpendOut])
async def get_by_category(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    dimension: Annotated[
        str,
        Query(
            description=(
                "Grouping dimension: 'operation', 'role', 'tier', "
                "'stage_name', 'model_name', or 'status'."
            ),
            pattern="^(operation|role|tier|stage_name|model_name|status)$",
        ),
    ] = "operation",
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    until: Annotated[
        str | None,
        Query(
            description=(
                "Exclusive window end (ISO date or datetime); omit for an "
                "open-ended window ending at NOW()."
            )
        ),
    ] = None,
    top_n: Annotated[int, Query(ge=1, le=200)] = 20,
    model: Annotated[
        str | None, Query(description="Filter to one model_name.")
    ] = None,
    role: Annotated[str | None, Query(description="Filter to one role.")] = None,
    operation: Annotated[
        str | None,
        Query(description="Filter to one operation (chat_completion|embedding)."),
    ] = None,
    call_status: Annotated[
        str | None,
        Query(alias="status", description="Filter to one call status."),
    ] = None,
) -> list[CategorySpendOut]:
    since_dt, until_dt = _parse_window(since, until, default_days=30)
    rows = await ai_costs_service.by_category(
        db,
        dimension=dimension,
        since=since_dt,
        until=until_dt,
        top_n=top_n,
        model=model,
        role=role,
        operation=operation,
        status=call_status,
    )
    return [CategorySpendOut.model_validate(r) for r in rows]


@router.get("/by-model", response_model=list[ModelEfficiencyOut])
async def get_by_model(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    until: Annotated[
        str | None,
        Query(
            description=(
                "Exclusive window end (ISO date or datetime); omit for an "
                "open-ended window ending at NOW()."
            )
        ),
    ] = None,
    top_n: Annotated[int, Query(ge=1, le=200)] = 20,
    model: Annotated[
        str | None, Query(description="Filter to one model_name.")
    ] = None,
    role: Annotated[str | None, Query(description="Filter to one role.")] = None,
    operation: Annotated[
        str | None,
        Query(description="Filter to one operation (chat_completion|embedding)."),
    ] = None,
    call_status: Annotated[
        str | None,
        Query(alias="status", description="Filter to one call status."),
    ] = None,
) -> list[ModelEfficiencyOut]:
    since_dt, until_dt = _parse_window(since, until, default_days=30)
    rows = await ai_costs_service.by_model(
        db,
        since=since_dt,
        until=until_dt,
        top_n=top_n,
        model=model,
        role=role,
        operation=operation,
        status=call_status,
    )
    return [ModelEfficiencyOut.model_validate(r) for r in rows]


@router.get("/recent", response_model=list[RecentCallOut])
async def get_recent(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RecentCallOut]:
    rows = await ai_costs_service.recent(db, limit=limit)
    return [RecentCallOut.model_validate(r) for r in rows]


__all__ = ["router"]
