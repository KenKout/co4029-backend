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
    if raw is None:
        return ai_costs_service.default_since(default_days)
    try:
        if "T" in raw or " " in raw:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        else:
            parsed = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_since",
                "message": "since must be YYYY-MM-DD or ISO-8601 datetime",
            },
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


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
    since_dt = _parse_since(since, default_days=30)
    payload: dict[str, Any] = await ai_costs_service.summary(
        db,
        since=since_dt,
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
    top_n: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[UserSpendOut]:
    since_dt = _parse_since(since, default_days=30)
    rows = await ai_costs_service.by_user(
        db,
        since=since_dt,
        top_n=top_n,
        is_today_window=ai_costs_service.is_today_window(since_dt),
    )
    return [UserSpendOut.model_validate(r) for r in rows]


@router.get("/by-pipeline", response_model=list[PipelineSpendOut])
async def get_by_pipeline(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[
        str | None,
        Query(description="ISO date or datetime; defaults to NOW() - 30 days."),
    ] = None,
    top_n: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[PipelineSpendOut]:
    since_dt = _parse_since(since, default_days=30)
    rows = await ai_costs_service.by_pipeline(db, since=since_dt, top_n=top_n)
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
    since_dt = _parse_since(since, default_days=30)
    rows = await ai_costs_service.by_category(
        db,
        dimension=dimension,
        since=since_dt,
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
    since_dt = _parse_since(since, default_days=30)
    rows = await ai_costs_service.by_model(
        db,
        since=since_dt,
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
