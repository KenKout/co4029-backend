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
    usd: float
    call_count: int


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
    usd: float
    latency_ms: int | None = None
    created_at: datetime
    pipeline_run_id: UUID | None = None


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
) -> SummaryOut:
    since_dt = _parse_since(since, default_days=30)
    payload: dict[str, Any] = await ai_costs_service.summary(db, since=since_dt, period=period)
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


@router.get("/recent", response_model=list[RecentCallOut])
async def get_recent(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RecentCallOut]:
    rows = await ai_costs_service.recent(db, limit=limit)
    return [RecentCallOut.model_validate(r) for r in rows]


__all__ = ["router"]
