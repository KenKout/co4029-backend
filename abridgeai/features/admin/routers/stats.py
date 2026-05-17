"""Stats router -- ``/admin/stats`` (T7.5).

All four endpoints require ``system.stats.read`` OR ``system.administer``.
Org-scoping is resolved via :func:`resolve_admin_scope`.

* ``/overview`` and ``/content`` are bounded scans (top-level COUNT(*) over
  finite tables) -- no ``since`` parameter needed.
* ``/active-users`` uses fixed 24h / 7d / 30d windows.
* ``/health`` requires ``since`` to keep the failed-jobs / failed-AI-calls
  scans bounded.
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


__all__ = ["router"]
