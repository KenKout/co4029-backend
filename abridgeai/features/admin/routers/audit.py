"""Audit router -- ``/admin/audit`` (T7.5).

* ``/role-changes`` and ``/data-changes`` require ``audit.read`` or
  ``system.administer``.
* ``/http`` requires the same permissions AND degrades gracefully to 503
  when T0.23's ``http_audit_log`` table has not yet been deployed.

All scan-style endpoints REQUIRE ``since`` (FastAPI ``Query(...)`` enforces
the 400 at framework level when the param is missing).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.admin.routers._scope import resolve_admin_scope
from abridgeai.features.admin.services import audit as audit_service

router = APIRouter(prefix="/admin/audit", tags=["admin", "audit"])

_REQUIRE_AUDIT = require_any_permission("audit.read", "system.administer")


class RoleChangeRow(BaseModel):
    assignment_id: UUID
    user_id: UUID
    role_id: UUID
    role_code: str
    scope_kind: str
    organization_id: UUID | None = None
    org_unit_id: UUID | None = None
    course_id: UUID | None = None
    granted_by: UUID | None = None
    active_from: datetime
    active_until: datetime | None = None
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None
    created_at: datetime
    updated_at: datetime


class HttpAuditRow(BaseModel):
    id: UUID
    user_id: UUID | None = None
    session_id: UUID | None = None
    method: str
    path: str
    status_code: int
    latency_ms: int | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: datetime


class CourseDataChangeOut(BaseModel):
    entity_id: UUID
    title: str
    slug: str
    status: str
    created_by: UUID | None = None
    updated_by: UUID | None = None
    deleted_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    organization_id: UUID


@router.get("/role-changes", response_model=list[RoleChangeRow])
async def get_role_changes(
    user: Annotated[CurrentUser, Depends(_REQUIRE_AUDIT)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[datetime, Query(description="Lower bound on updated_at (required).")],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RoleChangeRow]:
    org_id = await resolve_admin_scope(db, user)
    rows = await audit_service.role_changes(db, since=since, organization_id=org_id, limit=limit)
    return [RoleChangeRow.model_validate(r) for r in rows]


@router.get("/data-changes")
async def get_data_changes(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_AUDIT)],
    db: Annotated[AsyncSession, Depends(get_db)],
    table: Annotated[str, Query(description="Entity table; only 'courses' supported in T7.5.")],
    entity_id: Annotated[UUID, Query(description="Target entity primary key.")],
) -> dict[str, Any]:
    if table != "courses":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unsupported_table",
                "message": "data-changes lookup currently supports table='courses' only",
            },
        )
    row = await audit_service.course_data_changes(db, entity_id=entity_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "resource": "course", "id": str(entity_id)},
        )
    return CourseDataChangeOut.model_validate(row).model_dump(mode="json")


@router.get("/http", response_model=list[HttpAuditRow])
async def search_http_audit(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_AUDIT)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[datetime, Query(description="Lower bound on created_at (required).")],
    user_id: Annotated[UUID | None, Query()] = None,
    path_pattern: Annotated[
        str | None,
        Query(description="SQL LIKE pattern, e.g. '/api/v1/admin/%'."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[HttpAuditRow]:
    try:
        rows = await audit_service.http_audit_search(
            db,
            since=since,
            user_id=user_id,
            path_pattern=path_pattern,
            limit=limit,
        )
    except audit_service.HttpAuditUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "audit_log_unavailable",
                "message": "audit log middleware not deployed yet",
            },
        ) from exc
    return [HttpAuditRow.model_validate(r) for r in rows]


__all__ = ["router"]
