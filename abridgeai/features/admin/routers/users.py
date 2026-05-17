"""Users router -- ``/admin/users`` (T7.5).

* Reads (list, detail) require ``user.read`` or ``system.administer``.
* Disable / enable require ``system.administer`` (destructive on session state).

Disable maps to ``users.status='inactive'`` (the existing CHECK constraint
allows 'active','invited','inactive','suspended' -- there is no 'disabled' --
so we use the closest semantic match) AND revokes every active auth_session.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_permission,
)
from abridgeai.features.admin.routers._scope import resolve_admin_scope
from abridgeai.features.admin.services import users as users_service

router = APIRouter(prefix="/admin/users", tags=["admin", "users"])

_REQUIRE_READ = require_any_permission("user.read", "system.administer")
_REQUIRE_WRITE = require_permission("system.administer")


class UserListRow(BaseModel):
    user_id: UUID
    primary_email: str
    status: str
    last_login_at: Any | None = None
    created_at: Any
    updated_at: Any


class DisableUserOut(BaseModel):
    user_id: UUID
    status: str
    revoked_session_count: int


class EnableUserOut(BaseModel):
    user_id: UUID
    status: str


@router.get("", response_model=list[UserListRow])
async def list_users(
    user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    user_status: Annotated[str | None, Query(alias="status")] = None,
    role_code: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UserListRow]:
    org_id = await resolve_admin_scope(db, user)
    rows = await users_service.list_users(
        db,
        status_filter=user_status,
        role_code=role_code,
        organization_id=org_id,
        limit=limit,
        offset=offset,
    )
    return [UserListRow.model_validate(r) for r in rows]


@router.get("/{user_id}")
async def get_user_detail(
    user_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    try:
        detail = await users_service.user_detail(db, user_id=user_id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "resource": "user", "id": str(user_id)},
        ) from exc
    return detail


@router.post("/{user_id}/disable", response_model=DisableUserOut)
async def disable_user(
    user_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DisableUserOut:
    try:
        result = await users_service.disable_user(db, user_id=user_id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "resource": "user", "id": str(user_id)},
        ) from exc
    return DisableUserOut(**result)


@router.post("/{user_id}/enable", response_model=EnableUserOut)
async def enable_user(
    user_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EnableUserOut:
    try:
        result = await users_service.enable_user(db, user_id=user_id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "resource": "user", "id": str(user_id)},
        ) from exc
    return EnableUserOut(**result)


__all__ = ["router"]
