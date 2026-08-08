"""Users router -- ``/admin/users`` (T7.5).

* Reads (list, detail) require ``user.read`` or ``system.administer``.
* Disable / enable require ``user.disable`` or ``system.administer``
  (org-scoped account administration: managers manage their own org's
  accounts, IT admins manage globally).

Disable maps to ``users.status='inactive'`` (the existing CHECK constraint
allows 'active','invited','inactive','suspended' -- there is no 'disabled' --
so we use the closest semantic match) AND revokes every active auth_session.

Guard matrix for disable / enable (besides the permission dependency):

* self -- a user can never disable their own account (403).
* ``system.administer`` -- full bypass (IT admin, global scope).
* otherwise (manager): target must be an active member of the caller's
  organization (404 when not, so cross-org ids stay indistinguishable from
  absent ids), and the target must not hold a *peer* role -- ``manager``,
  ``hod``, or ``admin`` -- at the caller's org (global, org, org_unit, or
  course scope resolving to that org). Peer accounts get 403: managers
  administer teachers and students, not each other.
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
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.admin.routers._scope import resolve_admin_scope
from abridgeai.features.admin.services import users as users_service
from abridgeai.features.courses.queries.cross_feature import get_course_org

router = APIRouter(prefix="/admin/users", tags=["admin", "users"])

_REQUIRE_READ = require_any_permission("user.read", "system.administer")
_REQUIRE_WRITE = require_any_permission("user.disable", "system.administer")

# Roles a manager may never disable / re-enable -- managers administer
# teachers and students, not their peers.
_PEER_ROLE_CODES = frozenset({"manager", "hod", "admin"})


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "forbidden", "message": detail},
    )


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


async def _is_peer_at_org(
    db: AsyncSession, *, user_id: UUID, org_id: UUID
) -> bool:
    """True when ``user_id`` holds a peer role (manager/hod/admin) whose scope
    resolves to ``org_id`` (global, org, org_unit, or course scope)."""
    assignments = await access_control_api.get_role_assignments_for_user(db, user_id)
    for assignment in assignments:
        if assignment.role_code not in _PEER_ROLE_CODES:
            continue
        if assignment.scope_kind == "global":
            return True
        if (
            assignment.scope_kind == "organization"
            and assignment.organization_id == org_id
        ):
            return True
        if assignment.scope_kind == "org_unit" and assignment.org_unit_id is not None:
            ancestors = await access_control_api.get_org_unit_ancestors(
                db, assignment.org_unit_id
            )
            if any(unit.organization_id == org_id for unit in ancestors):
                return True
        if assignment.scope_kind == "course" and assignment.course_id is not None:
            course_org = await get_course_org(db, assignment.course_id)
            if course_org == org_id:
                return True
    return False


async def _assert_can_manage_user(
    db: AsyncSession,
    current_user: CurrentUser,
    user_id: UUID,
) -> None:
    """Enforce the disable/enable guard matrix documented above."""
    if current_user.user_id == user_id:
        raise _forbidden("cannot disable or enable your own account")

    # IT admin: global operator, no org / peer restriction.
    if current_user.has_permission("system.administer"):
        return

    org_id = await resolve_admin_scope(db, current_user)
    if org_id is None:
        raise _forbidden("no organization scope for account management")

    if not await access_control_api.is_user_member_of_org(
        db, user_id=user_id, org_id=org_id
    ):
        raise _not_found("user not found in your organization")

    if await _is_peer_at_org(db, user_id=user_id, org_id=org_id):
        raise _forbidden(
            "cannot disable or enable a peer (manager/hod/admin) account"
        )


class UserListRow(BaseModel):
    user_id: UUID
    primary_email: str
    status: str
    display_name: str | None = None
    last_login_at: Any | None = None
    created_at: Any
    updated_at: Any
    role_codes: list[str] = []


class UserListPage(BaseModel):
    """Cursor-paginated admin user listing.

    ``next_cursor`` is opaque and round-trips through subsequent calls.
    Set when the page is full (more rows may exist); ``None`` otherwise.
    Reconciliation §A10/§D2: cursor pagination, not offset.
    """

    items: list[UserListRow]
    next_cursor: str | None = None


class DisableUserOut(BaseModel):
    user_id: UUID
    status: str
    revoked_session_count: int


class EnableUserOut(BaseModel):
    user_id: UUID
    status: str


@router.get("", response_model=UserListPage)
async def list_users(
    user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    user_status: Annotated[str | None, Query(alias="status")] = None,
    role_code: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> UserListPage:
    org_id = await resolve_admin_scope(db, user)
    try:
        page = await users_service.list_users(
            db,
            status_filter=user_status,
            role_code=role_code,
            organization_id=org_id,
            q=q,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_cursor", "message": str(exc)},
        ) from exc
    role_codes = await access_control_api.get_role_codes_for_users(
        db, [row["user_id"] for row in page.items]
    )
    return UserListPage(
        items=[
            UserListRow.model_validate(
                {**row, "role_codes": role_codes.get(row["user_id"], [])}
            )
            for row in page.items
        ],
        next_cursor=page.next_cursor,
    )


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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DisableUserOut:
    await _assert_can_manage_user(db, current_user, user_id)
    try:
        result = await users_service.disable_user(db, user_id=user_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return DisableUserOut(**result)


@router.post("/{user_id}/enable", response_model=EnableUserOut)
async def enable_user(
    user_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EnableUserOut:
    await _assert_can_manage_user(db, current_user, user_id)
    try:
        result = await users_service.enable_user(db, user_id=user_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return EnableUserOut(**result)


__all__ = ["router"]
