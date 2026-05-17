"""Admin router for the access-control feature (T1.10 / FIX-CRIT-4).

**FIX-CRIT-4 invariant**: every endpoint declared in this module MUST
have a ``Depends(require_*)`` dependency on the operation. Routes that
only depend on :func:`get_current_user` (the legacy bug at
``backend/app/routes/access_control/router.py:26``) are forbidden -- the
companion test ``test_every_endpoint_has_permission_dependency`` walks
the ``router.routes`` tree and asserts at least one ``require_*``
factory in each route's dependency chain.

Permission map:

* catalog reads -> ``require_any_permission("system.administer",
  "audit.read")``
* assignments / grants -> ``require_any_permission("user.role_assign",
  "user.role_assign.hod", "system.administer")``
* org memberships -> ``require_any_permission("org_unit.manage",
  "user.bulk_import", "system.administer")``

The routers/services boundary (import-linter contract #2) is preserved:
this module imports
:mod:`abridgeai.features.access_control.services.admin`, never
:mod:`abridgeai.features.access_control.queries`.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.access_control.schemas.admin import (
    GrantCreate,
    GrantRead,
    MembershipCreate,
    MembershipRead,
    PermissionRead,
    RoleAssignmentCreate,
    RoleAssignmentRead,
    RoleWithPermissionsRead,
)
from abridgeai.features.access_control.services import admin as admin_service

router = APIRouter(tags=["admin", "access_control"])


_REQUIRE_CATALOG = require_any_permission("system.administer", "audit.read")
_REQUIRE_ASSIGN = require_any_permission(
    "user.role_assign", "user.role_assign.hod", "system.administer"
)
_REQUIRE_ORG_MANAGE = require_any_permission(
    "org_unit.manage", "user.bulk_import", "system.administer"
)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": "scope_invalid", "message": detail},
    )


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "forbidden", "message": detail},
    )


@router.get("/admin/permissions", response_model=list[PermissionRead])
async def list_permissions(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_CATALOG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PermissionRead]:
    rows = await admin_service.list_permission_catalog(db)
    return [PermissionRead.model_validate(r) for r in rows]


@router.get("/admin/roles", response_model=list[RoleWithPermissionsRead])
async def list_roles(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_CATALOG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RoleWithPermissionsRead]:
    pairs = await admin_service.list_role_catalog(db)
    return [RoleWithPermissionsRead(role=role, permissions=codes) for role, codes in pairs]


@router.get(
    "/admin/users/{user_id}/assignments",
    response_model=list[RoleAssignmentRead],
)
async def list_user_assignments(
    user_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RoleAssignmentRead]:
    rows = await admin_service.list_user_assignments(db, user_id)
    return [RoleAssignmentRead.model_validate(r) for r in rows]


@router.post(
    "/admin/users/{user_id}/assignments",
    response_model=RoleAssignmentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_assignment(
    user_id: UUID,
    payload: RoleAssignmentCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoleAssignmentRead:
    try:
        assignment = await admin_service.create_role_assignment(
            db,
            user_id=user_id,
            payload=payload,
            actor_id=current_user.user_id,
            actor_permissions=current_user.permissions,
        )
    except admin_service.ScopeValidationError as exc:
        raise _bad_request(str(exc)) from exc
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ForbiddenError as exc:
        raise _forbidden(str(exc)) from exc
    await db.commit()
    return RoleAssignmentRead.model_validate(assignment)


@router.delete(
    "/admin/users/{user_id}/assignments/{assignment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_user_assignment(
    user_id: UUID,
    assignment_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    del user_id
    try:
        await admin_service.revoke_role_assignment(db, assignment_id, actor_id=current_user.user_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@router.get(
    "/admin/users/{user_id}/grants",
    response_model=list[GrantRead],
)
async def list_user_grants(
    user_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[GrantRead]:
    rows = await admin_service.list_user_grants(db, user_id)
    return [GrantRead.model_validate(r) for r in rows]


@router.post(
    "/admin/users/{user_id}/grants",
    response_model=GrantRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_grant(
    user_id: UUID,
    payload: GrantCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GrantRead:
    try:
        grant = await admin_service.create_permission_grant(
            db,
            user_id=user_id,
            payload=payload,
            actor_id=current_user.user_id,
        )
    except admin_service.ScopeValidationError as exc:
        raise _bad_request(str(exc)) from exc
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return GrantRead.model_validate(grant)


@router.delete(
    "/admin/users/{user_id}/grants/{grant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_user_grant(
    user_id: UUID,
    grant_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    del user_id
    try:
        await admin_service.revoke_permission_grant(db, grant_id, actor_id=current_user.user_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@router.get(
    "/admin/organizations/{org_id}/memberships",
    response_model=list[MembershipRead],
)
async def list_org_memberships(
    org_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MembershipRead]:
    rows = await admin_service.list_organization_memberships(db, org_id)
    return [MembershipRead.model_validate(r) for r in rows]


@router.post(
    "/admin/organizations/{org_id}/memberships",
    response_model=MembershipRead,
    status_code=status.HTTP_201_CREATED,
)
async def add_org_membership(
    org_id: UUID,
    payload: MembershipCreate,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MembershipRead:
    membership = await admin_service.add_organization_membership(
        db, organization_id=org_id, payload=payload
    )
    await db.commit()
    return MembershipRead.model_validate(membership)


__all__ = ["router"]
