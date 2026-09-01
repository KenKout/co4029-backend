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
  "audit.read", "user.role_assign", "user.role_assign.hod")``
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
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_org_access,
)
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


# Catalog reads feed the assignment UI: the SPA loads /admin/roles to build
# the role filter and to resolve role ids when posting an assignment. Gating
# the catalog tighter than the assignment write (`_REQUIRE_ASSIGN` below)
# left a Dean holding user.role_assign(+.hod) able to POST /admin/users/{id}/
# assignments but unable to load the catalog his own management-users page
# needs — the promote-manager flow threw "Role catalog not loaded" on a 403.
# A role catalog is read-only, non-org data; the write gates stay as-is.
_REQUIRE_CATALOG = require_any_permission(
    "system.administer",
    "audit.read",
    "user.role_assign",
    "user.role_assign.hod",
)
_REQUIRE_ASSIGN = require_any_permission(
    "user.role_assign", "user.role_assign.hod", "system.administer"
)
# Shared by the dependency and the per-resource org check so the two cannot
# drift: the first asks "held anywhere?", the second "held for THIS org?".
_ORG_MEMBERSHIP_CODES = ("org_unit.manage", "user.bulk_import", "system.administer")
_REQUIRE_ORG_MANAGE = require_any_permission(*_ORG_MEMBERSHIP_CODES)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ASSIGN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RoleAssignmentRead]:
    rows = await admin_service.list_user_assignments(
        db,
        user_id,
        actor_id=current_user.user_id,
        actor_permissions=current_user.permissions,
    )
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
        await admin_service.revoke_role_assignment(
            db,
            assignment_id,
            actor_id=current_user.user_id,
            actor_permissions=current_user.permissions,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ForbiddenError as exc:
        raise _forbidden(str(exc)) from exc
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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MembershipRead]:
    # Memberships carry student_code / employee_code — PII of another
    # customer's staff and students if this is not scoped.
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MEMBERSHIP_CODES,
    )
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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MembershipRead:
    # The most consequential org check in the codebase. Writing a membership
    # into an arbitrary organization is not merely a cross-tenant write — it
    # MANUFACTURES the membership that every other org check in this codebase
    # relies on. Left open, a manager grants themselves entry to any tenant and
    # every `require_org_access` guard then passes legitimately.
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MEMBERSHIP_CODES,
    )
    membership = await admin_service.add_organization_membership(
        db, organization_id=org_id, payload=payload
    )
    await db.commit()
    return MembershipRead.model_validate(membership)


__all__ = ["router"]
