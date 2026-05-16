"""Admin-surface services for the access-control feature.

Routers (T1.10) call into this module; this module composes
:mod:`abridgeai.features.access_control.queries.admin` plus business rules
the database CHECK constraints can also catch (scope_kind / FK shape) so
the API returns a 422 with a useful body before Postgres raises a generic
constraint-violation error.

Import-linter posture: services do not import :mod:`sqlalchemy` directly
(contract #1). The ``AsyncSession`` parameter is annotated under
``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import ForbiddenError, NotFoundError
from abridgeai.features.access_control.models import (
    OrganizationMembership,
    UserPermissionGrant,
    UserRoleAssignment,
)
from abridgeai.features.access_control.queries import admin as admin_queries
from abridgeai.features.access_control.schemas.admin import (
    GrantCreate,
    MembershipCreate,
    RoleAssignmentCreate,
)

if TYPE_CHECKING:
    from abridgeai.core.db import AsyncSession  # type: ignore[attr-defined]

VALID_SCOPE_KINDS = frozenset({"global", "organization", "org_unit", "course"})

HOD_GATED_ROLES = frozenset({"hod"})


class ScopeValidationError(ValueError):
    """Raised when ``(scope_kind, organization_id, org_unit_id, course_id)``
    does not satisfy the baseline DDL CHECK constraint
    ``ck_user_role_assignments_scope_kind`` /
    ``ck_user_permission_grants_scope_kind``."""


def _validate_scope_shape(
    *,
    scope_kind: str,
    organization_id: UUID | None,
    org_unit_id: UUID | None,
    course_id: UUID | None,
) -> None:
    if scope_kind not in VALID_SCOPE_KINDS:
        raise ScopeValidationError(
            f"scope_kind must be one of {sorted(VALID_SCOPE_KINDS)}; got {scope_kind!r}"
        )

    if scope_kind == "global":
        if organization_id is not None or org_unit_id is not None or course_id is not None:
            raise ScopeValidationError(
                "scope_kind='global' requires organization_id/org_unit_id/course_id all NULL"
            )
        return

    if scope_kind == "organization":
        if organization_id is None or org_unit_id is not None or course_id is not None:
            raise ScopeValidationError(
                "scope_kind='organization' requires organization_id NOT NULL "
                "and org_unit_id/course_id NULL"
            )
        return

    if scope_kind == "org_unit":
        if organization_id is None or org_unit_id is None or course_id is not None:
            raise ScopeValidationError(
                "scope_kind='org_unit' requires organization_id AND org_unit_id NOT NULL "
                "and course_id NULL"
            )
        return

    if organization_id is None or course_id is None:
        raise ScopeValidationError(
            "scope_kind='course' requires organization_id AND course_id NOT NULL"
        )


async def list_permission_catalog(db: AsyncSession) -> list[object]:
    return list(await admin_queries.list_permissions(db))


async def list_role_catalog(
    db: AsyncSession,
) -> list[tuple[object, list[str]]]:
    """Return ``[(role, [permission_code, ...]), ...]`` for the catalog endpoint."""
    roles = await admin_queries.list_roles(db)
    out: list[tuple[object, list[str]]] = []
    for role in roles:
        codes = await admin_queries.get_role_permission_codes(db, role.id)
        out.append((role, codes))
    return out


async def list_user_assignments(db: AsyncSession, user_id: UUID) -> list[UserRoleAssignment]:
    return await admin_queries.list_assignments_for_user(db, user_id)


async def create_role_assignment(
    db: AsyncSession,
    *,
    user_id: UUID,
    payload: RoleAssignmentCreate,
    actor_id: UUID,
    actor_permissions: frozenset[str],
) -> UserRoleAssignment:
    """Create a role assignment for ``user_id``.

    Enforces:

    * scope-kind / FK shape (replicates the baseline DDL CHECK constraint
      so the API responds with 422 before Postgres throws
      ``IntegrityError``).
    * HOD-promotion gate: assigning a role in :data:`HOD_GATED_ROLES`
      requires the caller to hold ``user.role_assign.hod`` (or
      ``system.administer``). Plain ``user.role_assign`` (held by
      ``manager`` per T1.3 ``role_seeds.yaml``) is NOT sufficient. Admin
      holds ALL so promotes freely.
    """
    role = await admin_queries.get_role_by_code(db, payload.role_code)
    if role is None:
        raise NotFoundError(f"role '{payload.role_code}' not found")

    if payload.role_code in HOD_GATED_ROLES and not (
        "user.role_assign.hod" in actor_permissions or "system.administer" in actor_permissions
    ):
        raise ForbiddenError(
            f"role '{payload.role_code}' requires the 'user.role_assign.hod' permission"
        )

    _validate_scope_shape(
        scope_kind=payload.scope_kind,
        organization_id=payload.organization_id,
        org_unit_id=payload.org_unit_id,
        course_id=payload.course_id,
    )

    return await admin_queries.insert_assignment(
        db,
        user_id=user_id,
        role_id=role.id,
        scope_kind=payload.scope_kind,
        organization_id=payload.organization_id,
        org_unit_id=payload.org_unit_id,
        course_id=payload.course_id,
        granted_by=actor_id,
        active_until=payload.active_until,
    )


async def revoke_role_assignment(db: AsyncSession, assignment_id: UUID) -> None:
    if not await admin_queries.soft_delete_assignment(db, assignment_id):
        raise NotFoundError(f"role assignment {assignment_id} not found")


async def list_user_grants(db: AsyncSession, user_id: UUID) -> list[UserPermissionGrant]:
    return await admin_queries.list_grants_for_user(db, user_id)


async def create_permission_grant(
    db: AsyncSession,
    *,
    user_id: UUID,
    payload: GrantCreate,
    actor_id: UUID,
) -> UserPermissionGrant:
    permission = await admin_queries.get_permission_by_id(db, payload.permission_id)
    if permission is None:
        raise NotFoundError(f"permission {payload.permission_id} not found")

    _validate_scope_shape(
        scope_kind=payload.scope_kind,
        organization_id=payload.organization_id,
        org_unit_id=payload.org_unit_id,
        course_id=payload.course_id,
    )

    return await admin_queries.insert_grant(
        db,
        user_id=user_id,
        permission_id=payload.permission_id,
        scope_kind=payload.scope_kind,
        organization_id=payload.organization_id,
        org_unit_id=payload.org_unit_id,
        course_id=payload.course_id,
        granted_by=actor_id,
        expires_at=payload.expires_at,
    )


async def revoke_permission_grant(db: AsyncSession, grant_id: UUID) -> None:
    if not await admin_queries.delete_grant(db, grant_id):
        raise NotFoundError(f"permission grant {grant_id} not found")


async def list_organization_memberships(
    db: AsyncSession, organization_id: UUID
) -> list[OrganizationMembership]:
    return await admin_queries.list_memberships_for_organization(db, organization_id)


async def add_organization_membership(
    db: AsyncSession,
    *,
    organization_id: UUID,
    payload: MembershipCreate,
) -> OrganizationMembership:
    return await admin_queries.insert_membership(
        db,
        user_id=payload.user_id,
        organization_id=organization_id,
        org_unit_id=payload.org_unit_id,
        status=payload.status,
        student_code=payload.student_code,
        employee_code=payload.employee_code,
    )


__all__ = [
    "HOD_GATED_ROLES",
    "ScopeValidationError",
    "VALID_SCOPE_KINDS",
    "add_organization_membership",
    "create_permission_grant",
    "create_role_assignment",
    "list_organization_memberships",
    "list_permission_catalog",
    "list_role_catalog",
    "list_user_assignments",
    "list_user_grants",
    "revoke_permission_grant",
    "revoke_role_assignment",
]
