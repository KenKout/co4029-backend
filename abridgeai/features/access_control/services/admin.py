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

# Roles that only a platform admin (``system.administer``) may assign or
# revoke. A HOD manages managers, not other HODs.
ADMIN_ONLY_ROLES = frozenset({"hod"})

# Roles that escalate beyond the plain ``user.role_assign`` authority a
# manager holds. Assigning or revoking these requires ``user.role_assign.hod``
# (held by the HOD role) or ``system.administer``. ``manager`` is gated so a
# manager cannot self-replicate by promoting peers to manager.
HOD_GATED_ROLES = frozenset({"hod", "manager"})


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


async def list_user_assignments(
    db: AsyncSession,
    user_id: UUID,
    *,
    actor_id: UUID | None = None,
    actor_permissions: frozenset[str] | None = None,
) -> list[UserRoleAssignment]:
    """List a user's role assignments, org-scoped for non-admin callers.

    Admin (``system.administer``) sees everything. Any other caller sees
    only the assignments that belong to orgs where they hold assign
    authority (``user.role_assign`` / ``user.role_assign.hod``) — a
    manager inspecting a teacher never learns about that teacher's roles
    in other tenants.
    """
    rows = await admin_queries.list_assignments_for_user(db, user_id)
    if actor_permissions and "system.administer" in actor_permissions:
        return rows
    if actor_id is None:
        return []
    assign_orgs, has_global = await admin_queries.assign_org_ids_for_user(db, actor_id)
    if has_global:
        return rows
    return [r for r in rows if r.organization_id in assign_orgs]


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
    * no self-assignment: a caller may not grant a role to themselves.
    * no global scope for non-admins: ``scope_kind='global'`` is
      platform-wide authority and requires ``system.administer``.
    * org-scope for non-admins: the assignment's org must be one where
      the caller holds ``user.role_assign`` / ``user.role_assign.hod``
      (``has_global`` bypasses). A manager cannot assign roles in a
      tenant they have no authority in, even if the target user belongs
      to it.
    * HOD-promotion gate: assigning a role in :data:`HOD_GATED_ROLES`
      (``hod``, ``manager``) requires the caller to hold
      ``user.role_assign.hod`` (or ``system.administer``). Plain
      ``user.role_assign`` (held by ``manager`` per T1.3
      ``role_seeds.yaml``) is NOT sufficient — this is what stops a
      manager from promoting peers to manager. Admin holds ALL so
      promotes freely.
    * admin-only roles: assigning a role in :data:`ADMIN_ONLY_ROLES`
      (``hod``) additionally requires ``system.administer`` — a HOD
      manages managers, never other HODs.
    """
    role = await admin_queries.get_role_by_code(db, payload.role_code)
    if role is None:
        raise NotFoundError(f"role '{payload.role_code}' not found")

    if user_id == actor_id and "system.administer" not in actor_permissions:
        raise ForbiddenError("you cannot assign a role to yourself")

    if payload.role_code in ADMIN_ONLY_ROLES and (
        "system.administer" not in actor_permissions
    ):
        raise ForbiddenError(
            f"role '{payload.role_code}' requires the 'system.administer' permission"
        )

    if payload.role_code in HOD_GATED_ROLES and not (
        "user.role_assign.hod" in actor_permissions
        or "system.administer" in actor_permissions
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

    if "system.administer" not in actor_permissions:
        if payload.scope_kind == "global":
            raise ForbiddenError(
                "scope_kind='global' requires the 'system.administer' permission"
            )
        assign_orgs, has_global = await admin_queries.assign_org_ids_for_user(
            db, actor_id
        )
        if not has_global and payload.organization_id not in assign_orgs:
            raise ForbiddenError(
                "you can only assign roles in organizations where you hold "
                "'user.role_assign' or 'user.role_assign.hod'"
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


async def revoke_role_assignment(
    db: AsyncSession,
    assignment_id: UUID,
    *,
    actor_id: UUID | None = None,
    actor_permissions: frozenset[str] | None = None,
) -> None:
    """Revoke a role assignment, org-scoped for non-admin callers.

    Mirrors :func:`create_role_assignment`'s gates on the way out: the
    caller must hold authority in the assignment's org, and revoking a
    :data:`HOD_GATED_ROLES` role requires ``user.role_assign.hod`` —
    a plain manager cannot strip a peer's manager role to cover their
    tracks, and nobody revokes their own assignment (self-lockout guard).
    """
    pair = await admin_queries.get_assignment_by_id(db, assignment_id)
    if pair is None:
        raise NotFoundError(f"role assignment {assignment_id} not found")
    assignment, role_code = pair

    if actor_permissions and "system.administer" in actor_permissions:
        await admin_queries.soft_delete_assignment(
            db, assignment_id, actor_id=actor_id
        )
        return

    if actor_id is None:
        raise ForbiddenError("caller identity required to revoke an assignment")
    if assignment.user_id == actor_id:
        raise ForbiddenError("you cannot revoke your own role assignment")
    if role_code in ADMIN_ONLY_ROLES and not (
        actor_permissions and "system.administer" in actor_permissions
    ):
        raise ForbiddenError(
            f"role '{role_code}' requires the 'system.administer' permission"
        )
    if role_code in HOD_GATED_ROLES and not (
        actor_permissions and "user.role_assign.hod" in actor_permissions
    ):
        raise ForbiddenError(
            f"role '{role_code}' requires the 'user.role_assign.hod' permission"
        )

    if assignment.scope_kind == "global":
        raise ForbiddenError(
            "global role assignments require the 'system.administer' permission"
        )
    assign_orgs, has_global = await admin_queries.assign_org_ids_for_user(
        db, actor_id
    )
    if not has_global and assignment.organization_id not in assign_orgs:
        raise ForbiddenError(
            "you can only revoke role assignments in organizations where you "
            "hold 'user.role_assign' or 'user.role_assign.hod'"
        )

    await admin_queries.soft_delete_assignment(db, assignment_id, actor_id=actor_id)


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


async def revoke_permission_grant(
    db: AsyncSession,
    grant_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> None:
    if not await admin_queries.delete_grant(db, grant_id, actor_id=actor_id):
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
