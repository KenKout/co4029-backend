"""Public, typed cross-feature read API for the access_control feature.

This module is the *only* path other features may use to reach into
access_control. Sibling modules (``policies``, ``queries``, ``services``)
remain feature-internal.

The decorator re-exports below collapse the per-route ignore_imports
contract entries in ``pyproject.toml``: cross-feature routers can now
``from abridgeai.features.access_control.api.public import
require_permission`` and the ``Features are independent`` linter contract
will be satisfied without exceptions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import exists, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.api._dto import (
    OrgDTO,
    OrgUnitDTO,
    PermissionDTO,
    RoleAssignmentDTO,
)
from abridgeai.features.access_control.models import (
    Organization,
    OrganizationMembership,
    Role,
    UserRoleAssignment,
)
from abridgeai.features.access_control.policies import (
    can_manage_course,
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
    require_permission,
)

# Recursive CTE — the cleanest expression of the ancestor walk is the raw
# SQL form. SQLAlchemy supports recursive CTEs via Query.cte(recursive=True)
# but the ORM equivalent is harder to read than the inline ``text()``.
_ORG_UNIT_ANCESTORS_SQL = text(
    """
    WITH RECURSIVE org_unit_tree AS (
        SELECT id,
               organization_id,
               parent_unit_id,
               unit_type,
               name,
               code,
               0 AS depth
        FROM org_units
        WHERE id = :org_unit_id
          AND deleted_at IS NULL

        UNION

        SELECT ou.id,
               ou.organization_id,
               ou.parent_unit_id,
               ou.unit_type,
               ou.name,
               ou.code,
               t.depth + 1 AS depth
        FROM org_units ou
        JOIN org_unit_tree t ON ou.id = t.parent_unit_id
        WHERE ou.deleted_at IS NULL
    )
    SELECT id, organization_id, parent_unit_id, unit_type, name, code, depth
    FROM org_unit_tree
    ORDER BY depth ASC
    """
)


# UNION across two source tables (role-derived + direct grants) with
# DISTINCT semantics. The ORM equivalent (``select().union(select())``)
# is verbose and the column projection is small + well-documented, so
# the raw form here is the right tool.
_ACTIVE_PERMISSIONS_SQL = text(
    """
    SELECT DISTINCT p.id, p.code, p.name, p.description
    FROM permissions p
    JOIN role_permissions rp ON rp.permission_id = p.id
    JOIN user_role_assignments ura ON ura.role_id = rp.role_id
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.active_from <= :at
      AND (ura.active_until IS NULL OR ura.active_until > :at)
      AND p.deleted_at IS NULL

    UNION

    SELECT DISTINCT p.id, p.code, p.name, p.description
    FROM permissions p
    JOIN user_permission_grants upg ON upg.permission_id = p.id
    WHERE upg.user_id = :user_id
      AND upg.deleted_at IS NULL
      AND (upg.expires_at IS NULL OR upg.expires_at > :at)
      AND p.deleted_at IS NULL
    ORDER BY code
    """
)


def _now_at(at: datetime | None) -> datetime:
    return at if at is not None else datetime.now(UTC)


async def get_role_assignments_for_user(
    db: AsyncSession,
    user_id: UUID,
    *,
    at: datetime | None = None,
) -> list[RoleAssignmentDTO]:
    """Return all currently-active role assignments for ``user_id``.

    Includes every scope (global, organization, org_unit, course); the
    ``RoleAssignmentDTO.scope_kind`` field tells callers what they got.
    Filters out soft-deleted assignments and roles, and rows whose
    ``active_until`` window has elapsed.
    """
    at_value = _now_at(at)
    stmt = (
        select(
            UserRoleAssignment.id,
            UserRoleAssignment.user_id,
            UserRoleAssignment.role_id,
            Role.code.label("role_code"),
            UserRoleAssignment.scope_kind,
            UserRoleAssignment.organization_id,
            UserRoleAssignment.org_unit_id,
            UserRoleAssignment.course_id,
            UserRoleAssignment.active_from,
            UserRoleAssignment.active_until,
        )
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.deleted_at.is_(None),
            Role.deleted_at.is_(None),
            UserRoleAssignment.active_from <= at_value,
            or_(
                UserRoleAssignment.active_until.is_(None),
                UserRoleAssignment.active_until > at_value,
            ),
        )
        .order_by(UserRoleAssignment.active_from.desc(), UserRoleAssignment.id)
    )
    rows = (await db.execute(stmt)).mappings()
    return [RoleAssignmentDTO.model_validate(dict(row)) for row in rows]


async def get_org_unit_ancestors(db: AsyncSession, org_unit_id: UUID) -> list[OrgUnitDTO]:
    """Walk the org_unit tree from ``org_unit_id`` to root.

    Returns ``[self, parent, grandparent, ...]`` ordered by depth ascending,
    each row populated with its 0-indexed distance from the input. Returns
    an empty list when the unit does not exist or is soft-deleted.

    Mirrors the recursive CTE used by
    :data:`abridgeai.features.access_control.policies._ORG_UNIT_ANCESTORS_SQL`
    but additionally projects depth and the full :class:`OrgUnitDTO` shape
    so cross-feature callers do not need raw rows.
    """
    rows = (await db.execute(_ORG_UNIT_ANCESTORS_SQL, {"org_unit_id": org_unit_id})).mappings()
    return [OrgUnitDTO.model_validate(dict(row)) for row in rows]


async def get_user_primary_org(db: AsyncSession, user_id: UUID) -> OrgDTO | None:
    """Resolve the user's primary organization via membership.

    Sole source of truth: ``organization_memberships``. ``status='active'``
    only; soft-deleted rows excluded. When a user has multiple memberships
    the most recent (``created_at DESC``) wins.

    Role assignments are NOT consulted -- belonging-to-org and
    permissions-in-org are independent concepts. ``scope_kind='global'``
    is therefore irrelevant: platform admins are not implicitly members
    of any single org and must use endpoints that accept an explicit
    ``organization_id`` parameter. Returns ``None`` for users with no
    active membership.
    """
    stmt = (
        select(Organization)
        .join(
            OrganizationMembership,
            OrganizationMembership.organization_id == Organization.id,
        )
        .where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.status == "active",
            OrganizationMembership.deleted_at.is_(None),
            Organization.deleted_at.is_(None),
        )
        .order_by(OrganizationMembership.created_at.desc().nullslast())
        .limit(1)
    )
    org = (await db.execute(stmt)).scalar_one_or_none()
    return OrgDTO.model_validate(org, from_attributes=True) if org is not None else None


async def is_user_member_of_org(db: AsyncSession, *, user_id: UUID, org_id: UUID) -> bool:
    """Return True iff the user has an active, non-deleted membership in ``org_id``."""
    stmt = select(
        exists().where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id == org_id,
            OrganizationMembership.status == "active",
            OrganizationMembership.deleted_at.is_(None),
        )
    )
    return bool((await db.execute(stmt)).scalar())


async def get_active_permissions(
    db: AsyncSession, user_id: UUID, *, at: datetime | None = None
) -> list[PermissionDTO]:
    rows = (
        await db.execute(_ACTIVE_PERMISSIONS_SQL, {"user_id": user_id, "at": _now_at(at)})
    ).mappings()
    return [PermissionDTO.model_validate(dict(row)) for row in rows]


__all__ = [
    "OrgDTO",
    "OrgUnitDTO",
    "PermissionDTO",
    "RoleAssignmentDTO",
    "can_manage_course",
    "get_active_permissions",
    "get_org_unit_ancestors",
    "get_role_assignments_for_user",
    "get_user_primary_org",
    "is_user_member_of_org",
    "require_any_permission",
    "require_course_permission",
    "require_org_unit_permission",
    "require_permission",
]
