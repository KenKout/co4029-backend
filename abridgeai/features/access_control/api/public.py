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

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.api._dto import (
    OrgDTO,
    OrgUnitDTO,
    PermissionDTO,
    RoleAssignmentDTO,
)
from abridgeai.features.access_control.policies import (
    can_manage_course,
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
    require_permission,
)

_ROLE_ASSIGNMENTS_FOR_USER_SQL = text(
    """
    SELECT ura.id,
           ura.user_id,
           ura.role_id,
           r.code AS role_code,
           ura.scope_kind,
           ura.organization_id,
           ura.org_unit_id,
           ura.course_id,
           ura.active_from,
           ura.active_until
    FROM user_role_assignments ura
    JOIN roles r ON r.id = ura.role_id
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND r.deleted_at IS NULL
      AND ura.active_from <= :at
      AND (ura.active_until IS NULL OR ura.active_until > :at)
    ORDER BY ura.active_from DESC, ura.id
    """
)


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


_USER_PRIMARY_ORG_SQL = text(
    """
    SELECT o.id, o.slug, o.name, o.status
    FROM user_role_assignments ura
    JOIN organizations o ON o.id = ura.organization_id
    WHERE ura.user_id = :user_id
      AND ura.scope_kind IN ('organization', 'org_unit', 'course')
      AND ura.organization_id IS NOT NULL
      AND ura.deleted_at IS NULL
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
      AND o.deleted_at IS NULL
    ORDER BY ura.created_at DESC NULLS LAST
    LIMIT 1
    """
)


_IS_MEMBER_OF_ORG_SQL = text(
    """
    SELECT 1
    FROM organization_memberships
    WHERE user_id = :user_id
      AND organization_id = :org_id
      AND status = 'active'
      AND deleted_at IS NULL
    LIMIT 1
    """
)


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
    rows = (
        await db.execute(
            _ROLE_ASSIGNMENTS_FOR_USER_SQL,
            {"user_id": user_id, "at": _now_at(at)},
        )
    ).mappings()
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
    """Resolve the user's most recent active org-scoped assignment.

    Replaces the duplicated ``get_user_primary_organization_id`` helpers in
    ``features/courses/queries/published.py`` and
    ``features/career_paths/queries/published.py``: returns a typed
    :class:`OrgDTO` rather than just an id. ``scope_kind='global'`` is
    intentionally excluded: platform admins are not implicitly members.
    """
    row = (await db.execute(_USER_PRIMARY_ORG_SQL, {"user_id": user_id})).mappings().one_or_none()
    if row is None:
        return None
    return OrgDTO.model_validate(dict(row))


async def is_user_member_of_org(db: AsyncSession, *, user_id: UUID, org_id: UUID) -> bool:
    result = await db.execute(_IS_MEMBER_OF_ORG_SQL, {"user_id": user_id, "org_id": org_id})
    return result.scalar_one_or_none() is not None


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
