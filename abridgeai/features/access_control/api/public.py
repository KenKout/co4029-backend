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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import exists, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.access_control.api._dto import (
    OrgDTO,
    OrgUnitDTO,
    PermissionDTO,
    RoleAssignmentDTO,
)
from abridgeai.features.access_control.models import (
    Organization,
    OrganizationDomain,
    OrganizationMembership,
    OrgUnit,
    Permission,
    Role,
    RolePermission,
    UserFacultyAssignment,
    UserRoleAssignment,
)
from abridgeai.features.access_control.policies import (
    can_manage_course,
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
    require_permission,
)
from abridgeai.features.access_control.queries import organizations as org_queries


@dataclass(frozen=True)
class FacultyAccessDTO:
    faculty_ids: tuple[UUID, ...]
    has_organization_scope: bool


async def get_user_faculty_access(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
    permission_code: str,
) -> FacultyAccessDTO:
    """Faculty affiliations plus whether ``permission_code`` is org-wide.

    Affiliation chooses the owning faculty for new resources.  The scoped
    permission flag is kept separate so a Master Dean may deliberately create
    an organization-wide resource without being assigned to a faculty.
    """
    faculty_ids = await org_queries.list_active_faculty_ids_for_user(db, user_id, organization_id)
    stmt = (
        select(UserRoleAssignment.scope_kind)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .join(RolePermission, RolePermission.role_id == Role.id)
        .join(Permission, Permission.id == RolePermission.permission_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            Permission.code == permission_code,
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= datetime.now(UTC),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > datetime.now(UTC)),
            Permission.deleted_at.is_(None),
            Role.deleted_at.is_(None),
            or_(
                UserRoleAssignment.scope_kind == "global",
                (
                    (UserRoleAssignment.scope_kind == "organization")
                    & (UserRoleAssignment.organization_id == organization_id)
                ),
            ),
        )
        .limit(1)
    )
    has_org = (await db.execute(stmt)).scalar_one_or_none() is not None
    return FacultyAccessDTO(tuple(faculty_ids), has_org)


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


async def get_org_unit_subtree_ids(db: AsyncSession, org_unit_id: UUID) -> list[UUID]:
    """``org_unit_id`` plus every live unit BELOW it.

    The descending counterpart to :func:`get_org_unit_ancestors`. Scope
    filtering ("show me everything in this faculty") needs the subtree;
    permission checks ("is this course inside the HOD's unit?") need the
    ancestor chain. Both directions now have a blessed cross-feature entry
    point, so callers stop re-implementing the walk.

    Returns ``[]`` when the unit is missing or soft-deleted — callers
    should treat that as "no scope", not "no filter", or a deleted unit
    would silently widen a list to the whole organization.
    """
    return await org_queries.list_descendant_unit_ids(db, org_unit_id)


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


async def find_auto_provision_org_id(db: AsyncSession, *, email_domain: str) -> UUID | None:
    """Organization id whose registered email domain auto-provisions accounts.

    FR-2.7/FR-2.9: matches ``organization_domains.domain`` (CITEXT —
    case-insensitive exact match, no subdomain wildcards) with
    ``auto_provision = TRUE`` against an ``active`` organization. Returns
    ``None`` when no such domain exists — callers keep the invite-only
    rejection in that case.
    """
    stmt = (
        select(OrganizationDomain.organization_id)
        .join(Organization, Organization.id == OrganizationDomain.organization_id)
        .where(
            OrganizationDomain.domain == email_domain,
            OrganizationDomain.auto_provision.is_(True),
            OrganizationDomain.deleted_at.is_(None),
            Organization.status == "active",
            Organization.deleted_at.is_(None),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def grant_default_student_access(
    db: AsyncSession, *, user_id: UUID, organization_id: UUID
) -> None:
    """Attach an auto-provisioned user to an org with the Student role.

    Creates an active ``organization_memberships`` row plus an
    org-scoped ``user_role_assignments`` row for the ``student`` role
    (least privilege; ``granted_by`` stays NULL — system-granted).
    Idempotent at the caller level: only invoked for brand-new users.
    """
    role_id = (
        await db.execute(select(Role.id).where(Role.code == "student", Role.deleted_at.is_(None)))
    ).scalar_one()
    db.add(
        OrganizationMembership(
            user_id=user_id,
            organization_id=organization_id,
            status="active",
        )
    )
    db.add(
        UserRoleAssignment(
            user_id=user_id,
            role_id=role_id,
            scope_kind="organization",
            organization_id=organization_id,
        )
    )
    await db.flush()


async def grant_org_role_access(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
    role_code: str,
    granted_by: UUID | None = None,
) -> None:
    """Attach an existing user to an org with a chosen role (admin invite).

    Same shape as :func:`grant_default_student_access` but the role is
    selected by code (``student`` / ``teacher`` / ``manager`` / ...) and
    ``granted_by`` records the admin who invited the account. Used by the
    admin ``POST /users`` invite flow. Raises ``NoResultFound`` when
    ``role_code`` does not exist.
    """
    role_id = (
        await db.execute(select(Role.id).where(Role.code == role_code, Role.deleted_at.is_(None)))
    ).scalar_one_or_none()
    if role_id is None:
        raise NotFoundError(f"role '{role_code}' not found")
    db.add(
        OrganizationMembership(
            user_id=user_id,
            organization_id=organization_id,
            status="active",
        )
    )
    db.add(
        UserRoleAssignment(
            user_id=user_id,
            role_id=role_id,
            scope_kind="organization",
            organization_id=organization_id,
            granted_by=granted_by,
        )
    )
    await db.flush()


async def get_active_permissions(
    db: AsyncSession, user_id: UUID, *, at: datetime | None = None
) -> list[PermissionDTO]:
    rows = (
        await db.execute(_ACTIVE_PERMISSIONS_SQL, {"user_id": user_id, "at": _now_at(at)})
    ).mappings()
    return [PermissionDTO.model_validate(dict(row)) for row in rows]


async def get_role_codes_for_users(
    db: AsyncSession, user_ids: Sequence[UUID], *, at: datetime | None = None
) -> dict[UUID, list[str]]:
    """Batch-resolve the distinct active role codes for many users at once.

    Backs the admin user-list "Role" column: one query for the whole page
    instead of N per-user lookups. Returns ``{user_id: [role_code, ...]}`` with
    codes sorted alphabetically; users with no active assignment are absent
    from the dict (callers default to an empty list). Soft-deleted assignments
    / roles and elapsed ``active_until`` windows are excluded, matching
    :func:`get_role_assignments_for_user`.
    """
    if not user_ids:
        return {}
    at_value = _now_at(at)
    stmt = (
        select(UserRoleAssignment.user_id, Role.code)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id.in_(list(user_ids)),
            UserRoleAssignment.deleted_at.is_(None),
            Role.deleted_at.is_(None),
            or_(
                UserRoleAssignment.scope_kind != "org_unit",
                exists(
                    select(UserFacultyAssignment.id).where(
                        UserFacultyAssignment.user_id == UserRoleAssignment.user_id,
                        UserFacultyAssignment.faculty_id == UserRoleAssignment.org_unit_id,
                        UserFacultyAssignment.status == "active",
                        UserFacultyAssignment.deleted_at.is_(None),
                        (UserFacultyAssignment.active_until.is_(None))
                        | (UserFacultyAssignment.active_until > at_value),
                    )
                ),
            ),
            UserRoleAssignment.active_from <= at_value,
            or_(
                UserRoleAssignment.active_until.is_(None),
                UserRoleAssignment.active_until > at_value,
            ),
        )
        .distinct()
    )
    result: dict[UUID, set[str]] = {}
    for user_id, code in (await db.execute(stmt)).all():
        result.setdefault(user_id, set()).add(code)
    return {user_id: sorted(codes) for user_id, codes in result.items()}


async def list_user_ids_with_role(
    db: AsyncSession, role_code: str, *, at: datetime | None = None
) -> list[UUID]:
    """User ids holding an active assignment of ``role_code`` (any scope).

    Backs the admin user-list role filter: the identity search intersects its
    result with this id set rather than joining access-control tables itself
    (feature independence). Scope is not narrowed — a user with the role at any
    scope (global / org / org_unit / course) matches.
    """
    at_value = _now_at(at)
    stmt = (
        select(UserRoleAssignment.user_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            Role.code == role_code,
            UserRoleAssignment.deleted_at.is_(None),
            Role.deleted_at.is_(None),
            UserRoleAssignment.active_from <= at_value,
            or_(
                UserRoleAssignment.active_until.is_(None),
                UserRoleAssignment.active_until > at_value,
            ),
        )
        .distinct()
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_primary_orgs_for_users(
    db: AsyncSession, user_ids: Sequence[UUID]
) -> dict[UUID, OrgDTO]:
    """Batch-resolve each user's primary organization (one query for the page).

    Backs the admin user-list "Organization" column. Primary = the most recent
    active, non-deleted membership (matches :func:`get_user_primary_org`).
    Users with no active membership are absent from the dict.
    """
    if not user_ids:
        return {}
    om = OrganizationMembership
    # DISTINCT ON picks the newest active membership per user in one pass.
    stmt = (
        select(om.user_id, Organization)
        .join(Organization, Organization.id == om.organization_id)
        .where(
            om.user_id.in_(list(user_ids)),
            om.status == "active",
            om.deleted_at.is_(None),
            Organization.deleted_at.is_(None),
        )
        .order_by(om.user_id, om.created_at.desc().nullslast())
        .distinct(om.user_id)
    )
    result: dict[UUID, OrgDTO] = {}
    for user_id, org in (await db.execute(stmt)).all():
        if user_id not in result:
            result[user_id] = OrgDTO.model_validate(org, from_attributes=True)
    return result


async def list_user_ids_in_org(db: AsyncSession, org_id: UUID) -> list[UUID]:
    """User ids with an active, non-deleted membership in ``org_id``.

    Backs the admin user-list organization filter (identity search intersects
    its result with this id set).
    """
    om = OrganizationMembership
    stmt = (
        select(om.user_id)
        .where(
            om.organization_id == org_id,
            om.status == "active",
            om.deleted_at.is_(None),
        )
        .distinct()
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_user_ids_in_org_unit(db: AsyncSession, org_unit_id: UUID) -> list[UUID]:
    """User ids with an active affiliation to a live top-level Faculty.

    The legacy function name is retained for API compatibility while the
    physical ``org_units`` table now represents Faculties only. Students are
    absent because they never receive a Faculty affiliation.
    """
    stmt = (
        select(UserFacultyAssignment.user_id)
        .join(OrgUnit, OrgUnit.id == UserFacultyAssignment.faculty_id)
        .where(
            UserFacultyAssignment.faculty_id == org_unit_id,
            UserFacultyAssignment.status == "active",
            UserFacultyAssignment.deleted_at.is_(None),
            or_(
                UserFacultyAssignment.active_until.is_(None),
                UserFacultyAssignment.active_until > _now_at(None),
            ),
            OrgUnit.unit_type == "faculty",
            OrgUnit.parent_unit_id.is_(None),
            OrgUnit.deleted_at.is_(None),
        )
        .distinct()
    )
    return list((await db.execute(stmt)).scalars().all())


@dataclass(frozen=True)
class InactiveOrgDTO:
    """One organization that has gone quiet.

    ``last_activity_at`` / ``days_quiet`` are ``None`` when nothing was ever
    recorded for the tenant. That is a different state from "quiet since
    March" and the caller must be able to tell them apart — a brand-new
    organization and an abandoned one both cross the threshold, but only one
    of them is a problem.
    """

    id: UUID
    name: str
    slug: str
    last_activity_at: datetime | None
    days_quiet: int | None


async def list_inactive_organizations(
    db: AsyncSession,
    *,
    days: int,
    now: datetime | None = None,
    organization_id: UUID | None = None,
) -> list[InactiveOrgDTO]:
    """Organizations with no activity in ``days`` days.

    The one definition of tenant inactivity, shared by the operator
    dashboard's count and the organizations list that count links to. Exposed
    here because organizations belong to access_control while the surfaces that
    need this live in admin, and cross-feature traffic goes through the public
    API rather than reaching into another feature's queries.
    """
    rows = await org_queries.list_inactive_organizations(
        db,
        now=_now_at(now),
        days=days,
        organization_id=organization_id,
    )
    return [
        InactiveOrgDTO(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            last_activity_at=row["last_activity_at"],
            days_quiet=row["days_quiet"],
        )
        for row in rows
    ]


@dataclass(frozen=True)
class RoleDTO:
    """A role from the catalogue.

    Carries the display ``name`` alongside the ``code`` so a caller can label
    a role without keeping its own copy of the wording — the whole point of
    reading the catalogue rather than hardcoding "Faculty Dean" a second time.
    """

    id: UUID
    code: str
    name: str


async def list_roles(db: AsyncSession) -> list[RoleDTO]:
    """Every live role, ordered by display name."""
    stmt = select(Role).where(Role.deleted_at.is_(None)).order_by(Role.name)
    rows = (await db.execute(stmt)).scalars().all()
    return [RoleDTO(id=r.id, code=r.code, name=r.name) for r in rows]


async def get_roles_by_codes(db: AsyncSession, codes: Sequence[str]) -> dict[str, RoleDTO]:
    """``{code: RoleDTO}`` for the given codes; unknown codes are absent.

    Returning a mapping rather than a list lets the caller detect unknown
    codes by set difference instead of scanning.
    """
    if not codes:
        return {}
    stmt = select(Role).where(Role.code.in_(list(codes)), Role.deleted_at.is_(None))
    rows = (await db.execute(stmt)).scalars().all()
    return {r.code: RoleDTO(id=r.id, code=r.code, name=r.name) for r in rows}


__all__ = [
    "RoleDTO",
    "get_roles_by_codes",
    "list_roles",
    "FacultyAccessDTO",
    "InactiveOrgDTO",
    "list_inactive_organizations",
    "get_org_unit_subtree_ids",
    "list_user_ids_in_org_unit",
    "OrgDTO",
    "OrgUnitDTO",
    "PermissionDTO",
    "RoleAssignmentDTO",
    "can_manage_course",
    "find_auto_provision_org_id",
    "get_active_permissions",
    "get_user_faculty_access",
    "get_org_unit_ancestors",
    "get_role_assignments_for_user",
    "get_role_codes_for_users",
    "get_primary_orgs_for_users",
    "grant_default_student_access",
    "grant_org_role_access",
    "get_user_primary_org",
    "is_user_member_of_org",
    "list_user_ids_in_org",
    "list_user_ids_with_role",
    "require_any_permission",
    "require_course_permission",
    "require_org_unit_permission",
    "require_permission",
]
