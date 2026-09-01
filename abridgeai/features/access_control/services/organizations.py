"""Admin services for organization / org-unit / domain / membership management.

Routers call into this module; the service layer composes
:mod:`abridgeai.features.access_control.queries.organizations` plus
business rules (uniqueness checks, scope validation, status-transition
side effects).

Import-linter posture mirrors :mod:`services.admin`: this module does
not import :mod:`sqlalchemy` directly (contract #1). The ``AsyncSession``
parameter is annotated under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.pagination import (
    CursorPage,
    Page,
    decode_composite_cursor,
    encode_composite_cursor,
)
from abridgeai.features.access_control.models import (
    Organization,
    OrganizationDomain,
    OrganizationMembership,
    OrgUnit,
    UserFacultyAssignment,
)
from abridgeai.features.access_control.queries import organizations as org_queries
from abridgeai.features.access_control.schemas.admin import (
    BulkAssignUnitRequest,
    BulkAssignUnitResult,
    FacultyMembersAddRequest,
    MembershipPatch,
    OrganizationCreate,
    OrganizationDomainCreate,
    OrganizationDomainPatch,
    OrganizationPatch,
    OrgUnitCreate,
    OrgUnitNode,
    OrgUnitPatch,
)

if TYPE_CHECKING:
    from abridgeai.core.db import AsyncSession  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tenancy resolvers
#
# The routes for domains, units and memberships address the resource by its
# OWN id, so there is no ``org_id`` in the path for the permission dependency
# to scope against — and the flat permission set behind that dependency does
# not distinguish an ``org_unit.manage`` granted in org B from a global one.
# These resolve the owning organization so the router can call
# ``require_org_access`` before touching anything. ``None`` means the resource
# does not exist (or is soft-deleted); the router turns that into the same 404
# a foreign-org caller gets, so the two are indistinguishable.
# ---------------------------------------------------------------------------


async def organization_ids_for_user(db: AsyncSession, user_id: UUID) -> list[UUID]:
    """Organizations the user actively belongs to.

    The visibility set for org list/search. An empty list is a real answer —
    a user with no membership sees no organizations — and must not be confused
    with ``None``, which means "unrestricted".
    """
    return await org_queries.organization_ids_for_user(db, user_id)


async def organization_id_for_domain(db: AsyncSession, domain_id: UUID) -> UUID | None:
    row = await org_queries.get_domain(db, domain_id)
    if row is None or row.deleted_at is not None:
        return None
    return row.organization_id


async def organization_id_for_unit(db: AsyncSession, unit_id: UUID) -> UUID | None:
    row = await org_queries.get_unit(db, unit_id)
    if row is None or row.deleted_at is not None:
        return None
    return row.organization_id


async def organization_id_for_membership(db: AsyncSession, membership_id: UUID) -> UUID | None:
    row = await org_queries.get_membership(db, membership_id)
    if row is None or row.deleted_at is not None:
        return None
    return row.organization_id


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


async def list_organizations(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
    visible_to_ids: list[UUID] | None = None,
) -> CursorPage[Organization]:
    """Cursor-paginated organisation list ordered by ``(name ASC, id ASC)``.

    ``cursor`` is opaque (round-trips through
    :func:`encode_composite_cursor` / :func:`decode_composite_cursor`).
    ``next_cursor`` is set when the page filled to ``limit`` (more rows
    may exist); ``None`` otherwise.
    """
    after_name: str | None = None
    after_id: UUID | None = None
    if cursor:
        sort_value, last_id = decode_composite_cursor(cursor)
        if not isinstance(sort_value, str):
            raise ValueError("Invalid cursor")
        after_name = sort_value
        after_id = last_id
    rows = await org_queries.list_organizations(
        db,
        include_deleted=include_deleted,
        status=status,
        limit=limit,
        after_name=after_name,
        after_id=after_id,
        visible_to_ids=visible_to_ids,
    )
    next_cursor = (
        encode_composite_cursor(rows[-1].name, rows[-1].id) if len(rows) == limit else None
    )
    return CursorPage(items=rows, next_cursor=next_cursor)


async def search_organizations(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
    search: str | None = None,
    sort: str | None = None,
    sort_dir: str = "asc",
    page: int = 0,
    page_size: int = 25,
    visible_to_ids: list[UUID] | None = None,
    inactive_days: int | None = None,
    now: datetime | None = None,
) -> Page[Organization]:
    """Offset page of organisations (server-side search + sort). Thin
    delegate to the query layer, which owns the SQLAlchemy statement.

    ``visible_to_ids=None`` is unrestricted and reserved for
    ``system.administer``; see the query-layer docstring.

    ``inactive_days`` narrows to tenants with no activity in that many days,
    using the same definition the dashboard counts (see
    ``queries/sql/inactive_organizations.sql``). It composes with
    ``visible_to_ids`` by intersection rather than replacing it — an inactivity
    filter must never widen what a scoped caller can see.
    """
    if inactive_days is not None:
        inactive_rows = await org_queries.list_inactive_organizations(
            db, now=now or datetime.now(tz=UTC), days=inactive_days
        )
        inactive_ids = [row["id"] for row in inactive_rows]
        if visible_to_ids is None:
            visible_to_ids = inactive_ids
        else:
            allowed = set(visible_to_ids)
            visible_to_ids = [i for i in inactive_ids if i in allowed]
        if not visible_to_ids:
            # No tenant qualifies. An empty allowlist must page to zero rows —
            # passing None here would silently mean "unrestricted".
            return Page(
                items=[],
                total=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )

    return await org_queries.search_organizations(
        db,
        include_deleted=include_deleted,
        status=status,
        search=search,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
        visible_to_ids=visible_to_ids,
    )


async def count_organizations(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
) -> int:
    return await org_queries.count_organizations(db, include_deleted=include_deleted, status=status)


async def get_organization(db: AsyncSession, organization_id: UUID) -> Organization:
    row = await org_queries.get_organization(db, organization_id)
    if row is None or row.deleted_at is not None:
        raise NotFoundError("Organization not found")
    return row


async def create_organization(
    db: AsyncSession,
    payload: OrganizationCreate,
) -> Organization:
    existing = await org_queries.get_organization_by_slug(db, payload.slug)
    if existing is not None:
        raise AppError(f"Organization slug '{payload.slug}' already exists")
    return await org_queries.insert_organization(
        db,
        slug=payload.slug,
        name=payload.name,
        status=payload.status,
    )


async def patch_organization(
    db: AsyncSession,
    organization_id: UUID,
    payload: OrganizationPatch,
) -> Organization:
    current = await org_queries.get_organization(db, organization_id)
    if current is None or current.deleted_at is not None:
        raise NotFoundError("Organization not found")
    fields = payload.model_dump(exclude_unset=True)
    if "slug" in fields and fields["slug"] != current.slug:
        clash = await org_queries.get_organization_by_slug(db, fields["slug"])
        if clash is not None and clash.id != organization_id:
            raise AppError(f"Organization slug '{fields['slug']}' already exists")
    updated = await org_queries.update_organization(db, organization_id, fields=fields)
    if updated is None:
        raise NotFoundError("Organization not found")
    return updated


async def delete_organization(
    db: AsyncSession,
    organization_id: UUID,
    *,
    actor_id: UUID | None,
) -> None:
    deleted = await org_queries.soft_delete_organization(db, organization_id, actor_id=actor_id)
    if not deleted:
        raise NotFoundError("Organization not found")


# ---------------------------------------------------------------------------
# Organization domains
# ---------------------------------------------------------------------------


async def list_domains(db: AsyncSession, organization_id: UUID) -> list[OrganizationDomain]:
    await get_organization(db, organization_id)  # raises 404 if missing
    return await org_queries.list_domains_for_organization(db, organization_id)


async def create_domain(
    db: AsyncSession,
    organization_id: UUID,
    payload: OrganizationDomainCreate,
) -> OrganizationDomain:
    await get_organization(db, organization_id)
    existing = await org_queries.get_domain_by_name(db, payload.domain)
    if existing is not None:
        raise AppError(f"Domain '{payload.domain}' is already mapped")
    return await org_queries.insert_domain(
        db,
        organization_id=organization_id,
        domain=payload.domain,
        auto_provision=payload.auto_provision,
    )


async def patch_domain(
    db: AsyncSession,
    domain_id: UUID,
    payload: OrganizationDomainPatch,
) -> OrganizationDomain:
    current = await org_queries.get_domain(db, domain_id)
    if current is None or current.deleted_at is not None:
        raise NotFoundError("Domain not found")
    fields = payload.model_dump(exclude_unset=True)
    if "domain" in fields and fields["domain"] != current.domain:
        clash = await org_queries.get_domain_by_name(db, fields["domain"])
        if clash is not None and clash.id != domain_id:
            raise AppError(f"Domain '{fields['domain']}' is already mapped")
    updated = await org_queries.update_domain(db, domain_id, fields=fields)
    if updated is None:
        raise NotFoundError("Domain not found")
    return updated


async def delete_domain(
    db: AsyncSession,
    domain_id: UUID,
    *,
    actor_id: UUID | None,
) -> None:
    deleted = await org_queries.soft_delete_domain(db, domain_id, actor_id=actor_id)
    if not deleted:
        raise NotFoundError("Domain not found")


# ---------------------------------------------------------------------------
# Org units
# ---------------------------------------------------------------------------


async def list_units(
    db: AsyncSession,
    organization_id: UUID,
    *,
    parent_unit_id: UUID | None = None,
    only_roots: bool = False,
) -> list[OrgUnit]:
    await get_organization(db, organization_id)
    return await org_queries.list_units_for_organization(
        db,
        organization_id,
        parent_unit_id=parent_unit_id,
        only_roots=only_roots,
    )


async def list_unit_tree(db: AsyncSession, organization_id: UUID) -> list[OrgUnitNode]:
    """The org's units as a nested tree, in ONE query.

    Every unit in the org is fetched flat and assembled in memory rather
    than walked level by level — the tree UI needs the whole thing to
    render expand/collapse, and a per-level fetch would be a round-trip
    per expanded node.

    ``descendant_count`` rides along because the destructive actions need
    it: deleting a unit takes its whole subtree with it
    (``soft_delete_cascade`` follows ``OrgUnit.children``), so the confirm
    dialog has to be able to say how many units that is BEFORE the click,
    not after.

    A unit whose ``parent_unit_id`` points at a row that is missing or
    soft-deleted is surfaced as a ROOT rather than dropped. Losing a
    department from the manager's tree because its faculty was deleted
    would hide real courses and real people; showing it at top level is
    visibly odd, which is the point.
    """
    rows = await org_queries.list_units_for_organization(db, organization_id)
    # Built field by field rather than via ``model_validate(row)``. ``_ORM``
    # sets ``from_attributes=True``, and ``OrgUnitNode.children`` shares its
    # name with ``OrgUnit.children`` — a default-lazy relationship. Validating
    # the ORM row therefore READS that relationship, which on an async session
    # is a lazy load outside the greenlet and raises MissingGreenlet (HTTP
    # 500) instead of returning a tree. The children we want are the ones
    # assembled below from ``parent_unit_id``, not whatever the ORM would
    # emit a query for.
    nodes: dict[UUID, OrgUnitNode] = {
        row.id: OrgUnitNode(
            id=row.id,
            organization_id=row.organization_id,
            parent_unit_id=row.parent_unit_id,
            unit_type=row.unit_type,
            name=row.name,
            code=row.code,
            created_at=row.created_at,
            updated_at=row.updated_at,
            children=[],
            descendant_count=0,
        )
        for row in rows
    }
    roots: list[OrgUnitNode] = []
    for row in rows:
        node = nodes[row.id]
        parent = nodes.get(row.parent_unit_id) if row.parent_unit_id else None
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)

    def _count(node: OrgUnitNode) -> int:
        node.descendant_count = sum(1 + _count(child) for child in node.children)
        node.children.sort(key=lambda n: n.name.casefold())
        return node.descendant_count

    for root in roots:
        _count(root)
    roots.sort(key=lambda n: n.name.casefold())
    return roots


async def get_unit(db: AsyncSession, unit_id: UUID) -> OrgUnit:
    row = await org_queries.get_unit(db, unit_id)
    if row is None or row.deleted_at is not None:
        raise NotFoundError("Org unit not found")
    return row


async def create_unit(
    db: AsyncSession,
    organization_id: UUID,
    payload: OrgUnitCreate,
    *,
    actor_id: UUID,
) -> OrgUnit:
    await get_organization(db, organization_id)
    if not await org_queries.user_has_role_scope(
        db,
        user_id=actor_id,
        role_code="hod",
        organization_id=organization_id,
    ):
        raise ForbiddenError("only a Master Dean may create a faculty")
    return await org_queries.insert_unit(
        db,
        organization_id=organization_id,
        parent_unit_id=None,
        unit_type="faculty",
        name=payload.name,
        code=payload.code,
    )


async def patch_unit(
    db: AsyncSession,
    unit_id: UUID,
    payload: OrgUnitPatch,
    *,
    actor_id: UUID,
) -> OrgUnit:
    current = await get_unit(db, unit_id)
    if not await org_queries.user_has_role_scope(
        db,
        user_id=actor_id,
        role_code="hod",
        organization_id=current.organization_id,
    ):
        raise ForbiddenError("only a Master Dean may edit a faculty")
    fields = payload.model_dump(exclude_unset=True)
    if current.unit_type != "faculty" or current.parent_unit_id is not None:
        raise AppError("only top-level faculties can be edited")
    updated = await org_queries.update_unit(db, unit_id, fields=fields)
    if updated is None:
        raise NotFoundError("Org unit not found")
    return updated


async def assign_memberships_to_unit(
    db: AsyncSession,
    organization_id: UUID,
    payload: BulkAssignUnitRequest,
) -> BulkAssignUnitResult:
    """Move a set of memberships into ``org_unit_id`` in one statement.

    Two checks stand between a request body and the write, and both matter:

    1. The target unit must exist, be live, and belong to ``organization_id``
       — otherwise a manager could file their people under another tenant's
       department, which would then leak them into that tenant's scope
       filters.
    2. EVERY membership id must belong to ``organization_id``. Ids come from
       the client, and a foreign id is not skipped quietly — the whole call
       is rejected. A silent partial write here is worse than an error: the
       caller believes the cohort moved, and nobody looks again.

    Ids that are simply missing or soft-deleted ARE tolerated and reported
    in ``skipped``; that is a stale selection, not an attack, and failing the
    whole batch because one person was removed mid-flow would be hostile.
    """
    await get_organization(db, organization_id)

    if payload.org_unit_id is not None:
        unit = await org_queries.get_unit(db, payload.org_unit_id)
        if unit is None or unit.deleted_at is not None:
            raise AppError("org_unit_id does not exist or is deleted")
        if unit.organization_id != organization_id:
            raise AppError("org_unit_id belongs to a different organization")

    rows = await org_queries.list_memberships_by_ids(db, payload.membership_ids)
    found = {row.id: row for row in rows}

    foreign = [r.id for r in rows if r.organization_id != organization_id]
    if foreign:
        raise AppError("membership_ids contains memberships from a different organization")

    skipped = [mid for mid in payload.membership_ids if mid not in found]
    assigned = await org_queries.bulk_update_membership_unit(
        db, [r.id for r in rows], payload.org_unit_id
    )
    return BulkAssignUnitResult(assigned=assigned, skipped=skipped)


async def delete_unit(
    db: AsyncSession,
    unit_id: UUID,
    *,
    actor_id: UUID,
) -> None:
    unit = await get_unit(db, unit_id)
    if not await org_queries.user_has_role_scope(
        db,
        user_id=actor_id,
        role_code="hod",
        organization_id=unit.organization_id,
    ):
        raise ForbiddenError("only a Master Dean may archive a faculty")
    if unit.unit_type != "faculty":
        raise AppError("only faculties can be archived")
    if await org_queries.faculty_has_live_dependencies(db, unit_id):
        raise AppError(
            "faculty_has_dependencies: remove or archive its programs, courses, "
            "and staff assignments first"
        )
    deleted = await org_queries.soft_delete_unit(db, unit_id, actor_id=actor_id)
    if not deleted:
        raise NotFoundError("Org unit not found")


async def _faculty_management_level(
    db: AsyncSession,
    *,
    actor_id: UUID,
    organization_id: UUID,
    faculty_id: UUID,
) -> str | None:
    """Return ``master`` / ``dean`` for an authorized Faculty Dean."""
    if await org_queries.user_has_role_scope(
        db,
        user_id=actor_id,
        role_code="hod",
        organization_id=organization_id,
    ):
        return "master"
    if await org_queries.user_has_role_scope(
        db,
        user_id=actor_id,
        role_code="hod",
        organization_id=organization_id,
        faculty_id=faculty_id,
    ):
        return "dean"
    return None


async def list_faculty_assignments(
    db: AsyncSession,
    organization_id: UUID,
    *,
    faculty_id: UUID | None = None,
    actor_id: UUID,
    allow_system_admin: bool = False,
) -> list[UserFacultyAssignment]:
    await get_organization(db, organization_id)
    if faculty_id is not None:
        faculty = await get_unit(db, faculty_id)
        if faculty.organization_id != organization_id or faculty.unit_type != "faculty":
            raise NotFoundError("Faculty not found")
    rows = await org_queries.list_faculty_assignments(db, organization_id, faculty_id=faculty_id)
    if allow_system_admin or await org_queries.user_has_role_scope(
        db,
        user_id=actor_id,
        role_code="hod",
        organization_id=organization_id,
    ):
        visible_rows = rows
    else:
        actor_faculty_ids = set(
            await org_queries.list_active_faculty_ids_for_user(db, actor_id, organization_id)
        )
        if faculty_id is not None and faculty_id not in actor_faculty_ids:
            raise ForbiddenError("you may only view staff in your assigned faculties")
        visible_rows = [row for row in rows if row.faculty_id in actor_faculty_ids]

    roles_by_assignment = await org_queries.faculty_role_codes(
        db, [(row.user_id, row.faculty_id) for row in visible_rows]
    )
    for row in visible_rows:
        row.role_codes = roles_by_assignment.get((row.user_id, row.faculty_id), [])
    return visible_rows


async def add_faculty_members(
    db: AsyncSession,
    faculty_id: UUID,
    payload: FacultyMembersAddRequest,
    *,
    actor_id: UUID,
) -> list[UserFacultyAssignment]:
    faculty = await get_unit(db, faculty_id)
    if faculty.unit_type != "faculty" or faculty.parent_unit_id is not None:
        raise AppError("faculty_id must reference a live top-level faculty")
    level = await _faculty_management_level(
        db,
        actor_id=actor_id,
        organization_id=faculty.organization_id,
        faculty_id=faculty_id,
    )
    if level is None:
        raise ForbiddenError("only the Faculty Dean may assign faculty staff")
    if actor_id in payload.user_ids:
        raise ForbiddenError("you cannot add yourself to a faculty")

    members = await org_queries.active_org_member_user_ids(
        db, faculty.organization_id, payload.user_ids
    )
    missing = [user_id for user_id in payload.user_ids if user_id not in members]
    if missing:
        raise AppError("all selected staff must be active members of this organization")

    result: list[UserFacultyAssignment] = []
    for user_id in payload.user_ids:
        role_codes = await org_queries.active_role_codes_for_user(
            db, user_id, faculty.organization_id
        )
        staff_roles = role_codes & {"hod", "manager", "teacher"}
        if not staff_roles:
            raise AppError(
                f"user {user_id} is not eligible faculty staff; students are not assignable"
            )
        if level != "master" and "hod" in role_codes:
            raise ForbiddenError("a Faculty Dean cannot assign another Faculty Dean")
        existing = await org_queries.get_active_faculty_assignment(
            db, user_id=user_id, faculty_id=faculty_id
        )
        if existing is not None:
            result.append(existing)
            continue
        result.append(
            await org_queries.insert_faculty_assignment(
                db,
                user_id=user_id,
                organization_id=faculty.organization_id,
                faculty_id=faculty_id,
                actor_id=actor_id,
            )
        )
    return result


async def remove_faculty_member(
    db: AsyncSession,
    faculty_id: UUID,
    user_id: UUID,
    *,
    actor_id: UUID,
) -> None:
    faculty = await get_unit(db, faculty_id)
    level = await _faculty_management_level(
        db,
        actor_id=actor_id,
        organization_id=faculty.organization_id,
        faculty_id=faculty_id,
    )
    if level is None:
        raise ForbiddenError("only the Faculty Dean may remove faculty staff")
    if actor_id == user_id:
        raise ForbiddenError("you cannot remove yourself from a faculty")
    role_codes = await org_queries.active_role_codes_for_user(db, user_id, faculty.organization_id)
    if level != "master" and "hod" in role_codes:
        raise ForbiddenError("a Faculty Dean cannot remove another Faculty Dean")
    assignment = await org_queries.get_active_faculty_assignment(
        db, user_id=user_id, faculty_id=faculty_id
    )
    if assignment is None:
        raise NotFoundError("Faculty staff assignment not found")
    await org_queries.deactivate_faculty_assignment(db, assignment, actor_id=actor_id)


# ---------------------------------------------------------------------------
# Memberships (extend existing list/insert in services.admin)
# ---------------------------------------------------------------------------


async def patch_membership(
    db: AsyncSession,
    membership_id: UUID,
    payload: MembershipPatch,
) -> OrganizationMembership:
    current = await org_queries.get_membership(db, membership_id)
    if current is None or current.deleted_at is not None:
        raise NotFoundError("Membership not found")
    fields = payload.model_dump(exclude_unset=True)
    if "org_unit_id" in fields and fields["org_unit_id"] is not None:
        unit = await org_queries.get_unit(db, fields["org_unit_id"])
        if unit is None or unit.deleted_at is not None:
            raise AppError("org_unit_id does not exist or is deleted")
        if unit.organization_id != current.organization_id:
            raise AppError("org_unit_id belongs to a different organization")
    if fields.get("status") == "left" and current.status != "left" and current.left_at is None:
        fields["left_at"] = org_queries.now_utc()
    updated = await org_queries.update_membership(db, membership_id, fields=fields)
    if updated is None:
        raise NotFoundError("Membership not found")
    return updated


async def delete_membership(
    db: AsyncSession,
    membership_id: UUID,
    *,
    actor_id: UUID | None,
) -> None:
    """Mark a membership as ``left`` instead of soft-deleting it.

    ``left`` is a business state, not a deletion: the row stays visible in
    the roster with its ``left_at`` timestamp, and every org-scope check in
    the codebase filters on ``status == 'active'``, so a left member loses
    all org access immediately (same effect as the old soft-delete, but
    with the history preserved). ``actor_id`` is accepted for call-site
    compatibility; the leave is attributed via ``left_at``.
    """
    current = await org_queries.get_membership(db, membership_id)
    if current is None or current.deleted_at is not None:
        raise NotFoundError("Membership not found")
    fields: dict[str, Any] = {"status": "left"}
    if current.left_at is None:
        fields["left_at"] = org_queries.now_utc()
    updated = await org_queries.update_membership(db, membership_id, fields=fields)
    if updated is None:
        raise NotFoundError("Membership not found")


__all__ = [
    "add_faculty_members",
    "count_organizations",
    "create_domain",
    "create_organization",
    "create_unit",
    "delete_domain",
    "delete_membership",
    "delete_organization",
    "delete_unit",
    "get_organization",
    "get_unit",
    "list_domains",
    "list_faculty_assignments",
    "list_organizations",
    "list_units",
    "patch_domain",
    "patch_membership",
    "patch_organization",
    "patch_unit",
    "remove_faculty_member",
]
