"""Data accessors for organization / org-unit / domain / membership writes.

Conventions match :mod:`abridgeai.features.access_control.queries.admin`:

* SELECT via ``select(Model).where(...)``.
* INSERT via :func:`sqlalchemy.insert` against the mapped class — bypass
  the ORM unit-of-work walk that ``db.add()`` triggers (which fails
  with :class:`NoReferencedTableError` whenever a cross-feature FK
  target hasn't been imported into ``Base.metadata`` yet).
* UPDATE via ``update(Model).where(...).values(...)``.
* DELETE via :func:`abridgeai.core.db.recursive_delete.soft_delete_cascade`
  to preserve the audit trail and respect the ``hard_delete_guard``
  listener.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import resources
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import exists, func, insert, select, text, tuple_, update

from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.pagination import Page, paginate
from abridgeai.features.access_control.models import (
    Organization,
    OrganizationDomain,
    OrganizationMembership,
    OrgUnit,
    Role,
    UserFacultyAssignment,
    UserRoleAssignment,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


async def list_organizations(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
    limit: int = 100,
    after_name: str | None = None,
    after_id: UUID | None = None,
    visible_to_ids: list[UUID] | None = None,
) -> list[Organization]:
    stmt = select(Organization)
    if not include_deleted:
        stmt = stmt.where(Organization.deleted_at.is_(None))
    if status is not None:
        stmt = stmt.where(Organization.status == status)
    if visible_to_ids is not None:
        stmt = stmt.where(Organization.id.in_(visible_to_ids))
    if after_name is not None and after_id is not None:
        stmt = stmt.where(tuple_(Organization.name, Organization.id) > (after_name, after_id))
    stmt = stmt.order_by(Organization.name, Organization.id).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


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
) -> Page[Organization]:
    """Offset page of organisations with server-side search (name/slug) +
    whitelisted sort. Backs the page-numbered admin table.

    ``visible_to_ids`` restricts the result to those organizations. ``None``
    means unrestricted and is reserved for ``system.administer`` — every other
    caller passes the set it belongs to, because the permission guarding this
    route is a flat check that cannot tell an ``org_unit.manage`` granted in
    one tenant from a global one. Filtering in the query rather than the
    router keeps ``total`` and the cursor consistent with what is returned.
    """
    stmt = select(Organization)
    if not include_deleted:
        stmt = stmt.where(Organization.deleted_at.is_(None))
    if status is not None:
        stmt = stmt.where(Organization.status == status)
    if visible_to_ids is not None:
        stmt = stmt.where(Organization.id.in_(visible_to_ids))
    return await paginate(
        db,
        stmt,
        page=page,
        page_size=page_size,
        search=search,
        search_columns=[Organization.name, Organization.slug],
        sort=sort,
        sort_dir=sort_dir,
        sortable={
            "name": Organization.name,
            "status": Organization.status,
            "created_at": Organization.created_at,
        },
        default_order=[Organization.id],
    )


async def count_organizations(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
    visible_to_ids: list[UUID] | None = None,
) -> int:
    from sqlalchemy import func

    stmt = select(func.count(Organization.id))
    if not include_deleted:
        stmt = stmt.where(Organization.deleted_at.is_(None))
    if status is not None:
        stmt = stmt.where(Organization.status == status)
    if visible_to_ids is not None:
        stmt = stmt.where(Organization.id.in_(visible_to_ids))
    result = await db.execute(stmt)
    return int(result.scalar_one())


async def organization_ids_for_user(db: AsyncSession, user_id: UUID) -> list[UUID]:
    """Ids of organizations the user has an active membership in."""
    result = await db.execute(
        select(OrganizationMembership.organization_id).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.status == "active",
            OrganizationMembership.deleted_at.is_(None),
        )
    )
    return list(result.scalars().all())


async def get_organization(db: AsyncSession, organization_id: UUID) -> Organization | None:
    return await db.get(Organization, organization_id)


async def get_organization_by_slug(db: AsyncSession, slug: str) -> Organization | None:
    result = await db.execute(
        select(Organization).where(
            Organization.slug == slug,
            Organization.deleted_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def insert_organization(
    db: AsyncSession,
    *,
    slug: str,
    name: str,
    status: str,
) -> Organization:
    new_id = uuid4()
    await db.execute(
        insert(Organization).values(
            id=new_id,
            slug=slug,
            name=name,
            status=status,
        )
    )
    await db.flush()
    fetched = await db.execute(select(Organization).where(Organization.id == new_id))
    return fetched.scalar_one()


async def update_organization(
    db: AsyncSession,
    organization_id: UUID,
    *,
    fields: dict[str, Any],
) -> Organization | None:
    """Apply ``fields`` to the organization row.

    ``fields`` is the post-validation mapping from
    :class:`OrganizationPatch.model_dump(exclude_unset=True)`. Returns
    the refreshed row, or ``None`` if no row matched.
    """
    if not fields:
        return await get_organization(db, organization_id)
    await db.execute(
        update(Organization)
        .where(
            Organization.id == organization_id,
            Organization.deleted_at.is_(None),
        )
        .values(**fields)
    )
    await db.flush()
    return await get_organization(db, organization_id)


async def soft_delete_organization(
    db: AsyncSession,
    organization_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> bool:
    row = await db.get(Organization, organization_id)
    if row is None or row.deleted_at is not None:
        return False
    await soft_delete_cascade(db, row, actor_id=actor_id)
    return True


# ---------------------------------------------------------------------------
# Organization domains
# ---------------------------------------------------------------------------


async def list_domains_for_organization(
    db: AsyncSession, organization_id: UUID
) -> list[OrganizationDomain]:
    result = await db.execute(
        select(OrganizationDomain)
        .where(
            OrganizationDomain.organization_id == organization_id,
            OrganizationDomain.deleted_at.is_(None),
        )
        .order_by(OrganizationDomain.domain)
    )
    return list(result.scalars().all())


async def get_domain(db: AsyncSession, domain_id: UUID) -> OrganizationDomain | None:
    return await db.get(OrganizationDomain, domain_id)


async def get_domain_by_name(db: AsyncSession, domain: str) -> OrganizationDomain | None:
    result = await db.execute(
        select(OrganizationDomain).where(
            OrganizationDomain.domain == domain,
            OrganizationDomain.deleted_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def insert_domain(
    db: AsyncSession,
    *,
    organization_id: UUID,
    domain: str,
    auto_provision: bool,
) -> OrganizationDomain:
    new_id = uuid4()
    await db.execute(
        insert(OrganizationDomain).values(
            id=new_id,
            organization_id=organization_id,
            domain=domain,
            auto_provision=auto_provision,
        )
    )
    await db.flush()
    fetched = await db.execute(select(OrganizationDomain).where(OrganizationDomain.id == new_id))
    return fetched.scalar_one()


async def update_domain(
    db: AsyncSession,
    domain_id: UUID,
    *,
    fields: dict[str, Any],
) -> OrganizationDomain | None:
    if not fields:
        return await get_domain(db, domain_id)
    await db.execute(
        update(OrganizationDomain)
        .where(
            OrganizationDomain.id == domain_id,
            OrganizationDomain.deleted_at.is_(None),
        )
        .values(**fields)
    )
    await db.flush()
    return await get_domain(db, domain_id)


async def soft_delete_domain(
    db: AsyncSession,
    domain_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> bool:
    row = await db.get(OrganizationDomain, domain_id)
    if row is None or row.deleted_at is not None:
        return False
    await soft_delete_cascade(db, row, actor_id=actor_id)
    return True


# ---------------------------------------------------------------------------
# Org units
# ---------------------------------------------------------------------------


async def list_units_for_organization(
    db: AsyncSession,
    organization_id: UUID,
    *,
    parent_unit_id: UUID | None = None,
    only_roots: bool = False,
) -> list[OrgUnit]:
    """List org units in an organization.

    If ``only_roots`` is True, returns just the roots (parent_unit_id IS NULL).
    Otherwise filters by ``parent_unit_id`` exactly when provided.
    """
    stmt = select(OrgUnit).where(
        OrgUnit.organization_id == organization_id,
        OrgUnit.deleted_at.is_(None),
    )
    if only_roots:
        stmt = stmt.where(OrgUnit.parent_unit_id.is_(None))
    elif parent_unit_id is not None:
        stmt = stmt.where(OrgUnit.parent_unit_id == parent_unit_id)
    stmt = stmt.order_by(OrgUnit.name)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def list_descendant_unit_ids(
    db: AsyncSession, unit_id: UUID, *, include_self: bool = True
) -> list[UUID]:
    """Every live org unit at or below ``unit_id``, via a recursive CTE.

    The mirror image of ``queries/sql/org_unit_tree.sql``, which walks UP
    from a course to its ancestor units for the permission check. Scope
    FILTERING needs the opposite direction: "a manager standing on the
    Faculty of Engineering wants everything in it, including every
    department underneath".

    ``UNION`` (not ``UNION ALL``) is what makes this terminate if the
    parent chain ever contains a cycle — the same defence the ancestor
    walk relies on. :func:`services.organizations.patch_unit` now refuses
    to create one, but this query predates any guarantee that every row
    already in the table is acyclic, and an infinite recursion here would
    hang a request rather than return a wrong answer.

    Soft-deleted units are excluded at every level, so deleting a mid-tree
    unit detaches its subtree from scope queries rather than leaving the
    descendants silently reachable.
    """
    roots = (
        select(OrgUnit.id)
        .where(OrgUnit.id == unit_id, OrgUnit.deleted_at.is_(None))
        .cte("unit_subtree", recursive=True)
    )
    descendants = (
        select(OrgUnit.id)
        .join(roots, OrgUnit.parent_unit_id == roots.c.id)
        .where(OrgUnit.deleted_at.is_(None))
    )
    subtree = roots.union(descendants)

    rows = (await db.execute(select(subtree.c.id))).scalars().all()
    ids = [UUID(str(r)) for r in rows]
    if not include_self:
        ids = [i for i in ids if i != unit_id]
    return ids


async def get_unit(db: AsyncSession, unit_id: UUID) -> OrgUnit | None:
    return await db.get(OrgUnit, unit_id)


async def insert_unit(
    db: AsyncSession,
    *,
    organization_id: UUID,
    parent_unit_id: UUID | None,
    unit_type: str,
    name: str,
    code: str | None,
) -> OrgUnit:
    new_id = uuid4()
    await db.execute(
        insert(OrgUnit).values(
            id=new_id,
            organization_id=organization_id,
            parent_unit_id=parent_unit_id,
            unit_type=unit_type,
            name=name,
            code=code,
        )
    )
    await db.flush()
    fetched = await db.execute(select(OrgUnit).where(OrgUnit.id == new_id))
    return fetched.scalar_one()


async def update_unit(
    db: AsyncSession,
    unit_id: UUID,
    *,
    fields: dict[str, Any],
) -> OrgUnit | None:
    if not fields:
        return await get_unit(db, unit_id)
    await db.execute(
        update(OrgUnit).where(OrgUnit.id == unit_id, OrgUnit.deleted_at.is_(None)).values(**fields)
    )
    await db.flush()
    return await get_unit(db, unit_id)


async def soft_delete_unit(
    db: AsyncSession,
    unit_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> bool:
    row = await db.get(OrgUnit, unit_id)
    if row is None or row.deleted_at is not None:
        return False
    await soft_delete_cascade(db, row, actor_id=actor_id)
    return True


# ---------------------------------------------------------------------------
# Faculty staff affiliations
# ---------------------------------------------------------------------------


async def list_faculty_assignments(
    db: AsyncSession,
    organization_id: UUID,
    *,
    faculty_id: UUID | None = None,
) -> list[UserFacultyAssignment]:
    stmt = select(UserFacultyAssignment).where(
        UserFacultyAssignment.organization_id == organization_id,
        UserFacultyAssignment.status == "active",
        UserFacultyAssignment.deleted_at.is_(None),
        (UserFacultyAssignment.active_until.is_(None))
        | (UserFacultyAssignment.active_until > func.now()),
    )
    if faculty_id is not None:
        stmt = stmt.where(UserFacultyAssignment.faculty_id == faculty_id)
    return list(
        (await db.execute(stmt.order_by(UserFacultyAssignment.active_from.desc()))).scalars().all()
    )


async def faculty_role_codes(
    db: AsyncSession,
    assignment_keys: Sequence[tuple[UUID, UUID]],
) -> dict[tuple[UUID, UUID], list[str]]:
    """Active role codes scoped exactly to each ``(user, Faculty)`` pair."""
    if not assignment_keys:
        return {}
    stmt = (
        select(
            UserRoleAssignment.user_id,
            UserRoleAssignment.org_unit_id,
            Role.code,
        )
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            tuple_(UserRoleAssignment.user_id, UserRoleAssignment.org_unit_id).in_(
                list(assignment_keys)
            ),
            UserRoleAssignment.scope_kind == "org_unit",
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= func.now(),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
            Role.deleted_at.is_(None),
        )
        .distinct()
    )
    result: dict[tuple[UUID, UUID], set[str]] = {}
    for user_id, faculty_id, code in (await db.execute(stmt)).all():
        if faculty_id is not None:
            result.setdefault((user_id, faculty_id), set()).add(code)
    return {key: sorted(codes) for key, codes in result.items()}


async def list_active_faculty_ids_for_user(
    db: AsyncSession, user_id: UUID, organization_id: UUID
) -> list[UUID]:
    stmt = (
        select(UserFacultyAssignment.faculty_id)
        .where(
            UserFacultyAssignment.user_id == user_id,
            UserFacultyAssignment.organization_id == organization_id,
            UserFacultyAssignment.status == "active",
            UserFacultyAssignment.deleted_at.is_(None),
            (UserFacultyAssignment.active_until.is_(None))
            | (UserFacultyAssignment.active_until > func.now()),
        )
        .distinct()
    )
    return [UUID(str(value)) for value in (await db.execute(stmt)).scalars().all()]


async def get_active_faculty_assignment(
    db: AsyncSession, *, user_id: UUID, faculty_id: UUID
) -> UserFacultyAssignment | None:
    stmt = (
        select(UserFacultyAssignment)
        .where(
            UserFacultyAssignment.user_id == user_id,
            UserFacultyAssignment.faculty_id == faculty_id,
            UserFacultyAssignment.status == "active",
            UserFacultyAssignment.deleted_at.is_(None),
            (UserFacultyAssignment.active_until.is_(None))
            | (UserFacultyAssignment.active_until > func.now()),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def insert_faculty_assignment(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
    faculty_id: UUID,
    actor_id: UUID,
) -> UserFacultyAssignment:
    row = UserFacultyAssignment(
        user_id=user_id,
        organization_id=organization_id,
        faculty_id=faculty_id,
        status="active",
        created_by=actor_id,
        updated_by=actor_id,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


async def deactivate_faculty_assignment(
    db: AsyncSession, assignment: UserFacultyAssignment, *, actor_id: UUID
) -> None:
    assignment.status = "inactive"
    assignment.active_until = func.now()
    assignment.updated_by = actor_id
    await db.flush()


async def active_role_codes_for_user(
    db: AsyncSession, user_id: UUID, organization_id: UUID
) -> set[str]:
    stmt = (
        select(Role.code)
        .join(UserRoleAssignment, UserRoleAssignment.role_id == Role.id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.organization_id == organization_id,
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= func.now(),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
            Role.deleted_at.is_(None),
        )
        .distinct()
    )
    return set((await db.execute(stmt)).scalars().all())


async def user_has_role_scope(
    db: AsyncSession,
    *,
    user_id: UUID,
    role_code: str,
    organization_id: UUID,
    faculty_id: UUID | None = None,
) -> bool:
    stmt = (
        select(UserRoleAssignment.id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.organization_id == organization_id,
            Role.code == role_code,
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= func.now(),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
    )
    if faculty_id is None:
        stmt = stmt.where(UserRoleAssignment.scope_kind == "organization")
    else:
        stmt = stmt.where(
            UserRoleAssignment.scope_kind == "org_unit",
            UserRoleAssignment.org_unit_id == faculty_id,
            exists(
                select(UserFacultyAssignment.id).where(
                    UserFacultyAssignment.user_id == user_id,
                    UserFacultyAssignment.organization_id == organization_id,
                    UserFacultyAssignment.faculty_id == faculty_id,
                    UserFacultyAssignment.status == "active",
                    UserFacultyAssignment.deleted_at.is_(None),
                    (UserFacultyAssignment.active_until.is_(None))
                    | (UserFacultyAssignment.active_until > func.now()),
                )
            ),
        )
    return (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None


async def active_org_member_user_ids(
    db: AsyncSession, organization_id: UUID, user_ids: Sequence[UUID]
) -> set[UUID]:
    if not user_ids:
        return set()
    stmt = select(OrganizationMembership.user_id).where(
        OrganizationMembership.organization_id == organization_id,
        OrganizationMembership.user_id.in_(list(user_ids)),
        OrganizationMembership.status == "active",
        OrganizationMembership.deleted_at.is_(None),
    )
    return {UUID(str(value)) for value in (await db.execute(stmt)).scalars().all()}


async def faculty_has_live_dependencies(db: AsyncSession, faculty_id: UUID) -> bool:
    stmt = text(
        """
        SELECT EXISTS (
            SELECT 1 FROM learning_programs
            WHERE faculty_id = :faculty_id AND deleted_at IS NULL
            UNION ALL
            SELECT 1 FROM courses
            WHERE faculty_id = :faculty_id AND deleted_at IS NULL
            UNION ALL
            SELECT 1 FROM user_faculty_assignments
            WHERE faculty_id = :faculty_id AND deleted_at IS NULL
              AND status = 'active'
              AND (active_until IS NULL OR active_until > NOW())
        )
        """
    )
    return bool((await db.execute(stmt, {"faculty_id": faculty_id})).scalar_one())


# ---------------------------------------------------------------------------
# Memberships (extends the existing list / insert in queries.admin)
# ---------------------------------------------------------------------------


async def get_membership(db: AsyncSession, membership_id: UUID) -> OrganizationMembership | None:
    return await db.get(OrganizationMembership, membership_id)


async def list_memberships_by_ids(
    db: AsyncSession, membership_ids: Sequence[UUID]
) -> list[OrganizationMembership]:
    """Load the given memberships, skipping soft-deleted rows.

    The bulk assign uses this to verify EVERY id belongs to the caller's
    organization before it writes anything — ids arrive from a request body,
    and a membership id from another tenant must not be movable.
    """
    if not membership_ids:
        return []
    stmt = select(OrganizationMembership).where(
        OrganizationMembership.id.in_(list(membership_ids)),
        OrganizationMembership.deleted_at.is_(None),
    )
    return list((await db.execute(stmt)).scalars().all())


async def bulk_update_membership_unit(
    db: AsyncSession, membership_ids: Sequence[UUID], org_unit_id: UUID | None
) -> int:
    """Point many memberships at one org unit (or at NULL) in a single UPDATE.

    One statement rather than a loop of them: assigning a cohort is the
    normal case here, and N round-trips against the same table is what this
    endpoint exists to avoid. Returns the number of rows actually changed.

    Callers MUST have verified org ownership of every id first — this issues
    no tenancy check of its own.
    """
    if not membership_ids:
        return 0
    result = await db.execute(
        update(OrganizationMembership)
        .where(
            OrganizationMembership.id.in_(list(membership_ids)),
            OrganizationMembership.deleted_at.is_(None),
        )
        .values(org_unit_id=org_unit_id)
    )
    await db.flush()
    return int(result.rowcount or 0)


async def update_membership(
    db: AsyncSession,
    membership_id: UUID,
    *,
    fields: dict[str, Any],
) -> OrganizationMembership | None:
    """Apply ``fields`` to the membership row.

    Caller (service layer) is responsible for stamping ``left_at`` when
    transitioning to ``status='left'``.
    """
    if not fields:
        return await get_membership(db, membership_id)
    await db.execute(
        update(OrganizationMembership)
        .where(
            OrganizationMembership.id == membership_id,
            OrganizationMembership.deleted_at.is_(None),
        )
        .values(**fields)
    )
    await db.flush()
    return await get_membership(db, membership_id)


async def soft_delete_membership(
    db: AsyncSession,
    membership_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> bool:
    row = await db.get(OrganizationMembership, membership_id)
    if row is None or row.deleted_at is not None:
        return False
    await soft_delete_cascade(db, row, actor_id=actor_id)
    return True


def now_utc() -> datetime:
    """UTC now — exposed so service can stamp ``left_at`` on status flip."""
    return datetime.now(tz=UTC)


__all__ = [
    "active_org_member_user_ids",
    "active_role_codes_for_user",
    "count_organizations",
    "get_domain",
    "get_domain_by_name",
    "get_membership",
    "get_active_faculty_assignment",
    "faculty_has_live_dependencies",
    "faculty_role_codes",
    "get_organization",
    "get_organization_by_slug",
    "get_unit",
    "insert_domain",
    "insert_faculty_assignment",
    "insert_organization",
    "insert_unit",
    "list_domains_for_organization",
    "list_active_faculty_ids_for_user",
    "list_faculty_assignments",
    "list_organizations",
    "list_units_for_organization",
    "now_utc",
    "deactivate_faculty_assignment",
    "soft_delete_domain",
    "soft_delete_membership",
    "soft_delete_organization",
    "soft_delete_unit",
    "update_domain",
    "update_membership",
    "update_organization",
    "update_unit",
    "user_has_role_scope",
]


_INACTIVE_ORGS_SQL = text(
    resources.files("abridgeai.features.access_control.queries.sql")
    .joinpath("inactive_organizations.sql")
    .read_text(encoding="utf-8")
)


async def list_inactive_organizations(
    db: AsyncSession,
    *,
    now: datetime,
    days: int,
    organization_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """Organizations quiet for ``days`` (``queries/sql/inactive_organizations.sql``).

    The single definition of tenant inactivity. The operator dashboard counts
    these rows and the organizations list filters by them, so the count and the
    list it links to cannot disagree.
    """
    rows = (
        await db.execute(
            _INACTIVE_ORGS_SQL,
            {
                "now": now,
                "days": days,
                "organization_id": organization_id,
            },
        )
    ).mappings()
    return [dict(r) for r in rows]
