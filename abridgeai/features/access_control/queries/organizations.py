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

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import insert, select, tuple_, update

from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.pagination import Page, paginate
from abridgeai.features.access_control.models import (
    Organization,
    OrganizationDomain,
    OrganizationMembership,
    OrgUnit,
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
        stmt = stmt.where(
            tuple_(Organization.name, Organization.id) > (after_name, after_id)
        )
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
    fetched = await db.execute(
        select(OrganizationDomain).where(OrganizationDomain.id == new_id)
    )
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
        update(OrgUnit)
        .where(OrgUnit.id == unit_id, OrgUnit.deleted_at.is_(None))
        .values(**fields)
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
# Memberships (extends the existing list / insert in queries.admin)
# ---------------------------------------------------------------------------


async def get_membership(db: AsyncSession, membership_id: UUID) -> OrganizationMembership | None:
    return await db.get(OrganizationMembership, membership_id)


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
    "count_organizations",
    "get_domain",
    "get_domain_by_name",
    "get_membership",
    "get_organization",
    "get_organization_by_slug",
    "get_unit",
    "insert_domain",
    "insert_organization",
    "insert_unit",
    "list_domains_for_organization",
    "list_organizations",
    "list_units_for_organization",
    "now_utc",
    "soft_delete_domain",
    "soft_delete_membership",
    "soft_delete_organization",
    "soft_delete_unit",
    "update_domain",
    "update_membership",
    "update_organization",
    "update_unit",
]
