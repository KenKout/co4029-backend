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

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import AppError, NotFoundError
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
)
from abridgeai.features.access_control.queries import organizations as org_queries
from abridgeai.features.access_control.schemas.admin import (
    MembershipPatch,
    OrganizationCreate,
    OrganizationDomainCreate,
    OrganizationDomainPatch,
    OrganizationPatch,
    OrgUnitCreate,
    OrgUnitPatch,
)

if TYPE_CHECKING:
    from abridgeai.core.db import AsyncSession  # type: ignore[attr-defined]


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
    )
    next_cursor = (
        encode_composite_cursor(rows[-1].name, rows[-1].id)
        if len(rows) == limit
        else None
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
) -> Page[Organization]:
    """Offset page of organisations (server-side search + sort). Thin
    delegate to the query layer, which owns the SQLAlchemy statement."""
    return await org_queries.search_organizations(
        db,
        include_deleted=include_deleted,
        status=status,
        search=search,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )


async def count_organizations(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
) -> int:
    return await org_queries.count_organizations(
        db, include_deleted=include_deleted, status=status
    )


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
    deleted = await org_queries.soft_delete_organization(
        db, organization_id, actor_id=actor_id
    )
    if not deleted:
        raise NotFoundError("Organization not found")


# ---------------------------------------------------------------------------
# Organization domains
# ---------------------------------------------------------------------------


async def list_domains(
    db: AsyncSession, organization_id: UUID
) -> list[OrganizationDomain]:
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


async def get_unit(db: AsyncSession, unit_id: UUID) -> OrgUnit:
    row = await org_queries.get_unit(db, unit_id)
    if row is None or row.deleted_at is not None:
        raise NotFoundError("Org unit not found")
    return row


async def create_unit(
    db: AsyncSession,
    organization_id: UUID,
    payload: OrgUnitCreate,
) -> OrgUnit:
    await get_organization(db, organization_id)
    if payload.parent_unit_id is not None:
        parent = await org_queries.get_unit(db, payload.parent_unit_id)
        if parent is None or parent.deleted_at is not None:
            raise AppError("parent_unit_id does not exist or is deleted")
        if parent.organization_id != organization_id:
            raise AppError("parent_unit_id belongs to a different organization")
    return await org_queries.insert_unit(
        db,
        organization_id=organization_id,
        parent_unit_id=payload.parent_unit_id,
        unit_type=payload.unit_type,
        name=payload.name,
        code=payload.code,
    )


async def patch_unit(
    db: AsyncSession,
    unit_id: UUID,
    payload: OrgUnitPatch,
) -> OrgUnit:
    current = await get_unit(db, unit_id)
    fields = payload.model_dump(exclude_unset=True)
    if "parent_unit_id" in fields and fields["parent_unit_id"] is not None:
        if fields["parent_unit_id"] == unit_id:
            raise AppError("parent_unit_id cannot reference the unit itself")
        parent = await org_queries.get_unit(db, fields["parent_unit_id"])
        if parent is None or parent.deleted_at is not None:
            raise AppError("parent_unit_id does not exist or is deleted")
        if parent.organization_id != current.organization_id:
            raise AppError("parent_unit_id belongs to a different organization")
    updated = await org_queries.update_unit(db, unit_id, fields=fields)
    if updated is None:
        raise NotFoundError("Org unit not found")
    return updated


async def delete_unit(
    db: AsyncSession,
    unit_id: UUID,
    *,
    actor_id: UUID | None,
) -> None:
    deleted = await org_queries.soft_delete_unit(db, unit_id, actor_id=actor_id)
    if not deleted:
        raise NotFoundError("Org unit not found")


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
    if (
        fields.get("status") == "left"
        and current.status != "left"
        and current.left_at is None
    ):
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
    deleted = await org_queries.soft_delete_membership(
        db, membership_id, actor_id=actor_id
    )
    if not deleted:
        raise NotFoundError("Membership not found")


__all__ = [
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
    "list_organizations",
    "list_units",
    "patch_domain",
    "patch_membership",
    "patch_organization",
    "patch_unit",
]
