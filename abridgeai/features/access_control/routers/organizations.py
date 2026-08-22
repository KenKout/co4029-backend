"""Admin router for organization / org-unit / domain / membership.

Mirrors the existing
:mod:`abridgeai.features.access_control.routers.admin` posture: every
endpoint declares a ``Depends(require_*)`` so the
``test_every_endpoint_has_permission_dependency`` invariant continues
to hold. All write paths go through
:mod:`abridgeai.features.access_control.services.organizations`.

Permission map for this router: every endpoint is gated by
``_REQUIRE_ORG_MANAGE`` (``org_unit.manage`` / ``user.bulk_import`` /
``system.administer``). Reads on the listing/detail surfaces still
require a role with one of those permissions because the data carries
PII (``student_code`` / ``employee_code`` on memberships).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.pagination import PageResponse
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_org_access,
)
from abridgeai.features.access_control.schemas.admin import (
    MembershipPatch,
    MembershipRead,
    OrganizationCreate,
    OrganizationDomainCreate,
    OrganizationDomainPatch,
    OrganizationDomainRead,
    OrganizationListPage,
    OrganizationPatch,
    OrganizationRead,
    OrgUnitCreate,
    OrgUnitNode,
    OrgUnitPatch,
    OrgUnitRead,
)
from abridgeai.features.access_control.services import (
    organizations as org_service,
)

router = APIRouter(tags=["admin", "access_control", "organizations"])


# One tuple for both the dependency ("held anywhere?") and the per-resource
# org check ("held HERE?"), so the two cannot drift.
_ORG_MANAGE_CODES = ("org_unit.manage", "user.bulk_import", "system.administer")
_REQUIRE_ORG_MANAGE = require_any_permission(*_ORG_MANAGE_CODES)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"error": "validation", "message": detail},
    )


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )



async def _require_access_to(
    db: AsyncSession,
    current_user: CurrentUser,
    *,
    organization_id: UUID | None,
    resource: str,
    resource_id: UUID,
) -> None:
    """Guard a route addressed by a child resource's own id.

    ``organization_id is None`` means the resource is absent or soft-deleted.
    Raising the same 404 as a foreign-org caller keeps the two responses
    identical, so the endpoint cannot be used to probe which ids exist in
    another tenant.
    """
    if organization_id is None:
        raise _not_found(f"{resource} {resource_id} not found")
    await require_org_access(
        db,
        current_user,
        organization_id,
        resource=resource,
        resource_id=resource_id,
        permissions=_ORG_MANAGE_CODES,
    )


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------




async def _visible_org_ids(
    db: AsyncSession, current_user: CurrentUser
) -> list[UUID] | None:
    """Organizations this caller may see in a listing; ``None`` == unrestricted.

    ``_REQUIRE_ORG_MANAGE`` accepts ``org_unit.manage`` / ``user.bulk_import``,
    and the flat permission set behind it cannot tell a grant made inside one
    tenant from a global one. Unrestricted listing therefore leaked the whole
    tenant roster — names, slugs and statuses of every customer — to any
    manager. Only ``system.administer`` keeps the global view.
    """
    if current_user.has_permission("system.administer"):
        return None
    return await org_service.organization_ids_for_user(db, current_user.user_id)


@router.get("/admin/organizations", response_model=OrganizationListPage)
async def list_organizations_endpoint(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_deleted: bool = False,
    org_status: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> OrganizationListPage:
    if limit < 1 or limit > 500:
        raise _bad_request("limit must be between 1 and 500")
    try:
        page = await org_service.list_organizations(
            db,
            include_deleted=include_deleted,
            status=org_status,
            limit=limit,
            cursor=cursor,
            visible_to_ids=await _visible_org_ids(db, current_user),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_cursor", "message": str(exc)},
        ) from exc
    return OrganizationListPage(
        items=[OrganizationRead.model_validate(r) for r in page.items],
        next_cursor=page.next_cursor,
    )


@router.get(
    "/admin/organizations/search",
    response_model=PageResponse[OrganizationRead],
)
async def search_organizations_endpoint(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=200)] = None,
    org_status: Annotated[str | None, Query(alias="status")] = None,
    sort: Annotated[str | None, Query()] = None,
    sort_dir: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> PageResponse[OrganizationRead]:
    """Page-numbered admin org list with server-side search (name/slug) +
    whitelisted sort (``name`` / ``status`` / ``created_at``). Additive to
    the cursor endpoint above — this one backs the DataTable."""
    result = await org_service.search_organizations(
        db,
        status=org_status,
        search=search,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
        visible_to_ids=await _visible_org_ids(db, current_user),
    )
    return PageResponse[OrganizationRead](
        items=[OrganizationRead.model_validate(o) for o in result.items],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


@router.post(
    "/admin/organizations",
    response_model=OrganizationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_organization_endpoint(
    payload: OrganizationCreate,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationRead:
    try:
        org = await org_service.create_organization(db, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return OrganizationRead.model_validate(org)


@router.get(
    "/admin/organizations/{org_id}",
    response_model=OrganizationRead,
)
async def get_organization_endpoint(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationRead:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        org = await org_service.get_organization(db, org_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return OrganizationRead.model_validate(org)


@router.patch(
    "/admin/organizations/{org_id}",
    response_model=OrganizationRead,
)
async def patch_organization_endpoint(
    org_id: UUID,
    payload: OrganizationPatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationRead:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        org = await org_service.patch_organization(db, org_id, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return OrganizationRead.model_validate(org)


@router.delete(
    "/admin/organizations/{org_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_organization_endpoint(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        await org_service.delete_organization(
            db, org_id, actor_id=current_user.user_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


# ---------------------------------------------------------------------------
# Organization domains
# ---------------------------------------------------------------------------


@router.get(
    "/admin/organizations/{org_id}/domains",
    response_model=list[OrganizationDomainRead],
)
async def list_domains_endpoint(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[OrganizationDomainRead]:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        rows = await org_service.list_domains(db, org_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return [OrganizationDomainRead.model_validate(r) for r in rows]


@router.post(
    "/admin/organizations/{org_id}/domains",
    response_model=OrganizationDomainRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_domain_endpoint(
    org_id: UUID,
    payload: OrganizationDomainCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationDomainRead:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        dom = await org_service.create_domain(db, org_id, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return OrganizationDomainRead.model_validate(dom)


@router.patch(
    "/admin/organization-domains/{domain_id}",
    response_model=OrganizationDomainRead,
)
async def patch_domain_endpoint(
    domain_id: UUID,
    payload: OrganizationDomainPatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationDomainRead:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_domain(db, domain_id),
        resource="organization_domain",
        resource_id=domain_id,
    )
    try:
        dom = await org_service.patch_domain(db, domain_id, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return OrganizationDomainRead.model_validate(dom)


@router.delete(
    "/admin/organization-domains/{domain_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_domain_endpoint(
    domain_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_domain(db, domain_id),
        resource="organization_domain",
        resource_id=domain_id,
    )
    try:
        await org_service.delete_domain(
            db, domain_id, actor_id=current_user.user_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


# ---------------------------------------------------------------------------
# Org units
# ---------------------------------------------------------------------------


@router.get(
    "/admin/organizations/{org_id}/units",
    response_model=list[OrgUnitRead],
)
async def list_units_endpoint(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    parent_unit_id: UUID | None = None,
    only_roots: bool = False,
) -> list[OrgUnitRead]:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        rows = await org_service.list_units(
            db, org_id, parent_unit_id=parent_unit_id, only_roots=only_roots
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return [OrgUnitRead.model_validate(r) for r in rows]


@router.get(
    "/admin/organizations/{org_id}/units/tree",
    response_model=list[OrgUnitNode],
)
async def list_unit_tree_endpoint(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[OrgUnitNode]:
    """The organization's units as a nested tree (roots first).

    Registered BEFORE ``/units`` is irrelevant here (different suffix), but
    it must stay above ``/admin/org-units/{unit_id}`` in file order for the
    same reason the courses router orders its literal paths first.

    Same permission gate as the flat list — ``org_unit.manage`` — so the
    manager surface reaches it without ``system.administer``.
    """
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        return await org_service.list_unit_tree(db, org_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.post(
    "/admin/organizations/{org_id}/units",
    response_model=OrgUnitRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_unit_endpoint(
    org_id: UUID,
    payload: OrgUnitCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrgUnitRead:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_MANAGE_CODES,
    )
    try:
        unit = await org_service.create_unit(db, org_id, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return OrgUnitRead.model_validate(unit)


@router.get(
    "/admin/org-units/{unit_id}",
    response_model=OrgUnitRead,
)
async def get_unit_endpoint(
    unit_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrgUnitRead:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_unit(db, unit_id),
        resource="org_unit",
        resource_id=unit_id,
    )
    try:
        unit = await org_service.get_unit(db, unit_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return OrgUnitRead.model_validate(unit)


@router.patch(
    "/admin/org-units/{unit_id}",
    response_model=OrgUnitRead,
)
async def patch_unit_endpoint(
    unit_id: UUID,
    payload: OrgUnitPatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrgUnitRead:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_unit(db, unit_id),
        resource="org_unit",
        resource_id=unit_id,
    )
    try:
        unit = await org_service.patch_unit(db, unit_id, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return OrgUnitRead.model_validate(unit)


@router.delete(
    "/admin/org-units/{unit_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_unit_endpoint(
    unit_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_unit(db, unit_id),
        resource="org_unit",
        resource_id=unit_id,
    )
    try:
        await org_service.delete_unit(
            db, unit_id, actor_id=current_user.user_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


# ---------------------------------------------------------------------------
# Memberships (extends list / add already exposed by routers.admin)
# ---------------------------------------------------------------------------


@router.patch(
    "/admin/organization-memberships/{membership_id}",
    response_model=MembershipRead,
)
async def patch_membership_endpoint(
    membership_id: UUID,
    payload: MembershipPatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MembershipRead:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_membership(db, membership_id),
        resource="organization_membership",
        resource_id=membership_id,
    )
    try:
        m = await org_service.patch_membership(db, membership_id, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return MembershipRead.model_validate(m)


@router.delete(
    "/admin/organization-memberships/{membership_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_membership_endpoint(
    membership_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _require_access_to(
        db,
        current_user,
        organization_id=await org_service.organization_id_for_membership(db, membership_id),
        resource="organization_membership",
        resource_id=membership_id,
    )
    try:
        await org_service.delete_membership(
            db, membership_id, actor_id=current_user.user_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


__all__ = ["router"]
