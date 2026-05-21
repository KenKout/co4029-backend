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

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.access_control.schemas.admin import (
    MembershipPatch,
    MembershipRead,
    OrganizationCreate,
    OrganizationDomainCreate,
    OrganizationDomainPatch,
    OrganizationDomainRead,
    OrganizationPatch,
    OrganizationRead,
    OrgUnitCreate,
    OrgUnitPatch,
    OrgUnitRead,
)
from abridgeai.features.access_control.services import (
    organizations as org_service,
)

router = APIRouter(tags=["admin", "access_control", "organizations"])


_REQUIRE_ORG_MANAGE = require_any_permission(
    "org_unit.manage", "user.bulk_import", "system.administer"
)


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


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


@router.get("/admin/organizations", response_model=list[OrganizationRead])
async def list_organizations_endpoint(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_deleted: bool = False,
    org_status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[OrganizationRead]:
    if limit < 1 or limit > 500:
        raise _bad_request("limit must be between 1 and 500")
    if offset < 0:
        raise _bad_request("offset must be >= 0")
    rows = await org_service.list_organizations(
        db,
        include_deleted=include_deleted,
        status=org_status,
        limit=limit,
        offset=offset,
    )
    return [OrganizationRead.model_validate(r) for r in rows]


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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[OrganizationDomainRead]:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationDomainRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationDomainRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    parent_unit_id: UUID | None = None,
    only_roots: bool = False,
) -> list[OrgUnitRead]:
    try:
        rows = await org_service.list_units(
            db, org_id, parent_unit_id=parent_unit_id, only_roots=only_roots
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return [OrgUnitRead.model_validate(r) for r in rows]


@router.post(
    "/admin/organizations/{org_id}/units",
    response_model=OrgUnitRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_unit_endpoint(
    org_id: UUID,
    payload: OrgUnitCreate,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrgUnitRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrgUnitRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrgUnitRead:
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
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MembershipRead:
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
    try:
        await org_service.delete_membership(
            db, membership_id, actor_id=current_user.user_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


__all__ = ["router"]
