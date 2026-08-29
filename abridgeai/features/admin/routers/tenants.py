"""Tenant operations router -- ``/admin/organizations/{org_id}/operations``.

Sits under the organizations path because it is one more view of a tenant, not
a separate resource. Gated the same way per-org settings are (``org_unit.manage``
or ``system.administer``) and checked with ``require_org_access``, so a manager
sees their own tenant's operations and nobody else's -- the endpoint takes an
``org_id`` and is therefore covered by ``tests/lint/test_org_scoped_routes.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_org_access,
)
from abridgeai.features.admin.services import tenants as tenants_service

router = APIRouter(tags=["admin", "tenants"])

_TENANT_CODES = ("org_unit.manage", "system.administer")
_REQUIRE_TENANT = require_any_permission(*_TENANT_CODES)


class TenantOperationsOut(BaseModel):
    """One tenant's operational picture (PRD ADM-042).

    Job figures here ARE organization-scoped, unlike the platform-wide job
    aggregates elsewhere in this console. For a single tenant the entity set
    can be walked directly; across every tenant it cannot, which is why the
    dashboard still reports its job metrics as global.
    """

    organization_id: UUID
    as_of: datetime
    window_days: int

    active_members: int
    members_active_in_window: int

    course_count: int
    published_course_count: int
    material_count: int
    storage_bytes: int

    jobs_terminal_window: int
    jobs_failed_window: int
    jobs_in_flight: int
    #: ``None`` when no job reached a terminal state — never 0%.
    job_failure_rate_pct: float | None

    spend_window_usd: float
    #: How much of platform spend could be attributed to any tenant at all. A
    #: low value means this tenant's spend understates what it actually cost.
    spend_coverage_pct: float | None

    config_overrides: int

    is_inactive: bool
    last_activity_at: datetime | None
    days_quiet: int | None


@router.get(
    "/admin/organizations/{org_id}/operations",
    response_model=TenantOperationsOut,
)
async def get_tenant_operations(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_TENANT)],
    db: Annotated[AsyncSession, Depends(get_db)],
    window_days: Annotated[int, Query(ge=1, le=365)] = (
        tenants_service.DEFAULT_WINDOW_DAYS
    ),
) -> TenantOperationsOut:
    """People, inventory, storage, background work, spend and config for a tenant."""
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_TENANT_CODES,
    )
    result = await tenants_service.operations(
        db, organization_id=org_id, window_days=window_days
    )
    return TenantOperationsOut(**vars(result))


__all__ = ["router"]
