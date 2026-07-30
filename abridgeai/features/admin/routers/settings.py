"""Runtime settings admin router — ``/admin/settings`` and the per-org variant.

Two scopes, deliberately two routes rather than one with an optional query
parameter, because they need different authorization:

* ``/admin/settings`` — the deployment default. ``system.administer`` only.
* ``/admin/organizations/{org_id}/settings`` — one tenant's overrides. The
  caller must belong to that organization (or hold ``system.administer``).

The second is the reason this router exists in its current shape.
``system_settings`` is now an org-owned, course-less table — the exact family
that shipped without tenancy checks in career paths, invitation codes and the
organizations router. ``require_org_access`` is therefore wired in from the
first commit, and ``tests/lint/test_org_scoped_routes.py`` enforces it because
the handlers take an ``org_id``.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.core.settings_registry import SettingValidationError
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_org_access,
    require_permission,
)
from abridgeai.features.admin.services import settings as settings_service

router = APIRouter(tags=["admin", "settings"])

_REQUIRE_GLOBAL_WRITE = require_permission("system.administer")
# Shared by the dependency and the org check; see require_org_access.
_ORG_SETTINGS_CODES = ("org_unit.manage", "system.administer")
_REQUIRE_ORG_SETTINGS = require_any_permission(*_ORG_SETTINGS_CODES)


class SettingOut(BaseModel):
    key: str
    group: str
    type: str
    label: str
    description: str
    env_var: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    requires_reprocess: bool
    default_value: bool | int | float
    env_value: bool | int | float | None = None
    global_value: bool | int | float | None = None
    org_value: bool | int | float | None = None
    effective_value: bool | int | float
    source: str


class SettingWrite(BaseModel):
    value: Any = Field(description="Validated against the registry spec for this key.")


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"error": "validation", "message": detail},
    )


def _out(row: settings_service.ResolvedSetting) -> SettingOut:
    return SettingOut(**vars(row))


# ---------------------------------------------------------------------------
# Deployment-wide defaults
# ---------------------------------------------------------------------------


@router.get("/admin/settings", response_model=list[SettingOut])
async def list_global_settings(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[SettingOut]:
    return [_out(r) for r in await settings_service.list_settings(db, None)]


@router.put("/admin/settings/{setting_key}", response_model=SettingOut)
async def set_global_setting(
    setting_key: str,
    payload: SettingWrite,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SettingOut:
    try:
        row = await settings_service.set_setting(
            db,
            key=setting_key,
            value=payload.value,
            organization_id=None,
            actor_id=current_user.user_id,
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return _out(row)


@router.delete("/admin/settings/{setting_key}", response_model=SettingOut)
async def clear_global_setting(
    setting_key: str,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SettingOut:
    try:
        row = await settings_service.clear_setting(
            db, key=setting_key, organization_id=None
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return _out(row)


# ---------------------------------------------------------------------------
# Per-organization overrides
# ---------------------------------------------------------------------------


@router.get(
    "/admin/organizations/{org_id}/settings",
    response_model=list[SettingOut],
)
async def list_org_settings(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[SettingOut]:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_SETTINGS_CODES,
    )
    return [_out(r) for r in await settings_service.list_settings(db, org_id)]


@router.put(
    "/admin/organizations/{org_id}/settings/{setting_key}",
    response_model=SettingOut,
)
async def set_org_setting(
    org_id: UUID,
    setting_key: str,
    payload: SettingWrite,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SettingOut:
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_SETTINGS_CODES,
    )
    try:
        row = await settings_service.set_setting(
            db,
            key=setting_key,
            value=payload.value,
            organization_id=org_id,
            actor_id=current_user.user_id,
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return _out(row)


@router.delete(
    "/admin/organizations/{org_id}/settings/{setting_key}",
    response_model=SettingOut,
)
async def clear_org_setting(
    org_id: UUID,
    setting_key: str,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SettingOut:
    """Drop this organization's override so the global default applies again."""
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_SETTINGS_CODES,
    )
    try:
        row = await settings_service.clear_setting(
            db, key=setting_key, organization_id=org_id
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return _out(row)


__all__ = ["router"]
