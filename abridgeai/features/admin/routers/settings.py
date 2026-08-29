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

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
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
    """One applied configuration change.

    ``reason`` is required, not optional (PRD ADM-033). An audit trail whose
    reason column is usually empty answers "what changed" but never "why", and
    "why" is the question asked during the incident the trail exists for.
    """

    value: Any = Field(description="Validated against the registry spec for this key.")
    reason: str = Field(
        min_length=settings_service.REASON_MIN_LENGTH,
        max_length=settings_service.REASON_MAX_LENGTH,
        description="Why this change is being made. Recorded in the audit trail.",
    )


class SettingClear(BaseModel):
    """Removing an override is a change like any other and needs a reason."""

    reason: str = Field(
        min_length=settings_service.REASON_MIN_LENGTH,
        max_length=settings_service.REASON_MAX_LENGTH,
    )


class SettingPreviewIn(BaseModel):
    """A pending edit to dry-run. ``value`` omitted previews a clear."""

    value: Any = None
    clear: bool = False


class ChangeImpactOut(BaseModel):
    """What applying a pending change would do. Nothing is written."""

    key: str
    label: str
    description: str
    scope: str
    organization_id: UUID | None
    current_value: bool | int | float | None
    current_source: str
    new_value: bool | int | float | None
    unchanged: bool
    affected_organizations: int
    total_organizations: int
    requires_reprocess: bool


class SettingChangeOut(BaseModel):
    """One recorded change.

    ``before_value`` / ``after_value`` are nullable in both directions and the
    null carries meaning: null before = the value was inherited, null after =
    the override was removed and inheritance resumed.
    """

    id: UUID
    setting_key: str
    organization_id: UUID | None
    organization_name: str | None = None
    scope: str
    action: str
    before_value: Any = None
    after_value: Any = None
    reason: str
    actor_id: UUID | None
    actor_email: str | None = None
    source: str
    reverted_change_id: UUID | None
    created_at: datetime


class ApplyResult(BaseModel):
    """The new state plus the audit row it was recorded as."""

    setting: SettingOut
    change_id: UUID


class RollbackIn(BaseModel):
    reason: str = Field(
        min_length=settings_service.REASON_MIN_LENGTH,
        max_length=settings_service.REASON_MAX_LENGTH,
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"error": "validation", "message": detail},
    )


def _out(row: settings_service.ResolvedSetting) -> SettingOut:
    return SettingOut(**vars(row))


def _impact_out(impact: settings_service.ChangeImpact) -> ChangeImpactOut:
    return ChangeImpactOut(**vars(impact))


def _change_out(row: dict[str, Any]) -> SettingChangeOut:
    return SettingChangeOut(
        id=row["id"],
        setting_key=row["setting_key"],
        organization_id=row["organization_id"],
        organization_name=row.get("organization_name"),
        scope=row["scope"],
        action=row["action"],
        before_value=row["before_value_json"],
        after_value=row["after_value_json"],
        reason=row["reason"],
        actor_id=row["actor_id"],
        actor_email=row.get("actor_email"),
        source=row["source"],
        reverted_change_id=row["reverted_change_id"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Deployment-wide defaults
# ---------------------------------------------------------------------------


@router.get("/admin/settings", response_model=list[SettingOut])
async def list_global_settings(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[SettingOut]:
    return [_out(r) for r in await settings_service.list_settings(db, None)]


@router.post("/admin/settings/{setting_key}/preview", response_model=ChangeImpactOut)
async def preview_global_setting(
    setting_key: str,
    payload: SettingPreviewIn,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChangeImpactOut:
    """Dry-run a deployment-wide change: validate it and report its reach.

    POST rather than GET because the pending value is a body, not an identity,
    and because a validation failure here is the point of the call. Nothing is
    written and the transaction is not committed.
    """
    try:
        impact = (
            await settings_service.preview_clear(
                db, key=setting_key, organization_id=None
            )
            if payload.clear
            else await settings_service.preview_change(
                db, key=setting_key, value=payload.value, organization_id=None
            )
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    return _impact_out(impact)


@router.put("/admin/settings/{setting_key}", response_model=ApplyResult)
async def apply_global_setting(
    setting_key: str,
    payload: SettingWrite,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyResult:
    """Apply a deployment-wide change and record it.

    This is the ONLY path that writes a global setting. The value and its audit
    row commit together, so there is no ordering in which a change reaches the
    deployment without a record of who made it and why (ADM-030/033).
    """
    try:
        row, change = await settings_service.apply_change(
            db,
            key=setting_key,
            value=payload.value,
            organization_id=None,
            actor_id=current_user.user_id,
            reason=payload.reason,
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return ApplyResult(setting=_out(row), change_id=change["id"])


@router.delete("/admin/settings/{setting_key}", response_model=ApplyResult)
async def clear_global_setting(
    setting_key: str,
    reason: Annotated[
        str,
        Query(
            min_length=settings_service.REASON_MIN_LENGTH,
            max_length=settings_service.REASON_MAX_LENGTH,
            description="Why the override is being removed. Recorded in the audit trail.",
        ),
    ],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyResult:
    """Remove the deployment default so environment/code defaults apply again.

    The reason rides in the query string because DELETE bodies are dropped by
    enough intermediaries that requiring one would make the audit trail
    unreliable in exactly the deployments that need it most.
    """
    try:
        row, change = await settings_service.apply_clear(
            db,
            key=setting_key,
            organization_id=None,
            actor_id=current_user.user_id,
            reason=reason,
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return ApplyResult(setting=_out(row), change_id=change["id"])


@router.get("/admin/settings/changes", response_model=list[SettingChangeOut])
async def list_global_setting_changes(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    setting_key: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SettingChangeOut]:
    """Deployment-wide change history, newest first."""
    rows = await settings_service.list_changes(
        db, key=setting_key, global_only=True, limit=limit
    )
    return [_change_out(r) for r in rows]


@router.post(
    "/admin/settings/changes/{change_id}/rollback", response_model=ApplyResult
)
async def rollback_global_setting_change(
    change_id: UUID,
    payload: RollbackIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_GLOBAL_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyResult:
    """Restore the value a previous global change replaced (ADM-031).

    Appends a new change rather than editing history: the rollback is itself
    an event worth recording, and the original entry is what an investigation
    later needs to see.
    """
    try:
        row, change = await settings_service.rollback_change(
            db,
            change_id=change_id,
            actor_id=current_user.user_id,
            reason=payload.reason,
        )
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": str(exc)},
        ) from exc
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return ApplyResult(setting=_out(row), change_id=change["id"])


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
    response_model=ApplyResult,
)
async def apply_org_setting(
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
        row, change = await settings_service.apply_change(
            db,
            key=setting_key,
            value=payload.value,
            organization_id=org_id,
            actor_id=current_user.user_id,
            reason=payload.reason,
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return ApplyResult(setting=_out(row), change_id=change["id"])


@router.delete(
    "/admin/organizations/{org_id}/settings/{setting_key}",
    response_model=ApplyResult,
)
async def clear_org_setting(
    org_id: UUID,
    setting_key: str,
    reason: Annotated[
        str,
        Query(
            min_length=settings_service.REASON_MIN_LENGTH,
            max_length=settings_service.REASON_MAX_LENGTH,
        ),
    ],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyResult:
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
        row, change = await settings_service.apply_clear(
            db,
            key=setting_key,
            organization_id=org_id,
            actor_id=current_user.user_id,
            reason=reason,
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return ApplyResult(setting=_out(row), change_id=change["id"])


@router.post(
    "/admin/organizations/{org_id}/settings/{setting_key}/preview",
    response_model=ChangeImpactOut,
)
async def preview_org_setting(
    org_id: UUID,
    setting_key: str,
    payload: SettingPreviewIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChangeImpactOut:
    """Dry-run one tenant's override. Writes nothing."""
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_SETTINGS_CODES,
    )
    try:
        impact = (
            await settings_service.preview_clear(
                db, key=setting_key, organization_id=org_id
            )
            if payload.clear
            else await settings_service.preview_change(
                db, key=setting_key, value=payload.value, organization_id=org_id
            )
        )
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    return _impact_out(impact)


@router.get(
    "/admin/organizations/{org_id}/settings/changes",
    response_model=list[SettingChangeOut],
)
async def list_org_setting_changes(
    org_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
    setting_key: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SettingChangeOut]:
    """One tenant's change history. Global changes are not included: they are
    not this organization's to review or roll back."""
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_SETTINGS_CODES,
    )
    rows = await settings_service.list_changes(
        db, key=setting_key, organization_id=org_id, limit=limit
    )
    return [_change_out(r) for r in rows]


@router.post(
    "/admin/organizations/{org_id}/settings/changes/{change_id}/rollback",
    response_model=ApplyResult,
)
async def rollback_org_setting_change(
    org_id: UUID,
    change_id: UUID,
    payload: RollbackIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_SETTINGS)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyResult:
    """Roll back one of this tenant's changes.

    ``organization_id`` is passed down so the service refuses a change id
    belonging to another tenant or to the global scope — a 404, not a 403, so
    an org-scoped admin cannot probe for the existence of changes elsewhere.
    """
    await require_org_access(
        db,
        current_user,
        org_id,
        resource="organization",
        resource_id=org_id,
        permissions=_ORG_SETTINGS_CODES,
    )
    try:
        row, change = await settings_service.rollback_change(
            db,
            change_id=change_id,
            actor_id=current_user.user_id,
            reason=payload.reason,
            organization_id=org_id,
        )
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": str(exc)},
        ) from exc
    except SettingValidationError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return ApplyResult(setting=_out(row), change_id=change["id"])


__all__ = ["router"]
