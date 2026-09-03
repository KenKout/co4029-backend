"""Policy authoring for platform administrators.

Gated on ``system.administer`` — the same permission the runtime Settings
console uses. Policies are platform-wide documents, not per-tenant content, so
there is no org scoping here and no new permission code to invent.

Error mapping follows the house convention: ``NotFoundError`` -> 404,
``ConflictError`` -> 409, any other ``AppError`` -> 422. A rule that raises
correctly but surfaces as a 500 is still a broken endpoint.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_permission
from abridgeai.features.policies import services as policy_service
from abridgeai.features.policies.schemas import (
    PolicyAudienceUpdate,
    PolicyCreate,
    PolicyDetail,
    PolicyVersionCreate,
    PolicyVersionPatch,
    PolicyVersionRead,
    PolicyVersionSummary,
)

router = APIRouter(prefix="/admin/policies", tags=["admin", "policies"])

_REQUIRE_ADMIN = require_permission("system.administer")


def _raise(exc: AppError) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": str(exc)},
        )
    if isinstance(exc, ConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "conflict", "message": str(exc)},
        )
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"error": "validation", "message": str(exc)},
    )


@router.get("", response_model=list[PolicyDetail])
async def list_policies_endpoint(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PolicyDetail]:
    return await policy_service.list_policies(db)


@router.post("", response_model=PolicyDetail, status_code=status.HTTP_201_CREATED)
async def create_policy_endpoint(
    payload: PolicyCreate,
    user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyDetail:
    try:
        detail = await policy_service.create_policy(db, payload, actor_id=user.user_id)
    except AppError as exc:
        raise _raise(exc) from exc
    await db.commit()
    return detail


@router.get("/{policy_id}", response_model=PolicyDetail)
async def get_policy_endpoint(
    policy_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyDetail:
    try:
        return await policy_service.policy_detail(db, policy_id)
    except AppError as exc:
        raise _raise(exc) from exc


@router.get("/{policy_id}/versions/{version_id}", response_model=PolicyVersionRead)
async def get_version_endpoint(
    policy_id: UUID,
    version_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyVersionRead:
    try:
        return await policy_service.read_version(db, policy_id, version_id)
    except AppError as exc:
        raise _raise(exc) from exc


@router.post(
    "/{policy_id}/versions",
    response_model=PolicyVersionSummary,
    status_code=status.HTTP_201_CREATED,
)
async def open_draft_endpoint(
    policy_id: UUID,
    payload: PolicyVersionCreate,
    user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyVersionSummary:
    try:
        version = await policy_service.open_new_draft(db, policy_id, payload, actor_id=user.user_id)
    except AppError as exc:
        raise _raise(exc) from exc
    await db.commit()
    return version


@router.patch("/{policy_id}/versions/{version_id}", response_model=PolicyVersionSummary)
async def update_draft_endpoint(
    policy_id: UUID,
    version_id: UUID,
    payload: PolicyVersionPatch,
    user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyVersionSummary:
    del policy_id  # addressed by version id; the path keeps the URL readable
    try:
        version = await policy_service.update_draft(db, version_id, payload, actor_id=user.user_id)
    except AppError as exc:
        raise _raise(exc) from exc
    await db.commit()
    return version


@router.post("/{policy_id}/versions/{version_id}/publish", response_model=PolicyVersionSummary)
async def publish_endpoint(
    policy_id: UUID,
    version_id: UUID,
    user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyVersionSummary:
    del policy_id
    try:
        version = await policy_service.publish_version(db, version_id, actor_id=user.user_id)
    except AppError as exc:
        raise _raise(exc) from exc
    await db.commit()
    return version


@router.put("/{policy_id}/audience", response_model=PolicyDetail)
async def set_audience_endpoint(
    policy_id: UUID,
    payload: PolicyAudienceUpdate,
    user: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PolicyDetail:
    """Replace the audience. An empty list makes the policy public."""
    try:
        detail = await policy_service.set_audience(db, policy_id, payload, actor_id=user.user_id)
    except AppError as exc:
        raise _raise(exc) from exc
    await db.commit()
    return detail


__all__ = ["router"]
