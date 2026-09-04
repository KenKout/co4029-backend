"""Identity ``/users/me`` router — self-service profile + permissions.

Self-scoped endpoints under ``/users/me`` that the SPA uses for the signed-in
user:

* ``GET /users/me`` — current user details (with profile if present).
* ``PATCH /users/me/profile`` — partial update of the user's profile fields.
* ``PUT /users/me/avatar`` — replace the caller's avatar image.
* ``GET|POST /users/me/links`` and ``PATCH|DELETE /users/me/links/{link_id}``
  — the caller's external profile links (FR-2.8).
* ``GET /users/me/permissions`` — effective global permission codes.

Auth: every endpoint requires a valid bearer token via
:func:`get_current_user`. No additional permission check is needed because
each endpoint is self-scoped — users can always read and update their own
profile. Admin lookup of other users lives in ``users.py`` and uses
``require_permission("user.read")`` instead.

Service-commit discipline: ``services.profile.update_profile`` commits its
own transaction. ``services.permissions.get_effective_permissions`` is a
read-only wrapper over the canonical scope-aware query (T1.4a).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import (
    CurrentUser,
    get_current_user,
    get_current_user_pre_mfa,
)
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.services import (
    get_effective_permissions,
)
from abridgeai.features.identity.models import User
from abridgeai.features.identity.schemas import (
    UserPermissionsRead,
    UserProfileLinkIn,
    UserProfileLinkRead,
    UserProfileLinkUpdate,
    UserProfileUpdate,
    UserRead,
)
from abridgeai.features.identity.services import (
    get_current_user_read,
)
from abridgeai.features.identity.services import profile as profile_service

router = APIRouter(prefix="/users/me", tags=["users", "me"])
me_root_router = APIRouter(prefix="/me", tags=["users", "me"])


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _link_not_found(link_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": "profile_link", "id": str(link_id)},
    )


async def _load_user(db: AsyncSession, current_user: CurrentUser) -> User:
    user = await db.get(User, current_user.user_id)
    if user is None:
        raise _unauthorized("User not found")
    return user


@router.get("", response_model=UserRead)
async def read_me(
    current_user: Annotated[CurrentUser, Depends(get_current_user_pre_mfa)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserRead:
    """Return the signed-in user's identity.

    Uses ``get_current_user_pre_mfa`` so the SPA can hydrate basic
    identity (name, avatar) on the ``/login/mfa`` page before the user
    completes the second factor. All other endpoints stay behind the
    full ``get_current_user`` MFA gate.
    """
    user = await _load_user(db, current_user)
    read = await get_current_user_read(db, user)
    # Populate the primary organization. `UserRead` has carried these two
    # fields for the admin user-list all along, but only the search service
    # filled them, so /users/me answered with organization_id=None. Anything
    # org-scoped in the SPA (the org-unit tree, and the scope filters built on
    # it) then had no way to learn which organization the caller belongs to
    # without an admin-only lookup. Cheap: one membership read per /me.
    org = await access_control_api.get_user_primary_org(db, current_user.user_id)
    if org is not None:
        read.organization_id = org.id
        read.organization_name = org.name
    return read


@router.patch("/profile", response_model=UserRead)
async def update_my_profile(
    payload: UserProfileUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserRead:
    user = await _load_user(db, current_user)
    await profile_service.update_profile(db, user, payload)
    return await get_current_user_read(db, user)


@router.put("/avatar", response_model=UserRead)
async def upload_my_avatar(
    request: Request,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserRead:
    """Upload a new avatar image (JPEG/PNG/WebP/GIF, ≤ 2 MiB).

    The raw image bytes are sent as the request body with the image's MIME type
    in the ``Content-Type`` header (no multipart wrapper — keeps the backend
    dependency-free and matches the small-blob upload pattern). Stores the image
    in object storage and points the caller's profile at it. Self-scoped: a user
    can only change their own avatar.
    """
    user = await _load_user(db, current_user)
    data = await request.body()
    content_type = request.headers.get("content-type", "application/octet-stream")
    # Strip any parameters (e.g. "image/png; charset=binary").
    content_type = content_type.split(";", 1)[0].strip().lower()
    try:
        return await profile_service.upload_avatar(
            db,
            user,
            data=data,
            content_type=content_type,
        )
    except profile_service.AvatarUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("/links", response_model=list[UserProfileLinkRead])
async def list_my_links(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[UserProfileLinkRead]:
    """List the caller's external profile links (FR-2.8).

    The same links also ride along on ``GET /users/me`` under
    ``profile.links``; this endpoint exists so the profile editor can refetch
    just the list after a write without re-reading the whole user.
    """
    user = await _load_user(db, current_user)
    return await profile_service.list_links(db, user)


@router.post("/links", response_model=UserProfileLinkRead, status_code=status.HTTP_201_CREATED)
async def create_my_link(
    payload: UserProfileLinkIn,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserProfileLinkRead:
    """Add an external link (website / GitHub / LinkedIn / portfolio / other)."""
    user = await _load_user(db, current_user)
    try:
        return await profile_service.create_link(db, user, payload)
    except profile_service.ProfileLinkError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.patch("/links/{link_id}", response_model=UserProfileLinkRead)
async def update_my_link(
    link_id: UUID,
    payload: UserProfileLinkUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserProfileLinkRead:
    """Update one of the caller's own links.

    404 (not 403) when the link belongs to somebody else: the service scopes
    the lookup by owner, so a foreign id is never confirmed to exist.
    """
    user = await _load_user(db, current_user)
    try:
        return await profile_service.update_link(db, user, link_id=link_id, payload=payload)
    except profile_service.ProfileLinkNotFoundError as exc:
        raise _link_not_found(link_id) from exc


@router.delete("/links/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_link(
    link_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Remove one of the caller's own links (soft-delete)."""
    user = await _load_user(db, current_user)
    try:
        await profile_service.delete_link(db, user, link_id=link_id)
    except profile_service.ProfileLinkNotFoundError as exc:
        raise _link_not_found(link_id) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/permissions", response_model=UserPermissionsRead)
async def read_my_permissions(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserPermissionsRead:
    perms = await get_effective_permissions(db, current_user.user_id)
    return UserPermissionsRead(permissions=perms)


@me_root_router.get("/roles", response_model=list[str])
async def read_my_roles(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[str]:
    """Return the distinct role codes the caller currently holds.

    Powers the SPA's ``useMyRoles`` hook, which gates UI for
    multi-role users (e.g. show the manager dashboard tile only when
    the caller actually has ``manager``). Active = not soft-deleted
    AND ``active_until`` is NULL or in the future. Codes are
    de-duplicated across scopes.
    """
    rows = await access_control_api.get_role_assignments_for_user(db, current_user.user_id)
    seen: list[str] = []
    for assignment in rows:
        if assignment.role_code not in seen:
            seen.append(assignment.role_code)
    return seen


__all__ = ["me_root_router", "router"]
