"""Identity ``/users/me`` router — self-service profile + permissions.

Three endpoints under ``/users/me`` that the SPA uses for the signed-in user:

* ``GET /users/me`` — current user details (with profile if present).
* ``PATCH /users/me/profile`` — partial update of the user's profile fields.
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

from fastapi import APIRouter, Depends, HTTPException, Request, status
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
    return await get_current_user_read(db, user)


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
