"""Identity ``/users`` admin router — admin lookup + paginated list.

Two endpoints under ``/users`` for administrative consumers:

* ``GET /users/{user_id}`` — single-user lookup.
* ``GET /users`` — cursor-paginated user list.

Both endpoints REQUIRE the ``user.read`` permission (FIX-CRIT-4 — admin
endpoints must declare an explicit permission dependency; the legacy bug at
``backend/app/routes/users/router.py`` allowed any authenticated caller to
read any user). The companion test
``test_every_endpoint_has_permission_dependency`` walks
``router.routes`` and asserts each route's dependency chain contains a
``require_*`` factory; routes that depend only on
:func:`get_current_user` are forbidden.

Self-lookup: the public ``/users/me`` escape hatch lives in ``me.py``.
``/users/{id}`` does NOT short-circuit when ``id == current_user.user_id``
— the permission check is mandatory for the admin path.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_permission
from abridgeai.features.identity.schemas import UserListPage, UserRead
from abridgeai.features.identity.services import admin as admin_service

router = APIRouter(prefix="/users", tags=["users", "admin"])

_REQUIRE_USER_READ = require_permission("user.read")


def _not_found(user_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": "user", "id": str(user_id)},
    )


def _bad_cursor() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"error": "invalid_cursor"},
    )


@router.get("", response_model=UserListPage)
async def list_users(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_USER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> UserListPage:
    try:
        return await admin_service.list_users(db, cursor=cursor, limit=limit)
    except ValueError as exc:
        raise _bad_cursor() from exc


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_USER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserRead:
    user = await admin_service.get_user_with_profile(db, user_id)
    if user is None:
        raise _not_found(user_id)
    return user


__all__ = ["router"]
