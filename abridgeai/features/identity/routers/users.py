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
from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.core.pagination import PageResponse
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import require_permission
from abridgeai.features.identity.schemas import (
    UserCreate,
    UserListPage,
    UserOverviewRead,
    UserRead,
)
from abridgeai.features.identity.services import admin as admin_service

router = APIRouter(prefix="/users", tags=["users", "admin"])

_REQUIRE_USER_READ = require_permission("user.read")
_REQUIRE_SYSTEM_ADMINISTER = require_permission("system.administer")


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


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SYSTEM_ADMINISTER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserRead:
    """Admin invite — create a user + profile + org membership + role.

    Only a platform admin (``system.administer``) may provision accounts
    manually: this bypasses the invite-only pre-registration gate by
    design, so the audience is deliberately narrow. The created account
    is ``active`` and can sign in via Google OAuth immediately.
    """
    try:
        result = await admin_service.create_user_account(
            db, payload=payload, actor_id=current_user.user_id
        )
    except ConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "conflict", "message": str(exc)},
        ) from exc
    await db.commit()
    return result


# Declared before ``/{user_id}`` so ``/users/search`` isn't captured by the
# path-param route (which would 422 on UUID parsing). Additive to the cursor
# ``GET /users`` above — this one backs the page-numbered DataTable.
@router.get("/search", response_model=PageResponse[UserRead])
async def search_users(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_USER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=200)] = None,
    user_status: Annotated[str | None, Query(alias="status")] = None,
    role: Annotated[str | None, Query(max_length=50)] = None,
    organization: Annotated[UUID | None, Query()] = None,
    org_unit: Annotated[UUID | None, Query()] = None,
    sort: Annotated[str | None, Query()] = None,
    sort_dir: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> PageResponse[UserRead]:
    """Page-numbered admin user list with server-side search (email /
    display name), optional ``status`` / ``role`` / ``organization`` filters,
    and whitelisted sort (``email`` / ``status`` / ``created_at``). ``role``
    filters to users holding that role code at any scope; ``organization``
    filters to members of that org; ``org_unit`` narrows to one Faculty
    **and every unit beneath it**, backing the org-tree scope picker.

    Org scope: callers holding ``system.administer`` may search globally and
    pick any ``organization``. Everyone else (e.g. a manager with
    ``user.read``) is forced to their own primary organization — the
    ``organization`` query param is ignored and replaced with the caller's
    org, so a manager can never enumerate users outside their org.
    """
    if not current_user.has_permission("system.administer"):
        caller_org = await access_control_api.get_user_primary_org(db, current_user.user_id)
        if caller_org is None:
            return PageResponse[UserRead](
                items=[],
                total=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )
        organization = caller_org.id
    result = await admin_service.search_users(
        db,
        status=user_status,
        search=search,
        role=role,
        organization=organization,
        # Unlike `organization`, this is NOT overridden for non-admins: the
        # org filter above already pins the caller to their own org, and the
        # unit filter can only narrow further within it. A unit id from
        # another org intersects to the empty set rather than leaking.
        org_unit=org_unit,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )
    return PageResponse[UserRead](
        items=result.items,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


@router.get("/by-ids", response_model=list[UserRead])
async def get_users_by_ids(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_USER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    ids: Annotated[str, Query(description="Comma-separated user UUIDs (max 100).")],
) -> list[UserRead]:
    """Batch user lookup — resolve a set of UUIDs to displayable users.

    Powers the audit screens, which show actor/subject names instead of raw
    UUIDs. Returns one entry per resolvable id (missing ids are simply
    absent, matching the underlying batch API); non-admin callers are
    restricted to their own org exactly like ``GET /users/search``.
    """
    raw = [part for part in ids.split(",") if part.strip()]
    try:
        user_ids = [UUID(part.strip()) for part in raw]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "invalid_ids", "message": "ids must be UUIDs"},
        ) from exc
    if len(user_ids) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "too_many_ids", "message": "at most 100 ids"},
        )
    if not current_user.has_permission("system.administer"):
        caller_org = await access_control_api.get_user_primary_org(db, current_user.user_id)
        if caller_org is None:
            return []
        org_members = set(await access_control_api.list_user_ids_in_org(db, caller_org.id))
        user_ids = [uid for uid in user_ids if uid in org_members]
    users = []
    for uid in user_ids:
        user = await admin_service.get_user_with_profile(db, uid)
        if user is not None:
            users.append(user)
    return users


async def _assert_can_view_user(
    db: AsyncSession,
    current_user: CurrentUser,
    user_id: UUID,
) -> None:
    """Org-scope guard for the manager/HOD user-detail route.

    ``system.administer`` bypasses (global view). Everyone else must share
    the caller's primary organization with the target; a cross-org (or
    missing) lookup 404s so the endpoint cannot be used to enumerate
    another org's users. Unlike disable/enable there is NO peer block —
    managers and HODs may view peer accounts, they just get identity-only
    data (the service attaches learning sections only for students and
    teachers).
    """
    if current_user.has_permission("system.administer"):
        return
    caller_org = await access_control_api.get_user_primary_org(db, current_user.user_id)
    if caller_org is None:
        raise _not_found(user_id)
    if not await access_control_api.is_user_member_of_org(
        db, user_id=user_id, org_id=caller_org.id
    ):
        raise _not_found(user_id)


@router.get("/{user_id}/overview", response_model=UserOverviewRead)
async def get_user_overview(
    user_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_USER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserOverviewRead:
    """Org-scoped user detail for managers / HODs.

    Basic identity always; students additionally get enrolled courses with
    per-course progress, career-path enrolments with progress, and the
    latest activity time; teachers get their assigned courses. Manager /
    HOD / admin targets return identity only. Cross-org lookups 404.
    """
    await _assert_can_view_user(db, current_user, user_id)
    try:
        return await admin_service.get_user_overview(db, user_id=user_id)
    except NotFoundError as exc:
        raise _not_found(user_id) from exc


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
