"""Sub-resource permission wrappers for the authoring router (T3.7 / FIX-SEC-1).

**FIX-SEC-1 invariant** -- closes the legacy permission gap at
``backend/app/routes/teacher/courses_router.py`` where module / lesson /
resource / outcome / module_item endpoints used ``Depends(get_current_user)``
ALONE -- meaning *any* authenticated user could PATCH a sibling-course's
sub-resources. Reconciliation §A9 + §E4 mandate that every sub-resource
endpoint walks UP to the owning course and runs the standard course-scoped
permission check from :func:`features.access_control.policies.require_course_permission`.

Each factory in this module returns a FastAPI dependency that:

1. Resolves the path-param sub-resource id to its owning ``course_id`` via
   :mod:`queries.resolution` -- the soft-delete loader filter is applied
   automatically to every :class:`SoftDeleteMixin` table touched by the join.
2. Calls the SAME logic as
   :func:`features.access_control.policies.require_course_permission` --
   owner short-circuit, then ``load_course_permissions`` lookup -- so the
   semantics match course-level endpoints exactly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.policies import can_manage_course
from abridgeai.features.courses.queries.resolution import (
    resolve_lesson_to_course,
    resolve_module_item_to_course,
    resolve_module_to_course,
    resolve_outcome_to_course,
    resolve_resource_to_course,
)

_DEFAULT_AUTHORING_PERMS: tuple[str, ...] = ("course.update",)

SubResourceDependency = Callable[..., Awaitable[CurrentUser]]


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def _permission_denied(*, codes: tuple[str, ...], course_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "permission_denied",
            "required": list(codes),
            "scope": "course",
            "course_id": str(course_id),
        },
    )


async def _check_course_permission(
    db: AsyncSession,
    current_user: CurrentUser,
    course_id: UUID,
    owner_user_id: UUID,
    codes: tuple[str, ...],
) -> CurrentUser:
    """Owner short-circuit + per-code ``can_manage_course`` lookup -- mirror of T1.11 policy."""
    if owner_user_id == current_user.user_id:
        return current_user

    for code in codes:
        if await can_manage_course(db, current_user.user_id, course_id, manage_perm=code):
            return current_user
    raise _permission_denied(codes=codes, course_id=course_id)


def require_module_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Build a dependency that walks ``module_id -> course_id`` and enforces course perms.

    The path parameter MUST be named ``module_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        module_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await resolve_module_to_course(db, module_id)
        if resolved is None:
            raise _not_found("module", module_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


def require_lesson_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``lesson_id -> module_id -> course_id`` and enforces course perms.

    The path parameter MUST be named ``lesson_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        lesson_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await resolve_lesson_to_course(db, lesson_id)
        if resolved is None:
            raise _not_found("lesson", lesson_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


def require_resource_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``resource_id -> lesson -> module -> course`` and enforces course perms.

    The path parameter MUST be named ``resource_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        resource_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await resolve_resource_to_course(db, resource_id)
        if resolved is None:
            raise _not_found("lesson_resource", resource_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


def require_module_item_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``module_item_id -> module -> course`` and enforces course perms.

    The path parameter MUST be named ``module_item_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        module_item_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await resolve_module_item_to_course(db, module_item_id)
        if resolved is None:
            raise _not_found("module_item", module_item_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


def require_outcome_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``outcome_id -> course`` and enforces course perms.

    The path parameter MUST be named ``outcome_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        outcome_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await resolve_outcome_to_course(db, outcome_id)
        if resolved is None:
            raise _not_found("course_outcome", outcome_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


__all__ = [
    "SubResourceDependency",
    "require_lesson_authoring_access",
    "require_module_authoring_access",
    "require_module_item_authoring_access",
    "require_outcome_authoring_access",
    "require_resource_authoring_access",
]
