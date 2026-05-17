"""Sub-resource permission wrappers for the authoring router (T3.7 / FIX-SEC-1).

**FIX-SEC-1 invariant** -- closes the legacy permission gap at
``backend/app/routes/teacher/courses_router.py`` where module / lesson /
resource / outcome / module_item endpoints used ``Depends(get_current_user)``
ALONE -- meaning *any* authenticated user could PATCH a sibling-course's
sub-resources. Reconciliation §A9 + §E4 mandate that every sub-resource
endpoint walks UP to the owning course and runs the standard course-scoped
permission check from :func:`features.access_control.policies.require_course_permission`.

Each factory in this module returns a FastAPI dependency that:

1. Resolves the path-param sub-resource id to its owning ``course_id`` via a
   raw SQL ``SELECT`` (mirrors the inline ``_COURSE_OWNER_CTX_SQL`` pattern
   from T1.11 :mod:`features.access_control.policies`).
2. Calls the SAME logic as
   :func:`features.access_control.policies.require_course_permission` --
   owner short-circuit, then ``load_course_permissions`` lookup -- so the
   semantics match course-level endpoints exactly.

Resolution is performed inline (raw SQL, no extra query helper) per task
T3.7 instructions: keep the sub-resource → course mapping in this file
to avoid touching T3.4's :mod:`features.courses.queries.authoring` module
in a non-surgical way.

Why a separate module: keeping these wrappers out of ``authoring.py`` makes
the security perimeter independently auditable -- the FIX-SEC-1 grep test
asserts every authoring endpoint depends on a wrapper from this file (or
the course-level :func:`require_course_permission` factory), never on a
bare :func:`get_current_user`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.policies import can_manage_course

_DEFAULT_AUTHORING_PERMS: tuple[str, ...] = ("course.update",)

SubResourceDependency = Callable[..., Awaitable[CurrentUser]]


_RESOURCE_TO_COURSE_SQL = text(
    """
    SELECT m.course_id     AS course_id,
           c.owner_user_id AS owner_user_id
    FROM lesson_resources lr
    JOIN lessons l ON l.id = lr.lesson_id
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE lr.id = :resource_id
      AND lr.deleted_at IS NULL
      AND l.deleted_at IS NULL
      AND m.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)

_LESSON_TO_COURSE_SQL = text(
    """
    SELECT m.course_id     AS course_id,
           c.owner_user_id AS owner_user_id
    FROM lessons l
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE l.id = :lesson_id
      AND l.deleted_at IS NULL
      AND m.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)

_MODULE_TO_COURSE_SQL = text(
    """
    SELECT m.course_id     AS course_id,
           c.owner_user_id AS owner_user_id
    FROM modules m
    JOIN courses c ON c.id = m.course_id
    WHERE m.id = :module_id
      AND m.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)

_MODULE_ITEM_TO_COURSE_SQL = text(
    """
    SELECT m.course_id     AS course_id,
           c.owner_user_id AS owner_user_id
    FROM module_items mi
    JOIN modules m ON m.id = mi.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE mi.id = :module_item_id
      AND mi.deleted_at IS NULL
      AND m.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)

_OUTCOME_TO_COURSE_SQL = text(
    """
    SELECT clo.course_id   AS course_id,
           c.owner_user_id AS owner_user_id
    FROM course_learning_outcomes clo
    JOIN courses c ON c.id = clo.course_id
    WHERE clo.id = :outcome_id
      AND clo.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)


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
    """Build a dependency that walks ``module_id → course_id`` and enforces course perms.

    The path parameter MUST be named ``module_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        module_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_MODULE_TO_COURSE_SQL, {"module_id": module_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("module", module_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


def require_lesson_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``lesson_id → module_id → course_id`` and enforces course perms.

    The path parameter MUST be named ``lesson_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        lesson_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_LESSON_TO_COURSE_SQL, {"lesson_id": lesson_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("lesson", lesson_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


def require_resource_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``resource_id → lesson → module → course`` and enforces course perms.

    The path parameter MUST be named ``resource_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        resource_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_RESOURCE_TO_COURSE_SQL, {"resource_id": resource_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("lesson_resource", resource_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


def require_module_item_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``module_item_id → module → course`` and enforces course perms.

    The path parameter MUST be named ``module_item_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        module_item_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_MODULE_ITEM_TO_COURSE_SQL, {"module_item_id": module_item_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("module_item", module_item_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


def require_outcome_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``outcome_id → course`` and enforces course perms.

    The path parameter MUST be named ``outcome_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        outcome_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_OUTCOME_TO_COURSE_SQL, {"outcome_id": outcome_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("course_outcome", outcome_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


__all__ = [
    "SubResourceDependency",
    "require_lesson_authoring_access",
    "require_module_authoring_access",
    "require_module_item_authoring_access",
    "require_outcome_authoring_access",
    "require_resource_authoring_access",
]
