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
   an ORM ``select(...).join(...)`` walk -- the soft-delete loader filter
   (``core/db/soft_delete.py``) is applied automatically to every
   :class:`SoftDeleteMixin` table touched by the join, so manual
   ``deleted_at IS NULL`` clauses are unnecessary.
2. Calls the SAME logic as
   :func:`features.access_control.policies.require_course_permission` --
   owner short-circuit, then ``load_course_permissions`` lookup -- so the
   semantics match course-level endpoints exactly.

Resolution lives inline (helper functions, no separate query module) per
task T3.7's "keep the sub-resource → course mapping in this file" rule:
the security perimeter stays independently auditable and the FIX-SEC-1
grep test asserts every authoring endpoint depends on a wrapper from
this file (or the course-level :func:`require_course_permission`
factory), never on a bare :func:`get_current_user`.

T7 (orm-consolidation) migrated this module from raw ``text()`` SELECTs
to ORM ``select()`` joins. The soft-delete filter is now auto-applied
via the do_orm_execute listener registered in
:mod:`abridgeai.core.db.soft_delete`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.policies import can_manage_course
from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)

_DEFAULT_AUTHORING_PERMS: tuple[str, ...] = ("course.update",)

SubResourceDependency = Callable[..., Awaitable[CurrentUser]]


async def _resolve_module_to_course(db: AsyncSession, module_id: UUID) -> tuple[UUID, UUID] | None:
    """Walk ``module_id -> course_id`` via the FK column.

    Returns ``(course_id, owner_user_id)`` or ``None`` when the module
    is missing / soft-deleted (the soft-delete loader filter hides
    tombstoned rows on both ``modules`` and ``courses``).
    """
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Course, Course.id == Module.course_id)
        .where(Module.id == module_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def _resolve_lesson_to_course(db: AsyncSession, lesson_id: UUID) -> tuple[UUID, UUID] | None:
    """Walk ``lesson_id -> module_id -> course_id`` via FK columns."""
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .where(Lesson.id == lesson_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def _resolve_resource_to_course(
    db: AsyncSession, resource_id: UUID
) -> tuple[UUID, UUID] | None:
    """Walk ``resource_id -> lesson -> module -> course`` via FK columns."""
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .join(LessonResource, LessonResource.lesson_id == Lesson.id)
        .where(LessonResource.id == resource_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def _resolve_module_item_to_course(
    db: AsyncSession, module_item_id: UUID
) -> tuple[UUID, UUID] | None:
    """Walk ``module_item_id -> module -> course`` via FK columns."""
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Course, Course.id == Module.course_id)
        .join(ModuleItem, ModuleItem.module_id == Module.id)
        .where(ModuleItem.id == module_item_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def _resolve_outcome_to_course(
    db: AsyncSession, outcome_id: UUID
) -> tuple[UUID, UUID] | None:
    """Walk ``outcome_id -> course`` via the FK column."""
    stmt = (
        select(CourseLearningOutcome.course_id, Course.owner_user_id)
        .join(Course, Course.id == CourseLearningOutcome.course_id)
        .where(CourseLearningOutcome.id == outcome_id)
    )
    return (await db.execute(stmt)).tuples().first()


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
        resolved = await _resolve_module_to_course(db, module_id)
        if resolved is None:
            raise _not_found("module", module_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

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
        resolved = await _resolve_lesson_to_course(db, lesson_id)
        if resolved is None:
            raise _not_found("lesson", lesson_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

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
        resolved = await _resolve_resource_to_course(db, resource_id)
        if resolved is None:
            raise _not_found("lesson_resource", resource_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

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
        resolved = await _resolve_module_item_to_course(db, module_item_id)
        if resolved is None:
            raise _not_found("module_item", module_item_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

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
        resolved = await _resolve_outcome_to_course(db, outcome_id)
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
