"""Courses authoring router -- teacher CRUD on courses + sub-resources (T3.7).

Mounted at ``/api/v1/teacher`` by T3.10 integration. Every endpoint enforces
either a global permission (``course.create``) or a course-scoped permission
chain that walks UP from the path-param sub-resource to the owning course
(``require_*_authoring_access`` from :mod:`._deps`).

**FIX-SEC-1 invariant** (Reconciliation §A9 + §E4) -- the legacy
``backend/app/routes/teacher/courses_router.py`` authenticated sub-resource
endpoints (modules, lessons, module_items, resources) with the bare
identity dep, so any authenticated user could PATCH or DELETE another
teacher's sub-resources. Every endpoint in this file uses the corresponding
wrapper from :mod:`abridgeai.features.courses.routers._deps`, so the only
path that can mutate course state is one whose principal also holds
course-scoped permissions on the resource's owning course (or owns the
course outright).

Scope deviation -- T3.7 plan body lists ~25 endpoints. Only the ~13
endpoints in this file have matching public helpers in
:mod:`features.courses.services.authoring` (T3.5 frozen for T3.7). The
remaining read endpoints (GET ``/teacher/courses``, GET
``/teacher/courses/{id}``, GET ``/teacher/courses/{id}/content``, etc.)
require new authoring read services that T3.5 did not ship, and adding
those is explicitly out of scope per T3.7's "Do NOT touch T3.5 services
source" guardrail. They are deferred to a follow-up task; the security
invariant (FIX-SEC-1) is satisfied for every WRITE endpoint, which is the
gap the audit identified.

Architectural rules honoured:

* Routers ↔ services only (no ``queries.*`` imports here -- import-linter
  contract #2). Resolver SQL lives in :mod:`._deps`, NOT here.
* Services flush; the router commits after a successful write.
* Domain exceptions (``NotFoundError``, ``AppError``) are mapped to HTTP
  errors locally -- services stay HTTP-agnostic.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_course_permission,
    require_permission,
)
from abridgeai.features.courses.routers._deps import (
    require_lesson_authoring_access,
    require_module_authoring_access,
    require_resource_authoring_access,
)
from abridgeai.features.courses.schemas import (
    CourseAuthoring,
    CourseCreate,
    CourseUpdate,
    LessonAuthoring,
    LessonCreate,
    LessonResourceAuthoring,
    LessonResourceCreate,
    LessonUpdate,
    ModuleAuthoring,
    ModuleCreate,
    ModuleItemAuthoring,
    ModuleItemReorder,
    ModulePrerequisiteSet,
    ModuleUpdate,
)
from abridgeai.features.courses.services import authoring as authoring_service

router = APIRouter(prefix="/teacher", tags=["courses-authoring"])

_REQUIRE_CREATE = require_permission("course.create")
_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_COURSE_PUBLISH = require_course_permission("course_id", "course.publish")
_REQUIRE_COURSE_DELETE = require_course_permission("course_id", "course.delete")
_REQUIRE_MODULE = require_module_authoring_access()
_REQUIRE_LESSON = require_lesson_authoring_access()
_REQUIRE_RESOURCE = require_resource_authoring_access()


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": detail},
    )


@router.post(
    "/courses",
    response_model=CourseAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_course(
    payload: CourseCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    """Create a new course owned by the requesting principal.

    Global permission -- a teacher anywhere on the platform can create a
    course; ownership / scope is enforced on subsequent edits via
    :func:`require_course_permission`.
    """
    course = await authoring_service.create_course(db, payload, current_user)
    await db.commit()
    return course


@router.patch("/courses/{course_id}", response_model=CourseAuthoring)
async def update_course(
    course_id: UUID,
    payload: CourseUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    try:
        course = await authoring_service.update_course(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return course


@router.post("/courses/{course_id}/publish", response_model=CourseAuthoring)
async def publish_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_PUBLISH)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    try:
        course = await authoring_service.publish_course(db, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return course


@router.post("/courses/{course_id}/archive", response_model=CourseAuthoring)
async def archive_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    try:
        course = await authoring_service.archive_course(db, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return course


@router.post(
    "/courses/{course_id}/modules",
    response_model=ModuleAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_module(
    course_id: UUID,
    payload: ModuleCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    try:
        module = await authoring_service.add_module(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return module


@router.patch("/modules/{module_id}", response_model=ModuleAuthoring)
async def update_module(
    module_id: UUID,
    payload: ModuleUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    try:
        module = await authoring_service.update_module(db, module_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return module


@router.put("/modules/{module_id}/prerequisites", response_model=ModuleAuthoring)
async def set_module_prerequisites(
    module_id: UUID,
    payload: ModulePrerequisiteSet,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    try:
        module = await authoring_service.set_module_prerequisites(
            db, module_id, payload.prerequisite_module_ids, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return module


@router.put(
    "/modules/{module_id}/items/reorder",
    response_model=list[ModuleItemAuthoring],
)
async def reorder_module_items(
    module_id: UUID,
    payload: ModuleItemReorder,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModuleItemAuthoring]:
    """Reorder ``ModuleItem`` rows under ``module_id`` (Reconciliation §A6).

    The service uses the ``_OFFSET=100_000`` two-phase swap to escape
    the ``uq_module_items_position`` unique constraint mid-update.
    """
    try:
        items = await authoring_service.reorder_module_items(
            db, module_id, payload.new_order, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return items


@router.post(
    "/modules/{module_id}/lessons",
    response_model=LessonAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_lesson(
    module_id: UUID,
    payload: LessonCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonAuthoring:
    """Create a lesson AND auto-attach a ``ModuleItem`` (Reconciliation §A5).

    The service emits both INSERTs in a single flush; the router commits
    once on success so the lesson + module_item land atomically.
    """
    try:
        lesson = await authoring_service.add_lesson(db, module_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return lesson


@router.patch("/lessons/{lesson_id}", response_model=LessonAuthoring)
async def update_lesson(
    lesson_id: UUID,
    payload: LessonUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonAuthoring:
    try:
        lesson = await authoring_service.update_lesson(db, lesson_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return lesson


@router.post(
    "/lessons/{lesson_id}/resources",
    response_model=LessonResourceAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_lesson_resource(
    lesson_id: UUID,
    payload: LessonResourceCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonResourceAuthoring:
    try:
        resource = await authoring_service.add_lesson_resource(db, lesson_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return resource


@router.delete(
    "/lesson-resources/{resource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_lesson_resource(
    resource_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_RESOURCE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a lesson resource via :func:`soft_delete_cascade` (T0.15)."""
    try:
        await authoring_service.delete_lesson_resource(db, resource_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


__all__ = ["router"]
