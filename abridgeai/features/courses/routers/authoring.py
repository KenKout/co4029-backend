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

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
    require_permission,
)
from abridgeai.features.courses.routers._deps import (
    require_lesson_authoring_access,
    require_module_authoring_access,
    require_module_item_authoring_access,
    require_resource_authoring_access,
)
from abridgeai.features.courses.schemas import (
    CourseAuthoring,
    CourseContentAuthoring,
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
    ModuleItemUpdate,
    ModulePrerequisiteSet,
    ModuleUpdate,
)
from abridgeai.features.courses.services import authoring as authoring_service

router = APIRouter(prefix="/teacher", tags=["courses-authoring"])

_REQUIRE_CREATE = require_permission("course.create")
_REQUIRE_AUTHORING_LIST = require_any_permission("course.read.draft", "course.create")
_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_COURSE_PUBLISH = require_course_permission("course_id", "course.publish")
_REQUIRE_COURSE_DELETE = require_course_permission("course_id", "course.delete")
_REQUIRE_MODULE = require_module_authoring_access()
_REQUIRE_MODULE_ITEM = require_module_item_authoring_access()
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


def _conflict(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": detail},
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
    try:
        course = await authoring_service.create_course(db, payload, current_user)
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return course


class _SlugAvailability(BaseModel):
    available: bool


class _RosterEntry(BaseModel):
    """Roster row for ``GET /teacher/courses/{course_id}/roster``.

    Same shape as the HOD-scope ``RosterEntry`` in ``assignment.py`` —
    duplicated here intentionally to keep the authoring router free of
    cross-router imports. Both should evolve together; if Phase 7 lands
    a canonical ``EnrollmentRead`` shared schema, both can replace this.
    """

    enrollment_id: UUID
    student_id: UUID
    display_name: str | None = None
    primary_email: str
    status: str
    enrolled_at: datetime
    completed_at: datetime | None = None
    dropped_at: datetime | None = None


_LIST_ROSTER_SQL = text(
    """
    SELECT
        ce.id           AS enrollment_id,
        ce.student_id   AS student_id,
        u.primary_email AS primary_email,
        up.display_name AS display_name,
        ce.status       AS status,
        ce.enrolled_at  AS enrolled_at,
        ce.completed_at AS completed_at,
        ce.dropped_at   AS dropped_at
    FROM course_enrollments ce
    JOIN users u ON u.id = ce.student_id
    LEFT JOIN user_profiles up ON up.user_id = u.id
    WHERE ce.course_id = :course_id
    ORDER BY ce.enrolled_at DESC
    """
)


@router.get("/courses/check-slug", response_model=_SlugAvailability)
async def check_course_slug(
    slug: Annotated[str, Query(min_length=1, max_length=100)],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> _SlugAvailability:
    """Pre-flight check for the new-course form.

    Returns ``{"available": true}`` when ``slug`` is free in the caller's
    primary organization, ``false`` otherwise. Same auth as ``POST
    /teacher/courses`` so a 200 here implies the create attempt would not
    be rejected for permission reasons.
    """
    try:
        available = await authoring_service.check_course_slug_available(
            db, slug=slug, owner=current_user
        )
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return _SlugAvailability(available=available)


@router.get("/courses", response_model=list[CourseAuthoring])
async def list_authoring_courses(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_AUTHORING_LIST)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = False,
) -> list[CourseAuthoring]:
    """Courses the caller can author (owned + scope=course teacher assignments).

    Drafts and archived rows are visible to the author. Permission is
    intentionally lax (``course.read.draft`` OR ``course.create``) — the
    visibility filter happens in the service via owner/assignment match,
    not via permission gating, so a teacher seeing nothing is a UX
    problem rather than a 403.
    """
    return await authoring_service.list_authoring_courses_for_user(
        db, user=current_user, include_archived=include_archived
    )


@router.get("/courses/{course_id}", response_model=CourseAuthoring)
async def get_authoring_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    del current_user
    try:
        return await authoring_service.get_authoring_course(db, course_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.get(
    "/courses/{course_id}/content",
    response_model=CourseContentAuthoring,
)
async def get_authoring_course_content(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = False,
) -> CourseContentAuthoring:
    """Authoring content tree (drafts included) for ``course_id``."""
    del current_user
    try:
        tree = await authoring_service.get_authoring_content(
            db, course_id, include_archived=include_archived
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return CourseContentAuthoring.model_validate(tree)


@router.get(
    "/courses/{course_id}/roster",
    response_model=list[_RosterEntry],
)
async def get_authoring_course_roster(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[_RosterEntry]:
    """Roster of enrolled students for the teacher's course view.

    Same response shape as ``GET /dept/courses/{id}/roster`` (HOD-scope)
    but gated on ``course.update`` so the course owner / assigned teacher
    can read their own roster without HOD privileges.
    """
    del current_user
    rows = (await db.execute(_LIST_ROSTER_SQL, {"course_id": course_id})).mappings()
    return [_RosterEntry.model_validate(dict(row)) for row in rows]


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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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


@router.patch(
    "/module-items/{module_item_id}",
    response_model=ModuleItemAuthoring,
)
async def update_module_item(
    module_item_id: UUID,
    payload: ModuleItemUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE_ITEM)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleItemAuthoring:
    """Patch an item's ``unlock_rule_json`` (only mutable field).

    Position changes go through ``PUT /modules/{id}/items/reorder``;
    identity (lesson_id / quiz_id / interview_config_id) is immutable.
    """
    try:
        item = await authoring_service.update_module_item(db, module_item_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return item


@router.delete(
    "/module-items/{module_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_module_item(
    module_item_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE_ITEM)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a module item (un-pins from the order; target survives)."""
    try:
        await authoring_service.delete_module_item(db, module_item_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return lesson


@router.get("/lessons/{lesson_id}", response_model=LessonAuthoring)
async def get_authoring_lesson(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonAuthoring:
    """Authoring detail for a single lesson (drafts included)."""
    del current_user
    try:
        return await authoring_service.get_authoring_lesson(db, lesson_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


class _OutlineSection(BaseModel):
    """One section row in ``GET /lessons/{id}/outline``.

    Mirrors the SPA's ``OutlineSectionRead`` interface verbatim. Today
    this surface returns a single synthetic ``body`` section per lesson
    until the legacy ``build_lesson_outline`` semantic-section pipeline
    is ported (see ``quizzes/ai/pipelines/coverage.py:104``). The
    contract is deliberately stable so the SPA can render today and
    the section list can grow without an API break.
    """

    id: UUID
    title: str
    depth: int = 0
    chunk_count: int = 0
    char_count: int = 0
    page_range: tuple[int, int] = (0, 0)
    content_role: Literal["body", "summary", "review"] = "body"
    preview: str = ""


class _LessonOutline(BaseModel):
    lesson_id: UUID
    lesson_title: str
    sections: list[_OutlineSection]
    suggested_question_count: int = 0
    min_for_full_coverage: int = 0


@router.get("/lessons/{lesson_id}/outline", response_model=_LessonOutline)
async def get_lesson_outline(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> _LessonOutline:
    """Authoring outline preview (drafts visible).

    Surfaces under the teacher router (rather than the learner one) so
    the auth boundary matches the SPA's ``useLessonOutline`` consumer
    pages, and so drafts surface during course assembly. Returns a
    single synthetic ``body`` section until ``build_lesson_outline``
    lands; the contract matches the eventual semantic-section
    response field-for-field.
    """
    del current_user
    try:
        lesson = await authoring_service.get_authoring_lesson(db, lesson_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return _LessonOutline(
        lesson_id=lesson.id,
        lesson_title=lesson.title,
        sections=[
            _OutlineSection(
                id=lesson.id,
                title=lesson.title,
                depth=0,
                chunk_count=0,
                char_count=0,
                page_range=(0, 0),
                content_role="body",
                preview=(lesson.summary or "")[:200],
            )
        ],
        suggested_question_count=0,
        min_for_full_coverage=0,
    )


@router.get(
    "/lessons/{lesson_id}/resources",
    response_model=list[LessonResourceAuthoring],
)
async def list_authoring_lesson_resources(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonResourceAuthoring]:
    """All resources attached to ``lesson_id`` (drafts + hidden included).

    The learner-side ``/lessons/{id}/resources`` endpoint applies a
    ``visible_to_students=TRUE`` filter; this authoring sibling does
    not, so the teacher can see and reorder hidden / draft resources.
    """
    del current_user
    try:
        return await authoring_service.list_authoring_lesson_resources(db, lesson_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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


class _StreamUrlResponse(BaseModel):
    """Response shape for ``GET /teacher/lesson-resources/{id}/download-url``.

    Field name ``stream_url`` (not ``url``) matches the SPA's
    ``StreamUrlResponse`` TypeScript interface so the existing
    ``fetchTeacherResourceDownloadUrl`` helper works unchanged.
    """

    stream_url: str
    expires_at: datetime


@router.get(
    "/lesson-resources/{resource_id}/download-url",
    response_model=_StreamUrlResponse,
)
async def get_authoring_resource_download_url(
    resource_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_RESOURCE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> _StreamUrlResponse:
    """Mint a presigned GET URL for a teacher-visible resource.

    Same auth as the DELETE sibling — caller must have authoring access
    on the owning course (owner OR scope=course teacher assignment).
    Unlike the learner-side ``/lesson-resources/{id}/download-url`` this
    surfaces hidden / draft resources too.
    """
    del current_user
    try:
        url, expires_at = await authoring_service.get_authoring_resource_download_url(
            db, resource_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return _StreamUrlResponse(stream_url=url, expires_at=expires_at)


__all__ = ["router"]
