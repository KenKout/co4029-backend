"""Courses learner router — student catalog reads (T3.6).

Two ``APIRouter`` instances live here:

* :data:`router` — prefix ``""`` (paths self-prefix ``/courses``,
  ``/modules``, ``/lessons``, ``/lesson-resources``).
* :data:`me_courses_router` — prefix ``/me/courses`` for the
  enrolled-courses endpoint, mirroring the legacy SPA path
  ``/api/v1/me/courses`` (T1.9 :data:`me_router` follows the same shape).

All endpoints depend on :func:`get_current_user`. Visibility is filtered
upstream in :mod:`features.courses.queries.published` (T3.4) and
:mod:`features.courses.services.catalog` (T3.5). This router NEVER imports
authoring services / queries — enforced by import-linter contract
"Learner-facing routers do not touch authoring services" (T0.4 #4).

A bare ``ValueError`` from the service layer (e.g. malformed slug input
without ``organization_id``) maps to HTTP 400; ``None`` returns from the
service layer map to HTTP 404 so resource existence is never leaked.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.courses.schemas import (
    CourseContentPublic,
    CourseLearningOutcomePublic,
    CoursePage,
    CoursePublic,
    LessonPublic,
    LessonResourcePublic,
    ModuleItemPublic,
    ModulePublic,
    ResourceDownloadUrlResponse,
    TagPublic,
)
from abridgeai.features.courses.services import catalog as catalog_service

router = APIRouter(tags=["courses-learner"])
me_courses_router = APIRouter(prefix="/me/courses", tags=["courses-learner"])


def _not_found(resource: str, ident: str | UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(ident)},
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": detail},
    )


async def _resolve_org_or_400(db: AsyncSession, current_user: CurrentUser) -> UUID:
    org_id = await catalog_service.resolve_user_primary_organization_id(db, current_user.user_id)
    if org_id is None:
        raise _bad_request(
            "current user has no organization membership; cannot resolve catalog scope"
        )
    return org_id


async def _ensure_org_access(
    db: AsyncSession,
    current_user: CurrentUser,
    *,
    kind: str,
    resource_id: UUID,
) -> None:
    """Tenancy gate for by-id learner reads: 404 unless caller shares the org.

    Organizations do not share courses or quizzes. The learner catalog reads
    address courses / modules / lessons / resources by id under a bare
    ``get_current_user`` (students hold no course permission, so
    ``require_course_permission`` cannot gate them). This resolves the
    resource's owning organization and 404s when the caller is not a member —
    the same not-found shape a genuinely absent id returns, so an id cannot be
    used to probe another tenant's content.

    ``kind`` is one of ``course`` / ``module`` / ``lesson`` / ``resource``.
    """
    resource_org_id = await catalog_service.organization_id_for_course_resource(
        db, kind=kind, resource_id=resource_id
    )
    allowed = await catalog_service.user_can_access_org_resource(
        db, user_id=current_user.user_id, resource_org_id=resource_org_id
    )
    if not allowed:
        raise _not_found(kind, resource_id)


async def _ensure_course_enrolled(
    db: AsyncSession,
    current_user: CurrentUser,
    course_id: UUID,
) -> None:
    """BR gate for learner item reads: 403 unless enrolled (or course manager).

    A published course's landing page (detail, outcomes, tags) stays open to
    every org member, but NONE of its items — content tree, modules, lessons,
    resources, downloads — may be reached without an ``active`` /
    ``completed`` enrollment. Course managers (owner / ``course.update``
    grants) bypass so a teacher previewing their own course still passes;
    ``dropped`` / ``waitlisted`` students are blocked. Runs after
    :func:`_ensure_org_access`, so cross-tenant probes still 404 first.
    """
    from abridgeai.features.access_control.policies import (  # noqa: PLC0415
        can_manage_course,
    )
    from abridgeai.features.enrollments.api import public as enrollments_api  # noqa: PLC0415

    if await enrollments_api.has_active_or_completed_enrollment(
        db, student_id=current_user.user_id, course_id=course_id
    ):
        return
    if await can_manage_course(db, current_user.user_id, course_id):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "not_enrolled",
            "course_id": str(course_id),
        },
    )


async def _ensure_lesson_unlocked(
    db: AsyncSession, current_user: CurrentUser, lesson_id: UUID
) -> None:
    """FR-4.5 server-side gate: 403 with unlock requirements when locked.

    Combines the three unlock gates (prerequisites, SR coverage τ over
    cards with EF ≥ ef_min_unlock, optional interview pass) via the
    spaced_repetition public API. Disabled globally by the
    ``LESSON_GATING_ENFORCED=false`` emergency switch.

    The SR public API is imported lazily: its unlock helper reads back
    through ``courses.api.public``, so a module-level import here would
    create a circular import at app start-up.
    """
    if not get_settings().lesson_gating_enforced:
        return

    from abridgeai.features.spaced_repetition.api.public import (  # noqa: PLC0415
        check_lesson_unlock,
    )

    unlock = await check_lesson_unlock(
        db, student_id=current_user.user_id, lesson_id=lesson_id
    )
    if unlock.eligible:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "lesson_locked",
            "lesson_id": str(lesson_id),
            "current_ratio": unlock.current_ratio,
            "required_ratio": unlock.required_ratio,
            "total_cards": unlock.total_cards,
            "passing_cards": unlock.passing_cards,
            "prerequisites_met": unlock.prereq_lesson_ids_unlocked,
            "interview_pass_required": unlock.interview_pass_required,
            "interview_passed": unlock.interview_passed,
            "next_unlock_estimate": unlock.next_unlock_estimate,
        },
    )


@router.get("/courses", response_model=CoursePage)
async def list_courses(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> CoursePage:
    """Cursor-paginated list of published courses in the user's organization.

    The legacy ``?owned=true`` query param is intentionally NOT declared
    (plan §4300, Reconciliation §A14): a learner endpoint never returns
    authoring data. FastAPI silently drops unknown query params, so a
    request with ``?owned=true`` lands here with ``cursor=None`` /
    ``limit=20`` and returns the standard published catalog.
    """
    org_id = await _resolve_org_or_400(db, current_user)
    page = await catalog_service.list_published_courses_for_user(
        db, current_user.user_id, organization_id=org_id, limit=limit, cursor=cursor
    )
    return CoursePage(items=page.items, next_cursor=page.next_cursor)


@me_courses_router.get("", response_model=CoursePage)
async def list_my_enrolled_courses(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> CoursePage:
    page = await catalog_service.list_enrolled_courses_for_user(
        db, current_user.user_id, limit=limit, cursor=cursor
    )
    return CoursePage(items=page.items, next_cursor=page.next_cursor)


@router.get("/courses/by-slug/{slug}", response_model=CoursePublic)
async def get_course_by_slug(
    slug: str,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CoursePublic:
    org_id = await _resolve_org_or_400(db, current_user)
    try:
        course = await catalog_service.get_published_course_detail(db, slug, organization_id=org_id)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    if course is None:
        raise _not_found("course", slug)
    return course


@router.get("/courses/{course_id}", response_model=CoursePublic)
async def get_course(
    course_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CoursePublic:
    await _ensure_org_access(db, user, kind="course", resource_id=course_id)
    course = await catalog_service.get_published_course_detail(db, course_id)
    if course is None:
        raise _not_found("course", course_id)
    return course


@router.get("/courses/{course_id}/content", response_model=CourseContentPublic)
async def get_course_content(
    course_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseContentPublic:
    await _ensure_org_access(db, user, kind="course", resource_id=course_id)
    await _ensure_course_enrolled(db, user, course_id)
    tree = await catalog_service.get_published_course_content_for_learner(db, course_id)
    if tree is None:
        raise _not_found("course", course_id)
    return tree


@router.get("/courses/{course_id}/tags", response_model=list[TagPublic])
async def list_course_tags(
    course_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TagPublic]:
    await _ensure_org_access(db, user, kind="course", resource_id=course_id)
    tags = await catalog_service.list_published_course_tags_for_learner(db, course_id)
    if tags is None:
        raise _not_found("course", course_id)
    return tags


@router.get("/courses/{course_id}/outcomes", response_model=list[CourseLearningOutcomePublic])
async def list_course_outcomes(
    course_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CourseLearningOutcomePublic]:
    await _ensure_org_access(db, user, kind="course", resource_id=course_id)
    outcomes = await catalog_service.list_published_course_outcomes_for_learner(db, course_id)
    if outcomes is None:
        raise _not_found("course", course_id)
    return outcomes


@router.get("/courses/{course_id}/modules", response_model=list[ModulePublic])
async def list_course_modules(
    course_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModulePublic]:
    await _ensure_org_access(db, user, kind="course", resource_id=course_id)
    await _ensure_course_enrolled(db, user, course_id)
    modules = await catalog_service.list_published_modules_for_course(db, course_id)
    if modules is None:
        raise _not_found("course", course_id)
    return modules


@router.get("/modules/{module_id}", response_model=ModulePublic)
async def get_module(
    module_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModulePublic:
    await _ensure_org_access(db, user, kind="module", resource_id=module_id)
    course_id = await catalog_service.course_id_for_course_resource(
        db, kind="module", resource_id=module_id
    )
    if course_id is not None:
        await _ensure_course_enrolled(db, user, course_id)
    module = await catalog_service.get_published_module_for_learner(db, module_id)
    if module is None:
        raise _not_found("module", module_id)
    return module


@router.get("/modules/{module_id}/items", response_model=list[ModuleItemPublic])
async def list_module_items(
    module_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModuleItemPublic]:
    await _ensure_org_access(db, user, kind="module", resource_id=module_id)
    course_id = await catalog_service.course_id_for_course_resource(
        db, kind="module", resource_id=module_id
    )
    if course_id is not None:
        await _ensure_course_enrolled(db, user, course_id)
    items = await catalog_service.list_visible_module_items_for_learner(db, module_id)
    if items is None:
        raise _not_found("module", module_id)
    return items


@router.get("/modules/{module_id}/lessons", response_model=list[LessonPublic])
async def list_module_lessons(
    module_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonPublic]:
    await _ensure_org_access(db, user, kind="module", resource_id=module_id)
    course_id = await catalog_service.course_id_for_course_resource(
        db, kind="module", resource_id=module_id
    )
    if course_id is not None:
        await _ensure_course_enrolled(db, user, course_id)
    lessons = await catalog_service.list_published_lessons_for_module(db, module_id)
    if lessons is None:
        raise _not_found("module", module_id)
    return lessons


@router.get("/lessons/{lesson_id}", response_model=LessonPublic)
async def get_lesson(
    lesson_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonPublic:
    await _ensure_org_access(db, user, kind="lesson", resource_id=lesson_id)
    course_id = await catalog_service.course_id_for_course_resource(
        db, kind="lesson", resource_id=lesson_id
    )
    if course_id is not None:
        await _ensure_course_enrolled(db, user, course_id)
    lesson = await catalog_service.get_published_lesson_for_learner(db, lesson_id)
    if lesson is None:
        raise _not_found("lesson", lesson_id)
    await _ensure_lesson_unlocked(db, user, lesson_id)
    return lesson


@router.get("/lessons/{lesson_id}/resources", response_model=list[LessonResourcePublic])
async def list_lesson_resources(
    lesson_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonResourcePublic]:
    await _ensure_org_access(db, user, kind="lesson", resource_id=lesson_id)
    course_id = await catalog_service.course_id_for_course_resource(
        db, kind="lesson", resource_id=lesson_id
    )
    if course_id is not None:
        await _ensure_course_enrolled(db, user, course_id)
    resources = await catalog_service.list_visible_lesson_resources_for_learner(db, lesson_id)
    if resources is None:
        raise _not_found("lesson", lesson_id)
    await _ensure_lesson_unlocked(db, user, lesson_id)
    return resources


@router.get(
    "/lesson-resources/{resource_id}/download-url",
    response_model=ResourceDownloadUrlResponse,
)
async def get_lesson_resource_download_url(
    resource_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ResourceDownloadUrlResponse:
    await _ensure_org_access(db, user, kind="resource", resource_id=resource_id)
    course_id = await catalog_service.course_id_for_course_resource(
        db, kind="resource", resource_id=resource_id
    )
    if course_id is not None:
        await _ensure_course_enrolled(db, user, course_id)
    lesson_id = await catalog_service.get_visible_resource_lesson_id(db, resource_id)
    if lesson_id is not None:
        await _ensure_lesson_unlocked(db, user, lesson_id)
    download = await catalog_service.get_lesson_resource_download_url(db, resource_id)
    if download is None:
        raise _not_found("lesson_resource", resource_id)
    return ResourceDownloadUrlResponse(url=download.url, expires_at=download.expires_at.isoformat())


__all__ = ["me_courses_router", "router"]
