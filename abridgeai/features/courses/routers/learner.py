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
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CoursePublic:
    course = await catalog_service.get_published_course_detail(db, course_id)
    if course is None:
        raise _not_found("course", course_id)
    return course


@router.get("/courses/{course_id}/content", response_model=CourseContentPublic)
async def get_course_content(
    course_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseContentPublic:
    tree = await catalog_service.get_published_course_content_for_learner(db, course_id)
    if tree is None:
        raise _not_found("course", course_id)
    return tree


@router.get("/courses/{course_id}/tags", response_model=list[TagPublic])
async def list_course_tags(
    course_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TagPublic]:
    tags = await catalog_service.list_published_course_tags_for_learner(db, course_id)
    if tags is None:
        raise _not_found("course", course_id)
    return tags


@router.get("/courses/{course_id}/outcomes", response_model=list[CourseLearningOutcomePublic])
async def list_course_outcomes(
    course_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CourseLearningOutcomePublic]:
    outcomes = await catalog_service.list_published_course_outcomes_for_learner(db, course_id)
    if outcomes is None:
        raise _not_found("course", course_id)
    return outcomes


@router.get("/courses/{course_id}/modules", response_model=list[ModulePublic])
async def list_course_modules(
    course_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModulePublic]:
    modules = await catalog_service.list_published_modules_for_course(db, course_id)
    if modules is None:
        raise _not_found("course", course_id)
    return modules


@router.get("/modules/{module_id}", response_model=ModulePublic)
async def get_module(
    module_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModulePublic:
    module = await catalog_service.get_published_module_for_learner(db, module_id)
    if module is None:
        raise _not_found("module", module_id)
    return module


@router.get("/modules/{module_id}/items", response_model=list[ModuleItemPublic])
async def list_module_items(
    module_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModuleItemPublic]:
    items = await catalog_service.list_visible_module_items_for_learner(db, module_id)
    if items is None:
        raise _not_found("module", module_id)
    return items


@router.get("/modules/{module_id}/lessons", response_model=list[LessonPublic])
async def list_module_lessons(
    module_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonPublic]:
    lessons = await catalog_service.list_published_lessons_for_module(db, module_id)
    if lessons is None:
        raise _not_found("module", module_id)
    return lessons


@router.get("/lessons/{lesson_id}", response_model=LessonPublic)
async def get_lesson(
    lesson_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonPublic:
    lesson = await catalog_service.get_published_lesson_for_learner(db, lesson_id)
    if lesson is None:
        raise _not_found("lesson", lesson_id)
    return lesson


@router.get("/lessons/{lesson_id}/resources", response_model=list[LessonResourcePublic])
async def list_lesson_resources(
    lesson_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonResourcePublic]:
    resources = await catalog_service.list_visible_lesson_resources_for_learner(db, lesson_id)
    if resources is None:
        raise _not_found("lesson", lesson_id)
    return resources


@router.get(
    "/lesson-resources/{resource_id}/download-url",
    response_model=ResourceDownloadUrlResponse,
)
async def get_lesson_resource_download_url(
    resource_id: UUID,
    _user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ResourceDownloadUrlResponse:
    download = await catalog_service.get_lesson_resource_download_url(db, resource_id)
    if download is None:
        raise _not_found("lesson_resource", resource_id)
    return ResourceDownloadUrlResponse(url=download.url, expires_at=download.expires_at.isoformat())


__all__ = ["me_courses_router", "router"]
