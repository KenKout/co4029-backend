from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_permission,
)
from abridgeai.features.career_paths.schemas import (
    CareerPathAuthoring,
    CareerPathCourseAdd,
    CareerPathCourseAuthoring,
    CareerPathCourseReorder,
    CareerPathCreate,
    CareerPathStudentEnroll,
    CareerPathUpdate,
    PathReadinessOverview,
    StudentCareerEnrollmentAuthoring,
    StudentPathProgressAuthoring,
)
from abridgeai.features.career_paths.services import authoring as authoring_service
from abridgeai.features.career_paths.services import enrollment as enrollment_service
from abridgeai.features.career_paths.services import readiness as readiness_service

management_router = APIRouter(prefix="/management/career-paths", tags=["career-paths-authoring"])
teacher_router = APIRouter(prefix="/teacher/career-paths", tags=["career-paths-authoring"])


_REQUIRE_PATH_MANAGE = require_any_permission("course.create", "course.update", "system.administer")
_REQUIRE_PATH_PUBLISH = require_any_permission("course.publish", "system.administer")
_REQUIRE_PATH_DELETE = require_any_permission("course.delete", "system.administer")
_REQUIRE_PATH_ENROLL = require_any_permission("course.enrollment.create", "system.administer")
_REQUIRE_PATH_UNENROLL = require_any_permission("course.enrollment.remove", "system.administer")
_REQUIRE_PATH_ROSTER_READ = require_any_permission(
    "course.enrollment.read",
    "progress.read.cohort",
    "system.administer",
)
_REQUIRE_PATH_CREATE = require_permission("course.create")


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


def _conflict(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": detail},
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": detail},
    )


async def _ensure_caller_in_path_org(
    db: AsyncSession, current_user: CurrentUser, career_path_id: UUID
) -> None:
    """FR-2.6 — managers only see roster/readiness for paths in THEIR org.

    The permission deps above are permission-level (global scope); a
    manager from org B with ``course.enrollment.read`` must not read
    org A's student emails. Membership in the path's organization OR
    ``system.administer`` passes; everything else 404s (no existence
    leak — matches the resource-not-found shape).
    """
    path = await authoring_service.get_career_path(db, career_path_id)
    if await access_control_api.is_user_member_of_org(
        db, user_id=current_user.user_id, org_id=path.organization_id
    ):
        return
    permissions = await access_control_api.get_active_permissions(db, current_user.user_id)
    if any(p.code == "system.administer" for p in permissions):
        return
    raise _not_found(f"Career path {career_path_id} not found")


@management_router.post(
    "",
    response_model=CareerPathAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_career_path(
    payload: CareerPathCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathAuthoring:
    try:
        result = await authoring_service.create_career_path(db, payload, current_user)
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return result


@management_router.get(
    "",
    response_model=list[CareerPathAuthoring],
)
async def list_career_paths(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: UUID | None = None,
    include_archived: bool = False,
) -> list[CareerPathAuthoring]:
    """List career paths for an organization.

    ``organization_id`` is OPTIONAL: when omitted the caller's primary
    org is resolved from the bearer token, matching the contract used by
    POST. An explicit value is honoured (so platform admins can list any
    org); managers without scope and no override get a 400 instead of a
    confusing empty list.
    """
    if organization_id is None:
        from abridgeai.features.career_paths.queries.published import (  # noqa: PLC0415
            get_user_primary_organization_id,
        )

        organization_id = await get_user_primary_organization_id(db, current_user.user_id)
        if organization_id is None:
            raise _bad_request(
                f"User {current_user.user_id} has no primary organization; "
                "pass ?organization_id=... to list paths for a specific org."
            )
    return await authoring_service.list_career_paths_for_org(
        db, organization_id, include_archived=include_archived
    )


@management_router.get(
    "/{career_path_id}",
    response_model=CareerPathAuthoring,
)
async def get_career_path(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathAuthoring:
    del current_user
    try:
        return await authoring_service.get_career_path(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@management_router.patch(
    "/{career_path_id}",
    response_model=CareerPathAuthoring,
)
async def update_career_path(
    career_path_id: UUID,
    payload: CareerPathUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathAuthoring:
    try:
        result = await authoring_service.update_career_path(
            db, career_path_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return result


@management_router.get(
    "/{career_path_id}/courses",
    response_model=list[CareerPathCourseAuthoring],
)
async def list_career_path_courses(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathCourseAuthoring]:
    del current_user
    try:
        return await authoring_service.list_career_path_courses(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@management_router.post(
    "/{career_path_id}/courses",
    response_model=CareerPathCourseAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def add_course_to_path(
    career_path_id: UUID,
    payload: CareerPathCourseAdd,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathCourseAuthoring:
    try:
        result = await authoring_service.add_course_to_path(
            db,
            career_path_id,
            payload.course_id,
            position=payload.position,
            is_required=payload.is_required,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return result


@management_router.put(
    "/{career_path_id}/courses/reorder",
    response_model=list[CareerPathCourseAuthoring],
)
async def reorder_courses_in_path(
    career_path_id: UUID,
    payload: CareerPathCourseReorder,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathCourseAuthoring]:
    try:
        result = await authoring_service.reorder_courses_in_path(
            db, career_path_id, payload.course_ids, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@management_router.delete(
    "/{career_path_id}/courses/{course_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_course_from_path(
    career_path_id: UUID,
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await authoring_service.remove_course_from_path(db, career_path_id, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@management_router.post(
    "/{career_path_id}/students",
    response_model=StudentCareerEnrollmentAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def enroll_student_in_path(
    career_path_id: UUID,
    payload: CareerPathStudentEnroll,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_ENROLL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StudentCareerEnrollmentAuthoring:
    try:
        result = await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=career_path_id,
            student_id=payload.student_id,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return result


@management_router.delete(
    "/{career_path_id}/students/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def unenroll_student_from_path(
    career_path_id: UUID,
    student_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_UNENROLL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await enrollment_service.unenroll_student(
            db,
            career_path_id=career_path_id,
            student_id=student_id,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@management_router.post(
    "/{career_path_id}/publish",
    response_model=CareerPathAuthoring,
)
async def publish_path(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_PUBLISH)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathAuthoring:
    try:
        result = await authoring_service.publish_path(db, career_path_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@management_router.post(
    "/{career_path_id}/archive",
    response_model=CareerPathAuthoring,
)
async def archive_path(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathAuthoring:
    try:
        result = await authoring_service.archive_path(db, career_path_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return result


@teacher_router.get(
    "/{career_path_id}/students/progress",
    response_model=list[StudentPathProgressAuthoring],
)
async def list_path_roster_progress(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_ROSTER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[StudentPathProgressAuthoring]:
    await _ensure_caller_in_path_org(db, current_user, career_path_id)
    return await enrollment_service.get_roster_progress(db, career_path_id)


@management_router.get(
    "/{career_path_id}/readiness",
    response_model=PathReadinessOverview,
)
async def get_path_readiness_overview(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_ROSTER_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PathReadinessOverview:
    """Readiness aggregate (FR-6.8): latest snapshot per actively-enrolled
    student + path average. Rubric details are never included.

    Org-scoped (FR-2.6): caller must belong to the path's organization
    or hold ``system.administer`` — same gate as the roster read."""
    await _ensure_caller_in_path_org(db, current_user, career_path_id)
    return await readiness_service.get_path_readiness_overview(db, career_path_id)


__all__ = ["management_router", "teacher_router"]
