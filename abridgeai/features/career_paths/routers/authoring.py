from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_org_access,
    require_permission,
)
from abridgeai.features.career_paths.schemas import (
    CareerPathAuthoring,
    CareerPathCourseAdd,
    CareerPathCourseAuthoring,
    CareerPathCourseCandidate,
    CareerPathCourseMove,
    CareerPathCoursePatch,
    CareerPathCourseReorder,
    CareerPathCreate,
    CareerPathImpactRead,
    CareerPathStageAuthoring,
    CareerPathStageCreate,
    CareerPathStageReorder,
    CareerPathStageReorderResult,
    CareerPathStageUpdate,
    CareerPathStudentEnroll,
    CareerPathUpdate,
    CareerPathVersionRead,
    PathReadinessOverview,
    StudentCareerEnrollmentAuthoring,
    StudentPathProgressAuthoring,
)
from abridgeai.features.career_paths.services import authoring as authoring_service
from abridgeai.features.career_paths.services import enrollment as enrollment_service
from abridgeai.features.career_paths.services import readiness as readiness_service

management_router = APIRouter(prefix="/management/career-paths", tags=["career-paths-authoring"])
teacher_router = APIRouter(prefix="/teacher/career-paths", tags=["career-paths-authoring"])


# Codes are named once and reused by BOTH the route dependency and the
# per-resource org check, so the two can never drift apart. The dependency
# answers "does this principal hold the code anywhere"; the org check answers
# "was it granted for the organization that owns this path". Passing different
# code sets to the two would make the second check pass on the wrong grant.
_PATH_MANAGE_CODES = ("course.create", "course.update", "system.administer")
_PATH_READ_CODES = ("course.read", "system.administer")
_PATH_PUBLISH_CODES = ("course.publish", "system.administer")
_PATH_DELETE_CODES = ("course.delete", "system.administer")
_PATH_ENROLL_CODES = ("course.enrollment.create", "system.administer")
_PATH_UNENROLL_CODES = ("course.enrollment.remove", "system.administer")
_PATH_ROSTER_READ_CODES = (
    "course.enrollment.read",
    "progress.read.cohort",
    "system.administer",
)

_REQUIRE_PATH_MANAGE = require_any_permission(*_PATH_MANAGE_CODES)
_REQUIRE_PATH_READ = require_any_permission(*_PATH_READ_CODES)
_REQUIRE_PATH_PUBLISH = require_any_permission(*_PATH_PUBLISH_CODES)
_REQUIRE_PATH_DELETE = require_any_permission(*_PATH_DELETE_CODES)
_REQUIRE_PATH_ENROLL = require_any_permission(*_PATH_ENROLL_CODES)
_REQUIRE_PATH_UNENROLL = require_any_permission(*_PATH_UNENROLL_CODES)
_REQUIRE_PATH_ROSTER_READ = require_any_permission(*_PATH_ROSTER_READ_CODES)
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
    db: AsyncSession,
    current_user: CurrentUser,
    career_path_id: UUID,
    permissions: tuple[str, ...] = _PATH_MANAGE_CODES,
) -> None:
    """Require ``permissions`` to be granted FOR the path's organization.

    Every ``require_permission`` dependency in this module resolves against
    the caller's FLAT permission set, computed without regard to ``scope_kind``
    (see ``access_control/api/public.py::_ACTIVE_PERMISSIONS_SQL``). A role
    granted to a manager at ``scope_kind='organization'`` for org B yields the
    same codes as a global grant, so the dependency alone cannot keep that
    manager out of org A. Only a per-resource check can, and ``career_paths``
    is org-owned.

    ``permissions`` defaults to the manage set and every caller passes the same
    codes its own dependency declares, so the two questions line up: "do you
    hold this code at all" then "do you hold it *here*".

    This started life as an FR-2.6 guard on the two roster/readiness reads.
    It now applies to every endpoint that resolves a path by id — read, mutate,
    publish, archive, enrol. Without it a manager in any org could rename,
    re-order, publish or archive another org's career path and enrol students
    into it.

    404 rather than 403, so the endpoint cannot be used to discover which
    paths another organization owns.
    """
    path = await authoring_service.get_career_path(db, career_path_id)
    await require_org_access(
        db,
        current_user,
        path.organization_id,
        resource="career_path",
        resource_id=career_path_id,
        permissions=permissions,
    )


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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: UUID | None = None,
    include_archived: bool = False,
) -> list[CareerPathAuthoring]:
    """List career paths for an organization.

    Read-only: gated on ``course.read`` (HODs can view the pathway
    catalogue) rather than the manage set. ``organization_id`` is
    OPTIONAL: when omitted the caller's primary
    org is resolved from the bearer token, matching the contract used by
    POST. Managers without a primary org and no override get a 400 instead
    of a confusing empty list.

    An explicit value is honoured only for an org the caller belongs to, or
    for ``system.administer``. It used to be honoured unconditionally "so
    platform admins can list any org" — but the guard on this route is a flat
    permission check, so the same query parameter let a manager in any org
    enumerate another org's career paths by passing its id.
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
    else:
        await require_org_access(
            db,
            current_user,
            organization_id,
            resource="organization",
            resource_id=organization_id,
            permissions=_PATH_READ_CODES,
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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathAuthoring:
    try:
        await _ensure_caller_in_path_org(
            db, current_user, career_path_id, _PATH_READ_CODES
        )
        return await authoring_service.get_career_path(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@management_router.get(
    "/{career_path_id}/impact",
    response_model=CareerPathImpactRead,
)
async def get_path_impact(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathImpactRead:
    """Blast radius of editing this path (Gap 3 §2.1).

    Who is walking the path right now, per stage — call this BEFORE a
    mutation on a published path so the edit is informed, not silent.
    """
    try:
        await _ensure_caller_in_path_org(
            db, current_user, career_path_id, _PATH_READ_CODES
        )
        return await authoring_service.get_path_impact(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@management_router.get(
    "/{career_path_id}/versions",
    response_model=list[CareerPathVersionRead],
)
async def list_path_versions(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathVersionRead]:
    """Versions of the path, newest first (Gap 3). A published version is
    frozen; the draft (if any) is what manager edits land on."""
    try:
        await _ensure_caller_in_path_org(
            db, current_user, career_path_id, _PATH_READ_CODES
        )
        return await authoring_service.list_path_versions(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@management_router.post(
    "/{career_path_id}/versions",
    response_model=CareerPathVersionRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_path_version(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathVersionRead:
    """Copy-on-write fork (Gap 3 D2a explicit): clone the latest published
    version into a new DRAFT. Subsequent stage/course edits land on the
    draft; publishing it freezes it and leaves existing enrollments pinned
    to their own versions."""
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        version = await authoring_service.create_path_version(
            db, career_path_id, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return CareerPathVersionRead.model_validate(version)


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
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathCourseAuthoring]:
    try:
        await _ensure_caller_in_path_org(
            db, current_user, career_path_id, _PATH_READ_CODES
        )
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
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        result = await authoring_service.add_course_to_path(
            db,
            career_path_id,
            payload.course_id,
            stage_id=payload.stage_id,
            position=payload.position,
            is_required=payload.is_required,
            satisfied_by=payload.satisfied_by,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return result


@management_router.get(
    "/{career_path_id}/course-candidates",
    response_model=list[CareerPathCourseCandidate],
)
async def list_course_candidates(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathCourseCandidate]:
    """Full org course catalogue (ANY status) for the attach-to-path picker.

    The learner ``/courses`` endpoint returns only published courses, but a
    draft path may hold draft/archived courses — the publish gate re-checks
    every link. This returns the path's whole organization so the manager can
    build the skeleton before courses are published.
    """
    try:
        await _ensure_caller_in_path_org(
            db, current_user, career_path_id, _PATH_READ_CODES
        )
        return await authoring_service.list_course_candidates(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


# --- stages ------------------------------------------------------------


@management_router.get(
    "/{career_path_id}/stages",
    response_model=list[CareerPathStageAuthoring],
)
async def list_path_stages(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathStageAuthoring]:
    try:
        await _ensure_caller_in_path_org(
            db, current_user, career_path_id, _PATH_READ_CODES
        )
        return await authoring_service.list_path_stages(db, career_path_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@management_router.post(
    "/{career_path_id}/stages",
    response_model=CareerPathStageAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_stage(
    career_path_id: UUID,
    payload: CareerPathStageCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathStageAuthoring:
    """Create a stage. An EMPTY stage is valid, including on a published path
    — "every stage has a course" is a publish-gate rule, not a mutation rule.
    """
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        result = await authoring_service.create_stage(db, career_path_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@management_router.patch(
    "/{career_path_id}/stages/{stage_id}",
    response_model=CareerPathStageAuthoring,
)
async def update_stage(
    career_path_id: UUID,
    stage_id: UUID,
    payload: CareerPathStageUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathStageAuthoring:
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        result = await authoring_service.update_stage(
            db, career_path_id, stage_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@management_router.delete(
    "/{career_path_id}/stages/{stage_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_stage(
    career_path_id: UUID,
    stage_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a stage. 409 ``stage_in_use`` when it holds courses OR when any
    student has latched progress against it (deleting that would move their
    progress bar without them doing anything)."""
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        await authoring_service.delete_stage(db, career_path_id, stage_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()


@management_router.put(
    "/{career_path_id}/stages/reorder",
    response_model=CareerPathStageReorderResult,
)
async def reorder_stages(
    career_path_id: UUID,
    payload: CareerPathStageReorder,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathStageReorderResult:
    """Reorder stages. Returns WARNINGS rather than rewriting unlock policy —
    moving a non-``always`` stage into position 1 silently unlocks it, and
    moving position 1 out can re-lock a stage students are working in."""
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        result = await authoring_service.reorder_stages(
            db, career_path_id, payload.stage_ids, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@management_router.put(
    "/{career_path_id}/courses/{course_id}/stage",
    response_model=list[CareerPathCourseAuthoring],
)
async def move_course_to_stage(
    career_path_id: UUID,
    course_id: UUID,
    payload: CareerPathCourseMove,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathCourseAuthoring]:
    """Move a course between stages (or reposition within one)."""
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        result = await authoring_service.move_course_to_stage(
            db,
            career_path_id,
            course_id,
            stage_id=payload.stage_id,
            position=payload.position,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@management_router.patch(
    "/{career_path_id}/courses/{course_id}",
    response_model=list[CareerPathCourseAuthoring],
)
async def update_course_in_path(
    career_path_id: UUID,
    course_id: UUID,
    payload: CareerPathCoursePatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_PATH_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerPathCourseAuthoring]:
    """Patch an attached course's policy flags (required / satisfied-by).

    Separate from the move + reorder routes because those mutate whole
    ``(stage_id, position)`` sequences while this touches one row. Flipping a
    course to required re-runs the stage integrity check, so a change that
    would push ``min_optional_to_complete`` above the remaining optional
    count is rejected instead of leaving an uncompletable stage.
    """
    try:
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
        result = await authoring_service.update_path_course(
            db,
            career_path_id,
            course_id,
            is_required=payload.is_required,
            satisfied_by=payload.satisfied_by,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
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
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
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
        await _ensure_caller_in_path_org(db, current_user, career_path_id)
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
    del career_path_id, payload, current_user, db
    raise _conflict(
        "direct_career_path_enrollment_disabled: enroll the student in a Learning Program"
    )


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
    del career_path_id, student_id, current_user, db
    raise _conflict(
        "direct_career_path_unenrollment_disabled: withdraw the Learning Program enrollment"
    )


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
        await _ensure_caller_in_path_org(db, current_user, career_path_id, _PATH_PUBLISH_CODES)
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
        await _ensure_caller_in_path_org(db, current_user, career_path_id, _PATH_DELETE_CODES)
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
    await _ensure_caller_in_path_org(db, current_user, career_path_id, _PATH_ROSTER_READ_CODES)
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
    await _ensure_caller_in_path_org(db, current_user, career_path_id, _PATH_ROSTER_READ_CODES)
    return await readiness_service.get_path_readiness_overview(db, career_path_id)


__all__ = ["management_router", "teacher_router"]
