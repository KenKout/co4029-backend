from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.learning_programs import services
from abridgeai.features.learning_programs.schemas import (
    ChangePathRequestCreate,
    ChangeRequestDecision,
    PathChangeRequestRead,
    ProgramAuthoringOptions,
    ProgramCreate,
    ProgramEnrollmentRead,
    ProgramEnrollRequest,
    ProgramRead,
    ProgramUpdate,
    ProgramVersionRead,
    ProgramWithdrawRequest,
    SelectPathRequest,
)

management_router = APIRouter(prefix="/management/learning-programs", tags=["learning-programs"])
learner_router = APIRouter(prefix="/me/learning-program-enrollments", tags=["learning-programs"])

_REQUIRE_READ = require_any_permission("learning_program.read", "learning_program.manage")
_REQUIRE_MANAGE = require_any_permission("learning_program.manage")
_REQUIRE_ENROLL = require_any_permission("learning_program.enroll")
_REQUIRE_REVIEW = require_any_permission("learning_program.switch.review")


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, detail={"error": str(exc)})
    if isinstance(exc, ForbiddenError):
        return HTTPException(status.HTTP_403_FORBIDDEN, detail={"error": str(exc)})
    return HTTPException(status.HTTP_409_CONFLICT, detail={"error": str(exc)})


@management_router.get("/options", response_model=ProgramAuthoringOptions)
async def get_authoring_options(
    actor: Annotated[CurrentUser, Depends(_REQUIRE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramAuthoringOptions:
    return await services.get_authoring_options(db, actor)


@management_router.post("", response_model=ProgramRead, status_code=status.HTTP_201_CREATED)
async def create_program(
    payload: ProgramCreate,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramRead:
    try:
        result = await services.create_program(db, payload, actor)
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.get("", response_model=list[ProgramRead])
async def list_programs(
    actor: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: UUID | None = None,
) -> list[ProgramRead]:
    if organization_id is None:
        org = await access_control_api.get_user_primary_org(db, actor.user_id)
        if org is None:
            return []
        organization_id = org.id
    return await services.list_programs(db, organization_id=organization_id, actor=actor)


@management_router.get("/{program_id}", response_model=ProgramRead)
async def get_program(
    program_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramRead:
    try:
        return await services.get_program_for_operator(db, program_id=program_id, actor=actor)
    except (NotFoundError, ForbiddenError) as exc:
        raise _http_error(exc) from exc


@management_router.get("/{program_id}/versions", response_model=list[ProgramVersionRead])
async def list_program_versions(
    program_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ProgramVersionRead]:
    try:
        return await services.list_program_versions(db, program_id=program_id, actor=actor)
    except (NotFoundError, ForbiddenError) as exc:
        raise _http_error(exc) from exc


@management_router.get("/{program_id}/versions/{version_id}", response_model=ProgramRead)
async def get_program_version(
    program_id: UUID,
    version_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramRead:
    try:
        return await services.get_program_version(
            db, program_id=program_id, version_id=version_id, actor=actor
        )
    except (NotFoundError, ForbiddenError) as exc:
        raise _http_error(exc) from exc


@management_router.patch("/{program_id}", response_model=ProgramRead)
async def update_program(
    program_id: UUID,
    payload: ProgramUpdate,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramRead:
    try:
        result = await services.update_program(
            db, program_id=program_id, payload=payload, actor=actor
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.post("/{program_id}/publish", response_model=ProgramRead)
async def publish_program(
    program_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramRead:
    try:
        result = await services.publish_program(db, program_id=program_id, actor=actor)
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.post("/{program_id}/archive", response_model=ProgramRead)
async def archive_program(
    program_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramRead:
    try:
        result = await services.archive_program(db, program_id=program_id, actor=actor)
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.get("/{program_id}/students", response_model=list[ProgramEnrollmentRead])
async def list_roster(
    program_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ProgramEnrollmentRead]:
    try:
        return await services.list_roster(db, program_id=program_id, actor=actor)
    except (NotFoundError, ForbiddenError) as exc:
        raise _http_error(exc) from exc


@management_router.post(
    "/{program_id}/students",
    response_model=list[ProgramEnrollmentRead],
    status_code=status.HTTP_201_CREATED,
)
async def enroll_students(
    program_id: UUID,
    payload: ProgramEnrollRequest,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_ENROLL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ProgramEnrollmentRead]:
    try:
        result = await services.enroll_students(
            db, program_id=program_id, student_ids=payload.student_ids, actor=actor
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.delete(
    "/{program_id}/students/{student_id}", response_model=ProgramEnrollmentRead
)
async def withdraw_student(
    program_id: UUID,
    student_id: UUID,
    payload: ProgramWithdrawRequest,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_ENROLL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramEnrollmentRead:
    try:
        result = await services.withdraw_student(
            db,
            program_id=program_id,
            student_id=student_id,
            reason=payload.reason,
            actor=actor,
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.get(
    "/{program_id}/path-change-requests", response_model=list[PathChangeRequestRead]
)
async def list_change_requests(
    program_id: UUID,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PathChangeRequestRead]:
    try:
        return await services.list_change_requests(db, program_id=program_id, actor=actor)
    except (NotFoundError, ForbiddenError) as exc:
        raise _http_error(exc) from exc


@management_router.post(
    "/path-change-requests/{request_id}/approve", response_model=PathChangeRequestRead
)
async def approve_change_request(
    request_id: UUID,
    payload: ChangeRequestDecision,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_REVIEW)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PathChangeRequestRead:
    try:
        result = await services.decide_change_request(
            db,
            request_id=request_id,
            approve=True,
            decision_reason=payload.reason,
            actor=actor,
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@management_router.post(
    "/path-change-requests/{request_id}/reject", response_model=PathChangeRequestRead
)
async def reject_change_request(
    request_id: UUID,
    payload: ChangeRequestDecision,
    actor: Annotated[CurrentUser, Depends(_REQUIRE_REVIEW)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PathChangeRequestRead:
    try:
        result = await services.decide_change_request(
            db,
            request_id=request_id,
            approve=False,
            decision_reason=payload.reason,
            actor=actor,
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@learner_router.get("", response_model=list[ProgramEnrollmentRead])
async def list_my_programs(
    actor: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ProgramEnrollmentRead]:
    return await services.list_my_enrollments(db, actor.user_id)


@learner_router.post("/{enrollment_id}/select-path", response_model=ProgramEnrollmentRead)
async def select_path(
    enrollment_id: UUID,
    payload: SelectPathRequest,
    actor: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProgramEnrollmentRead:
    try:
        result = await services.select_path(
            db,
            enrollment_id=enrollment_id,
            career_path_id=payload.career_path_id,
            student_id=actor.user_id,
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@learner_router.post(
    "/{enrollment_id}/path-change-requests",
    response_model=PathChangeRequestRead,
    status_code=status.HTTP_201_CREATED,
)
async def request_path_change(
    enrollment_id: UUID,
    payload: ChangePathRequestCreate,
    actor: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PathChangeRequestRead:
    try:
        result = await services.request_path_change(
            db,
            enrollment_id=enrollment_id,
            target_path_id=payload.target_career_path_id,
            reason=payload.reason,
            student_id=actor.user_id,
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


@learner_router.delete("/path-change-requests/{request_id}", response_model=PathChangeRequestRead)
async def cancel_change_request(
    request_id: UUID,
    actor: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PathChangeRequestRead:
    try:
        result = await services.cancel_change_request(
            db, request_id=request_id, student_id=actor.user_id
        )
        await db.commit()
        return result
    except (NotFoundError, ForbiddenError, ConflictError) as exc:
        raise _http_error(exc) from exc


__all__ = ["learner_router", "management_router"]
