from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.enrollments.schemas import EnrollmentRead
from abridgeai.features.enrollments.services import student_view as student_service

me_enrollments_router = APIRouter(prefix="/me/enrollments", tags=["enrollments-learner"])


@me_enrollments_router.get("", response_model=list[EnrollmentRead])
async def list_my_enrollments(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[EnrollmentRead]:
    return await student_service.list_my_enrollments(db, current_user.user_id)


@me_enrollments_router.get("/{course_id}", response_model=EnrollmentRead)
async def get_my_enrollment_detail(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EnrollmentRead:
    enrollment = await student_service.get_my_enrollment_status(
        db, current_user.user_id, course_id
    )
    if enrollment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "resource": "enrollment",
                "course_id": str(course_id),
            },
        )
    return enrollment


__all__ = ["me_enrollments_router"]
