from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.progress.schemas.public import (
    LessonProgressPublic,
    MaterialEngagementCreate,
    MaterialEngagementPublic,
    MyCourseProgressSummary,
)
from abridgeai.features.progress.services import reporting, tracking

router = APIRouter(tags=["progress-learner"])


def _not_found(resource: str, ident: str | UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(ident)},
    )


@router.get(
    "/me/progress/lessons/{lesson_id}",
    response_model=LessonProgressPublic,
)
async def get_my_lesson_progress(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonProgressPublic:
    view = await reporting.get_my_lesson_progress_view(
        db, user_id=current_user.user_id, lesson_id=lesson_id
    )
    if view is None:
        raise _not_found("lesson_progress", lesson_id)
    return view


@router.get(
    "/me/progress/courses/{course_id}",
    response_model=MyCourseProgressSummary,
)
async def get_my_course_progress(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MyCourseProgressSummary:
    return await reporting.get_my_course_progress_summary(
        db, user_id=current_user.user_id, course_id=course_id
    )


@router.post(
    "/me/progress/material-engagement",
    response_model=MaterialEngagementPublic,
    status_code=status.HTTP_201_CREATED,
)
async def post_material_engagement(
    payload: MaterialEngagementCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialEngagementPublic:
    try:
        result = await tracking.record_material_engagement(
            db, user_id=current_user.user_id, payload=payload
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "message": str(exc)},
        ) from exc
    await db.commit()
    return result


__all__ = ["router"]
