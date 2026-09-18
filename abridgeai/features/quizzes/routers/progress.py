"""Learner progress endpoints for quiz completion summaries."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.courses.api.public import can_view_course_content
from abridgeai.features.quizzes.schemas import QuizProgressRead
from abridgeai.features.quizzes.services import learner_progress as learner_progress_service

router = APIRouter(tags=["quizzes-learner"])


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


@router.get(
    "/courses/{course_id}/quiz-progress",
    response_model=list[QuizProgressRead],
)
async def list_my_quiz_progress(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizProgressRead]:
    """Per-quiz completion state for the calling student in a course."""
    if not await can_view_course_content(
        db, user_id=current_user.user_id, course_id=course_id
    ):
        raise _not_found("course", course_id)
    rows = await learner_progress_service.list_my_quiz_progress(
        db, course_id=course_id, user_id=current_user.user_id
    )
    return [QuizProgressRead.model_validate(r) for r in rows]


__all__ = ["router"]
