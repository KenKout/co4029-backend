"""HTTP endpoints for the course-scoped curated Quiz Question Bank."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_course_permission
from abridgeai.features.quizzes.routers._deps import require_quiz_authoring_access
from abridgeai.features.quizzes.schemas import (
    QuizQuestionAuthoring,
    QuizQuestionBankCopyRequest,
    QuizQuestionBankImportRequest,
    QuizQuestionBankItemCreate,
    QuizQuestionBankItemRead,
    QuizQuestionBankItemUpdate,
    QuizQuestionBankPage,
)
from abridgeai.features.quizzes.services import curated_question_bank as curated_bank_service

router = APIRouter(tags=["quizzes-authoring"])
_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_QUIZ = require_quiz_authoring_access()


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": message},
    )


def _conflict(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": message},
    )


class _QuizBankStatusBody(BaseModel):
    status: str


@router.get(
    "/courses/{course_id}/quiz-question-bank",
    response_model=QuizQuestionBankPage,
)
async def list_curated_quiz_question_bank(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    bank_status: str | None = None,
    question_type: str | None = None,
    bloom_level: str | None = None,
    difficulty: str | None = None,
    search: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> QuizQuestionBankPage:
    del current_user
    try:
        page = await curated_bank_service.list_curated_bank_items(
            db,
            course_id=course_id,
            status=bank_status,
            question_type=question_type,
            bloom_level=bloom_level,
            difficulty=difficulty,
            search=search,
            limit=limit,
            cursor=cursor,
        )
    except NotFoundError as exc:
        raise _not_found("course", course_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return QuizQuestionBankPage(
        items=[QuizQuestionBankItemRead.model_validate(item) for item in page.items],
        next_cursor=page.next_cursor,
    )


@router.post(
    "/courses/{course_id}/quiz-question-bank",
    response_model=QuizQuestionBankItemRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_curated_quiz_question_bank_item(
    course_id: UUID,
    payload: QuizQuestionBankItemCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionBankItemRead:
    try:
        item = await curated_bank_service.create_curated_bank_item(
            db, course_id=course_id, payload=payload, actor=current_user
        )
    except NotFoundError as exc:
        raise _not_found("course", course_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return QuizQuestionBankItemRead.model_validate(item)


@router.post(
    "/courses/{course_id}/quiz-question-bank/from-questions",
    response_model=list[QuizQuestionBankItemRead],
    status_code=status.HTTP_201_CREATED,
)
async def copy_quiz_questions_to_curated_bank(
    course_id: UUID,
    payload: QuizQuestionBankCopyRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizQuestionBankItemRead]:
    try:
        items = await curated_bank_service.copy_questions_to_curated_bank(
            db,
            course_id=course_id,
            question_ids=payload.question_ids,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", course_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return [QuizQuestionBankItemRead.model_validate(item) for item in items]


@router.patch(
    "/courses/{course_id}/quiz-question-bank/{item_id}",
    response_model=QuizQuestionBankItemRead,
)
async def update_curated_quiz_question_bank_item(
    course_id: UUID,
    item_id: UUID,
    payload: QuizQuestionBankItemUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionBankItemRead:
    try:
        item = await curated_bank_service.update_curated_bank_item(
            db,
            course_id=course_id,
            item_id=item_id,
            payload=payload,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question_bank_item", item_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return QuizQuestionBankItemRead.model_validate(item)


@router.post(
    "/courses/{course_id}/quiz-question-bank/{item_id}/status",
    response_model=QuizQuestionBankItemRead,
)
async def set_curated_quiz_question_bank_item_status(
    course_id: UUID,
    item_id: UUID,
    payload: _QuizBankStatusBody,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionBankItemRead:
    try:
        item = await curated_bank_service.set_curated_bank_item_status(
            db,
            course_id=course_id,
            item_id=item_id,
            status=payload.status,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question_bank_item", item_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return QuizQuestionBankItemRead.model_validate(item)


@router.delete(
    "/courses/{course_id}/quiz-question-bank/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_curated_quiz_question_bank_item(
    course_id: UUID,
    item_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    try:
        await curated_bank_service.delete_curated_bank_item(
            db, course_id=course_id, item_id=item_id, actor=current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question_bank_item", item_id) from exc
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/quizzes/{quiz_id}/questions/import-bank",
    response_model=list[QuizQuestionAuthoring],
    status_code=status.HTTP_201_CREATED,
)
async def import_curated_quiz_question_bank_items(
    quiz_id: UUID,
    payload: QuizQuestionBankImportRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizQuestionAuthoring]:
    try:
        created = await curated_bank_service.import_curated_bank_items(
            db,
            target_quiz_id=quiz_id,
            item_ids=payload.item_ids,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question_bank_item", quiz_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return [QuizQuestionAuthoring.model_validate(question) for question in created]
