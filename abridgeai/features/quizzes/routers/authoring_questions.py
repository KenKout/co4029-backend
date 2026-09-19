# ruff: noqa
"""Split teacher quiz authoring routes.

The modules share the historical ``authoring.router`` object so the public
router import and dependency overrides remain unchanged.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.schemas import (
    CourseAssessmentSummaryRead,
    FeedbackBandIn,
    FeedbackBandRead,
    ManualGradeIn,
    ManualGradeRead,
    NeedsGradingRow,
    QuestionBankEntry,
    QuestionBankImportRequest,
    QuestionBankPage,
    QuizAttemptIntegrityEvent,
    QuizAttemptReviewOption,
    QuizAttemptReviewQuestion,
    QuizAttemptTeacherPage,
    QuizAttemptTeacherRead,
    QuizAttemptTeacherReview,
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizGenerationProgress,
    QuizGenerationRequest,
    QuizGenerationRunRead,
    QuizGradeRow,
    QuizOptionDistribution,
    QuizOverrideIn,
    QuizOverrideRead,
    QuizPerStudentRow,
    QuizQuestionAuthoring,
    QuizQuestionBreakdown,
    QuizResultsRead,
    QuizResultsSummary,
    QuizScoreBucket,
    RegradeRunRead,
    RegradeScopeIn,
)
from abridgeai.features.quizzes.routers.authoring import *  # noqa: F403
from abridgeai.features.quizzes.routers.authoring import (
    _QUIZ_RESULT_PATTERN,
    _REQUIRE_COURSE_UPDATE,
    _REQUIRE_QUESTION,
    _REQUIRE_QUIZ,
    _StudentAvatarTarget,
    _attempt_teacher_view,
    _bad_request,
    _conflict,
    _not_found,
    _resolve_student_contacts,
    _resolve_student_names,
)
from abridgeai.features.quizzes.services import authoring as authoring_service
from abridgeai.features.quizzes.services import question_bank as question_bank_service
from abridgeai.features.quizzes.services.authoring import QuizPublishValidationError
from abridgeai.features.quizzes.services.publish_gate import QuizApprovalRequiredError
from abridgeai.infrastructure.s3 import create_stream_url

@router.post(
    "/quizzes/{quiz_id}/questions",
    response_model=QuizQuestionAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_question(
    quiz_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionAuthoring:
    """Manual (non-AI) question creation — MCQ flavours validated up front."""
    create_payload = _AttrShim(payload)
    try:
        question = await authoring_service.create_question(
            db, quiz_id, create_payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await _attach_question_options(db, question)
    await db.commit()
    if question.learning_outcome_id:
        positions = await _resolve_outcome_positions(db, {question.learning_outcome_id})
        _fill_outcome_positions([question], positions)
    return QuizQuestionAuthoring.model_validate(question)


@router.patch(
    "/quizzes/{quiz_id}/questions/{question_id}",
    response_model=QuizQuestionAuthoring,
)
async def update_question(
    quiz_id: UUID,
    question_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionAuthoring:
    del quiz_id
    try:
        question = await authoring_service.update_question(
            db, question_id, _AttrShim(payload), current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await _attach_question_options(db, question)
    await db.commit()
    if question.learning_outcome_id:
        positions = await _resolve_outcome_positions(db, {question.learning_outcome_id})
        _fill_outcome_positions([question], positions)
    return QuizQuestionAuthoring.model_validate(question)


@router.delete(
    "/quizzes/{quiz_id}/questions/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_question(
    quiz_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a question + repack sibling positions."""
    del quiz_id
    try:
        await authoring_service.delete_question(db, question_id, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()


@router.post(
    "/quizzes/{quiz_id}/questions/{question_id}/duplicate",
    response_model=QuizQuestionAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def duplicate_question(
    quiz_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionAuthoring:
    """Clone a question in place at the end of its own quiz.

    The copy is always ``review_status='pending'`` (unpublished) regardless of
    the source's state, so a duplicate re-enters the review queue rather than
    inheriting approval it was never granted.
    """
    del quiz_id
    try:
        clone = await question_bank_service.duplicate_question(
            db, question_id=question_id, actor=current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    await _attach_question_options(db, clone)
    if clone.learning_outcome_id:
        positions = await _resolve_outcome_positions(db, {clone.learning_outcome_id})
        _fill_outcome_positions([clone], positions)
    return QuizQuestionAuthoring.model_validate(clone)


@router.post(
    "/quizzes/{quiz_id}/questions/{question_id}/regenerate",
    response_model=QuizGenerationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_question(
    quiz_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> QuizGenerationRunRead:
    """Create a per-question regeneration run + enqueue ARQ.

    The service commits inline so the worker can read the run row.
    """
    try:
        run = await authoring_service.regenerate_question(
            db, question_id, current_user, arq_pool=arq_pool
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    return _generation_run_view(run, quiz_id)


@router.get(
    "/courses/{course_id}/question-bank",
    response_model=QuestionBankPage,
)
async def list_question_bank(  # noqa: PLR0913 -- filters mirror service signature
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    module_id: UUID | None = None,
    lesson_id: UUID | None = None,
    question_type: str | None = None,
    bloom_level: str | None = None,
    difficulty: str | None = None,
    review_status: str | None = "approved",
    search: str | None = None,
    exclude_quiz_id: UUID | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> QuestionBankPage:
    """Browse authored questions across the course for cross-quiz reuse."""
    del current_user  # permission already enforced by Depends
    try:
        page = await question_bank_service.list_bank_entries(
            db,
            course_id=course_id,
            module_id=module_id,
            lesson_id=lesson_id,
            question_type=question_type,
            bloom_level=bloom_level,
            difficulty=difficulty,
            review_status=review_status if review_status else None,
            search=search,
            exclude_quiz_id=exclude_quiz_id,
            limit=limit,
            cursor=cursor,
        )
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return QuestionBankPage(
        items=[QuestionBankEntry.model_validate(row) for row in page.items],
        next_cursor=page.next_cursor,
    )


@router.post(
    "/quizzes/{quiz_id}/questions/import",
    response_model=list[QuizQuestionAuthoring],
    status_code=status.HTTP_201_CREATED,
)
async def import_questions_from_bank(
    quiz_id: UUID,
    payload: QuestionBankImportRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizQuestionAuthoring]:
    """Clone bank questions into ``quiz_id``.

    Each clone has a fresh id, ``review_status='pending'``, and an
    ``imported_from_question_id`` back-pointer. Options are cloned in
    place. Source questions must live in the same course as the target.
    """
    try:
        cloned = await question_bank_service.import_questions(
            db,
            target_quiz_id=quiz_id,
            source_question_ids=payload.source_question_ids,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", quiz_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return [QuizQuestionAuthoring.model_validate(question) for question in cloned]


class _AttrShim:
    """Adapt a dict to the model and attribute interface used by services."""

    # Ignore retired settings still sent by cached clients.
    _RETIRED_KEYS = frozenset({"browser_security"})

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = {k: v for k, v in data.items() if k not in self._RETIRED_KEYS}

    def model_dump(
        self,
        exclude_unset: bool = False,
        mode: str | None = None,
        exclude: set[str] | None = None,
        include: set[str] | None = None,
    ) -> dict[str, Any]:
        del exclude_unset, mode
        data = dict(self._data)
        if include is not None:
            data = {k: v for k, v in data.items() if k in include}
        if exclude:
            data = {k: v for k, v in data.items() if k not in exclude}
        return data

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401  -- shim returns whatever the dict holds
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            value = self._data[name]
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [_AttrShim(item) for item in value]
            if isinstance(value, dict):
                return _AttrShim(value)
            return value
        return None


def _generation_run_view(run: GenerationRun, quiz_id: UUID) -> QuizGenerationRunRead:
    config = run.config_json or {}
    failure = config.get("failure") if isinstance(config, dict) else None
    error_message = (
        str(failure.get("message")) if isinstance(failure, dict) and "message" in failure else None
    )
    # Live-progress projection (migration 0035). ``progress_json`` is
    progress = None
    raw_progress = getattr(run, "progress_json", None)
    if isinstance(raw_progress, dict) and raw_progress:
        try:
            progress = QuizGenerationProgress.model_validate(raw_progress)
        except ValidationError:
            progress = None
    return QuizGenerationRunRead(
        id=run.id,
        quiz_id=quiz_id,
        status=run.status,
        started_at=run.started_at or run.created_at,
        completed_at=run.finished_at,
        error_message=error_message,
        pipeline_run_id=None,
        progress=progress,
    )


async def _attach_question_options(db: AsyncSession, question: QuizQuestion) -> None:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.quizzes.models import QuizQuestionOption  # noqa: PLC0415

    options = list(
        (
            await db.execute(
                select(QuizQuestionOption)
                .where(QuizQuestionOption.question_id == question.id)
                .order_by(QuizQuestionOption.position)
            )
        )
        .scalars()
        .all()
    )
    question.options = options  # type: ignore[attr-defined]


async def _resolve_outcome_positions(
    db: AsyncSession, outcome_ids: set[UUID]
) -> dict[UUID, tuple[int, str]]:
    """Resolve outcome IDs to their sibling position and dotted code."""
    if not outcome_ids:
        return {}
    from sqlalchemy import text as _text  # noqa: PLC0415

    rows = (
        await db.execute(
            _text(
                """
                WITH RECURSIVE coded AS (
                    SELECT id, parent_id, position, position::text AS code
                    FROM course_learning_outcomes
                    WHERE parent_id IS NULL AND deleted_at IS NULL
                    UNION ALL
                    SELECT c.id, c.parent_id, c.position,
                           coded.code || '.' || c.position::text
                    FROM course_learning_outcomes c
                    JOIN coded ON c.parent_id = coded.id
                    WHERE c.deleted_at IS NULL
                )
                SELECT id, position, code FROM coded WHERE id = ANY(:ids)
                """
            ),
            {"ids": list(outcome_ids)},
        )
    ).all()
    return {row[0]: (row[1], row[2]) for row in rows}


def _fill_outcome_positions(
    questions: list[QuizQuestion], positions: dict[UUID, tuple[int, str]]
) -> None:
    """Attach projection-only outcome positions and codes to questions."""
    for question in questions:
        lo_id = question.learning_outcome_id
        resolved = positions.get(lo_id) if lo_id is not None else None
        question.outcome_position = resolved[0] if resolved else None  # type: ignore[attr-defined]
        question.outcome_code = resolved[1] if resolved else None  # type: ignore[attr-defined]
