"""Quizzes learner router (T5.14).

Six endpoints under split path roots ``/quizzes`` + ``/attempts`` +
``/me`` (legacy parity — no single common prefix). Composes
:mod:`features.quizzes.services.taking` for student attempt lifecycle
and :mod:`features.quizzes.queries.published` for the published-quiz
fetch.

Security invariant (plan §5398, T5.2)
-------------------------------------
Every learner-facing response serializes through
:class:`QuizPublic` / :class:`QuizQuestionPublic` /
:class:`QuizQuestionOptionPublic` which deliberately drop the
``is_correct`` flag (and any other answer-correctness fields) so the
client cannot peek at correctness during the take. The grading service
is the only consumer of the authoring projection.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user, utcnow
from abridgeai.features.quizzes.schemas import (
    QuizAttemptRead,
    QuizAttemptReviewRead,
    QuizAttemptStart,
    QuizForTakingPublic,
    QuizPublic,
)
from abridgeai.features.quizzes.services import taking as taking_service
from abridgeai.features.quizzes.services.taking import (
    AllCardsInCooldownError,
    InterviewPassRequiredError,
)

router = APIRouter(tags=["quizzes-learner"])


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


class QuizAttemptAnswerInput(BaseModel):
    """Body for ``POST /attempts/{attempt_id}/answers``.

    Mirrors the legacy ``QuizAttemptAnswerCreate`` shape: a single
    answer for one question. Server computes ``is_correct`` from the
    selected option's truth flag — the request payload's correctness
    field (if any) is intentionally absent from this DTO so a malicious
    client cannot self-grade.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: UUID
    selected_option_id: UUID | None = None
    answer_text: str | None = None
    t_actual_ms: int | None = Field(default=None, ge=0)
    hint_used: bool = False


class QuizAttemptAnswerRead(BaseModel):
    """Response shape for the per-answer write endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    attempt_id: UUID
    question_id: UUID
    selected_option_id: UUID | None = None


@router.get("/quizzes/{quiz_id}", response_model=QuizPublic)
async def get_published_quiz(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizPublic:
    """Public projection of one published quiz (no ``is_correct`` leak)."""
    del current_user
    quiz = await taking_service.get_published_quiz(db, quiz_id)
    if quiz is None:
        raise _not_found("quiz", quiz_id)
    return QuizPublic.model_validate(quiz)


@router.post(
    "/quizzes/{quiz_id}/attempts",
    response_model=QuizForTakingPublic,
    status_code=status.HTTP_201_CREATED,
)
async def start_attempt(
    quiz_id: UUID,
    payload: QuizAttemptStart,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizForTakingPublic:
    """Create a :class:`QuizAttempt` and return the no-leak take payload.

    Maps :class:`AllCardsInCooldownError` (raised when every question in
    the quiz is still in SR cooldown — thesis UC-LEARN-01 Alt 1a) to
    HTTP 429 with a ``Retry-After`` header counting down to the earliest
    card's ``due_at``.
    """
    if payload.quiz_id != quiz_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "quiz_id_mismatch"},
        )
    try:
        _, take_payload = await taking_service.start_attempt(
            db,
            quiz_id,
            current_user,
            idempotency_key=payload.idempotency_key,
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except InterviewPassRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "interview_pass_required",
                "module_id": str(exc.module_id),
                "interview_config_id": str(exc.interview_config_id),
            },
        ) from exc
    except AllCardsInCooldownError as exc:
        retry_after_seconds = max(0, int((exc.retry_available_at - utcnow()).total_seconds()))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "reason": "all_cards_in_cooldown",
                "retry_available_at": exc.retry_available_at.isoformat(),
                "cards_due_at": [
                    {"question_id": str(qid), "due_at": due_at.isoformat()}
                    for qid, due_at in exc.cards_due_at
                ],
            },
            headers={"Retry-After": str(retry_after_seconds)},
        ) from exc
    await db.commit()
    return take_payload


@router.post(
    "/attempts/{attempt_id}/answers",
    response_model=QuizAttemptAnswerRead,
    status_code=status.HTTP_201_CREATED,
)
async def record_answer(
    attempt_id: UUID,
    payload: QuizAttemptAnswerInput,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptAnswerRead:
    """Record one answer for an in-flight attempt."""
    try:
        answer = await taking_service.answer_attempt(db, attempt_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz_attempt", attempt_id) from exc
    await db.commit()
    return QuizAttemptAnswerRead.model_validate(answer)


@router.post("/attempts/{attempt_id}/submit", response_model=QuizAttemptRead)
async def submit_attempt(
    attempt_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptRead:
    """Grade and finalize an attempt."""
    try:
        attempt = await taking_service.submit_attempt(db, attempt_id, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz_attempt", attempt_id) from exc
    await db.commit()
    return QuizAttemptRead.model_validate(attempt)


@router.get("/attempts/{attempt_id}", response_model=QuizAttemptRead)
async def get_attempt(
    attempt_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptRead:
    """Return one attempt detail, scoped to the calling student."""
    from abridgeai.features.quizzes.models import QuizAttempt  # noqa: PLC0415

    attempt = await db.get(QuizAttempt, attempt_id)
    if attempt is None or attempt.student_id != current_user.user_id:
        raise _not_found("quiz_attempt", attempt_id)
    return QuizAttemptRead.model_validate(attempt)


@router.get(
    "/attempts/{attempt_id}/review",
    response_model=QuizAttemptReviewRead,
)
async def get_attempt_review(
    attempt_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptReviewRead:
    """Full review payload — attempt summary + per-question correctness.

    Returns 404 when the attempt is missing, belongs to a different student,
    or hasn't been submitted yet (review only opens after submission so
    correct-option flags can't leak mid-attempt).
    """
    review = await taking_service.get_attempt_review(db, attempt_id=attempt_id, actor=current_user)
    if review is None:
        raise _not_found("quiz_attempt", attempt_id)
    return review


@router.get("/me/quizzes/{quiz_id}/attempts", response_model=list[QuizAttemptRead])
async def list_my_attempts(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizAttemptRead]:
    """List the calling student's attempts on this quiz."""
    attempts = await taking_service.get_attempt_history(db, quiz_id, current_user)
    return [QuizAttemptRead.model_validate(a) for a in attempts]


__all__ = ["QuizAttemptAnswerInput", "QuizAttemptAnswerRead", "router"]
