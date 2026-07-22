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

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import CurrentUser, get_current_user, utcnow
from abridgeai.features.quizzes.queries.published import (
    QuizClosed,
    QuizNotYetOpen,
)
from abridgeai.features.quizzes.schemas import (
    QuizAttemptProgressRead,
    QuizAttemptRead,
    QuizAttemptReviewRead,
    QuizAttemptStart,
    QuizPublic,
)
from abridgeai.features.quizzes.services import taking as taking_service
from abridgeai.features.quizzes.services.taking import (
    AllCardsInCooldownError,
    CooldownActive,
    MaxAttemptsReached,
    QuizClosed,
    QuizNotYetOpen,
)
from abridgeai.features.spaced_repetition.api.public import (
    dispatch_remediation_for_card_failure,
)

router = APIRouter(tags=["quizzes-learner"])

_logger = get_logger(__name__)


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


# ── Integrity (proctoring) ingest ────────────────────────────────────────
# Mirrors the interview integrity contract but kept quiz-local to honour the
# cross-feature independence rule. The literals + 50-event cap match the
# ``assessment_integrity_events`` CHECK constraints (assessment_kind='quiz').
_QUIZ_INTEGRITY_MAX_BATCH = 50

QuizIntegrityEventType = Literal[
    "focus_lost",
    "tab_switch",
    "fullscreen_exit",
    "warning_issued",
    "reconnect",
    "disconnect",
]
QuizIntegritySeverity = Literal["info", "warning", "critical"]


class QuizIntegrityEventItem(BaseModel):
    event_type: QuizIntegrityEventType
    severity: QuizIntegritySeverity = "info"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QuizIntegrityEventBatchRequest(BaseModel):
    events: list[QuizIntegrityEventItem] = Field(min_length=1, max_length=_QUIZ_INTEGRITY_MAX_BATCH)


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
    response_model=QuizAttemptProgressRead,
    status_code=status.HTTP_201_CREATED,
)
async def start_attempt(
    quiz_id: UUID,
    payload: QuizAttemptStart,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptProgressRead:
    """Create a :class:`QuizAttempt` and return the no-leak take payload.

    Response shares the :class:`QuizAttemptProgressRead` shape with
    ``GET /attempts/{id}/progress`` (``answers=[]`` here since nothing is
    saved yet) so the client gets ``attempt_id`` back immediately and can
    reuse one hydration path for both "start" and "resume".

    Maps :class:`AllCardsInCooldownError` (raised when every question in
    the quiz is still in SR cooldown — thesis UC-LEARN-01 Alt 1a) to
    HTTP 429 with a ``Retry-After`` header counting down to the earliest
    card's ``due_at``.

    FR-4.3 quiz-level retake policy maps the same way:
    :class:`CooldownActive` (inside ``cooldown_hours`` of the last
    submission) → 429 + ``Retry-After``; :class:`MaxAttemptsReached`
    (``max_attempts`` used up, or a retake with
    ``allow_retakes=False``) → 409.
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
    except CooldownActive as exc:
        retry_after_seconds = max(0, int((exc.retry_after - utcnow()).total_seconds()))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "reason": "quiz_cooldown_active",
                "retry_available_at": exc.retry_after.isoformat(),
            },
            headers={"Retry-After": str(retry_after_seconds)},
        ) from exc
    except MaxAttemptsReached as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "reason": "max_attempts_reached",
                "max_attempts": exc.max_attempts,
            },
        ) from exc
    except QuizNotYetOpen as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "reason": "quiz_not_yet_open",
                "available_from": exc.available_from.isoformat(),
            },
        ) from exc
    except QuizClosed as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "reason": "quiz_closed",
                "available_until": exc.available_until.isoformat(),
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
    """Record one answer for an in-flight attempt.

    The service fires the SM-2 review inside the same transaction
    (FR-4.4); any :class:`CardFailedEvent` comes back on
    ``CardReviewResult.pending_events`` and is dispatched **after**
    commit (caller-dispatches-after-commit pattern, T7.5.10 BUG-2) so a
    rolled-back review can never trigger a ghost notification.
    Remediation failures are logged and never surface to the student.
    """
    try:
        answer, review = await taking_service.answer_attempt(db, attempt_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz_attempt", attempt_id) from exc
    await db.commit()

    for event in review.pending_events if review else []:
        try:
            await dispatch_remediation_for_card_failure(
                db,
                student_id=event.student_id,
                question_id=event.question_id,
                quiz_attempt_id=event.quiz_attempt_id,
            )
        except Exception:  # noqa: BLE001 — side-effect must not fail the answer
            _logger.exception(
                "remediation_dispatch_failed",
                question_id=str(event.question_id),
                attempt_id=str(attempt_id),
            )
    return QuizAttemptAnswerRead.model_validate(answer)


@router.post(
    "/attempts/{attempt_id}/integrity-events",
    status_code=status.HTTP_202_ACCEPTED,
)
async def record_quiz_integrity_events(
    attempt_id: UUID,
    payload: QuizIntegrityEventBatchRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, int]:
    """Best-effort ingest of browser integrity signals for a live attempt.

    Ownership is enforced (attempt must belong to the caller). Events are
    recorded only while the attempt is ``in_progress`` — late events for a
    submitted/graded/abandoned attempt are silently dropped so this never
    blocks the take. Append-only; post-attempt / teacher review only, never
    surfaced to the student.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        AssessmentIntegrityEvent,
    )
    from abridgeai.features.quizzes.models import QuizAttempt  # noqa: PLC0415

    attempt = await db.get(QuizAttempt, attempt_id)
    if attempt is None or attempt.student_id != current_user.user_id:
        raise _not_found("quiz_attempt", attempt_id)
    if attempt.status != "in_progress":
        return {"accepted": 0}

    for item in payload.events:
        db.add(
            AssessmentIntegrityEvent(
                assessment_kind="quiz",
                quiz_attempt_id=attempt_id,
                student_id=current_user.user_id,
                event_type=item.event_type,
                severity=item.severity,
                metadata_json=dict(item.metadata),
            )
        )
    await db.commit()
    return {"accepted": len(payload.events)}


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
    "/attempts/{attempt_id}/progress",
    response_model=QuizAttemptProgressRead,
)
async def get_attempt_progress(
    attempt_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptProgressRead:
    """Resume payload for an in-progress attempt.

    Lets the client rehydrate per-question state after an interruption
    (power-off, reload, critical notification) instead of wiping progress.
    Returns 404 when the attempt is missing, belongs to a different
    student, or is no longer in flight (submitted / graded / abandoned).

    No-leak: unlike ``/review`` this fires WHILE the attempt is open, so
    the payload carries only the student's own saved inputs — never
    ``is_correct`` / ``points_awarded`` (see ``QuizAttemptProgressAnswer``).
    """
    progress = await taking_service.get_attempt_progress(
        db, attempt_id=attempt_id, actor=current_user
    )
    if progress is None:
        raise _not_found("quiz_attempt", attempt_id)
    return progress


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
