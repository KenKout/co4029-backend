"""Student-side quiz taking service (T5.13).

Ports the attempt lifecycle from
``backend/app/routes/quizzes/service.py`` (legacy 465 LOC god-file):
``create_attempt`` → ``start_attempt``, ``answer_attempt``,
``submit_attempt``, plus ``get_attempt_history`` for the learner
dashboard.

Security invariant (plan §5398): :func:`start_attempt` returns
questions through the :class:`QuizQuestionPublic` schema, which
intentionally drops :attr:`QuizQuestionOption.is_correct` (and the
question's ``correct_option_id``-style hints) so the learner client
cannot peek at answer correctness during the take. The grading
service is the only consumer of the authoring projection.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
)
from abridgeai.features.quizzes.queries import authoring as authoring_queries
from abridgeai.features.quizzes.queries import published as published_queries
from abridgeai.features.quizzes.schemas.public import (
    QuizForTakingPublic,
    QuizPublic,
    QuizQuestionPublic,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_published_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz | None:
    """Pass-through to :func:`published_queries.get_published_quiz`.

    Routers cannot import queries directly (T0.4 contract); learner
    callers reach the published-quiz fetcher through this thin service
    indirection.
    """
    return await published_queries.get_published_quiz(db, quiz_id)


async def _require_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz:
    quiz = await authoring_queries.get_quiz_for_authoring(db, quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    return quiz


async def _require_attempt(db: AsyncSession, attempt_id: UUID) -> QuizAttempt:
    attempt = await db.get(QuizAttempt, attempt_id)
    if attempt is None:
        raise NotFoundError(f"Quiz attempt {attempt_id} not found")
    return attempt


async def _next_attempt_number(db: AsyncSession, quiz_id: UUID, student_id: UUID) -> int:
    from sqlalchemy import text  # noqa: PLC0415

    row = (
        await db.execute(
            text(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_no "
                "FROM quiz_attempts "
                "WHERE quiz_id = :quiz_id AND student_id = :student_id"
            ),
            {"quiz_id": quiz_id, "student_id": student_id},
        )
    ).first()
    return int(row.next_no) if row is not None else 1


async def _load_quiz_questions_for_taking(db: AsyncSession, quiz_id: UUID) -> list[QuizQuestion]:
    from sqlalchemy import select  # noqa: PLC0415

    questions = (
        (
            await db.execute(
                select(QuizQuestion)
                .where(QuizQuestion.quiz_id == quiz_id)
                .order_by(QuizQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    question_ids = [q.id for q in questions]
    options_by_qid: dict[UUID, list[QuizQuestionOption]] = {qid: [] for qid in question_ids}
    if question_ids:
        option_rows = (
            (
                await db.execute(
                    select(QuizQuestionOption)
                    .where(QuizQuestionOption.question_id.in_(question_ids))
                    .order_by(QuizQuestionOption.position)
                )
            )
            .scalars()
            .all()
        )
        for option in option_rows:
            options_by_qid.setdefault(option.question_id, []).append(option)

    for question in questions:
        question.options = options_by_qid.get(question.id, [])  # type: ignore[attr-defined]
    return list(questions)


async def start_attempt(
    db: AsyncSession,
    quiz_id: UUID,
    actor: CurrentUser,
    *,
    idempotency_key: UUID | None = None,
) -> tuple[QuizAttempt, QuizForTakingPublic]:
    """Create a :class:`QuizAttempt` and return the no-leak take payload.

    The serialized response goes through :class:`QuizQuestionPublic`
    which drops ``is_correct`` (and any other answer-correctness fields)
    before the bytes leave the service boundary. Routers must serialize
    the returned :class:`QuizForTakingPublic` directly — never re-hydrate
    via the authoring schema.
    """
    quiz = await _require_quiz(db, quiz_id)
    next_number = await _next_attempt_number(db, quiz_id, actor.user_id)
    attempt = QuizAttempt(
        quiz_id=quiz_id,
        student_id=actor.user_id,
        attempt_number=next_number,
        idempotency_key=idempotency_key,
    )
    db.add(attempt)
    await db.flush()
    await db.refresh(attempt)

    questions = await _load_quiz_questions_for_taking(db, quiz_id)
    public_quiz = QuizPublic.model_validate(quiz)
    public_questions = [QuizQuestionPublic.model_validate(question) for question in questions]
    take_payload = QuizForTakingPublic(quiz=public_quiz, questions=public_questions)
    return attempt, take_payload


async def answer_attempt(
    db: AsyncSession,
    attempt_id: UUID,
    payload: object,
    actor: CurrentUser,
) -> QuizAttemptAnswer:
    """Record one answer for an in-flight attempt.

    Computes ``is_correct`` server-side by looking up the selected
    option's truth flag — the request payload's correctness field (if
    any) is discarded so a malicious client cannot self-grade.
    """
    del actor
    attempt = await _require_attempt(db, attempt_id)

    selected_option_id = getattr(payload, "selected_option_id", None)
    is_correct = False
    if selected_option_id is not None:
        option = await db.get(QuizQuestionOption, selected_option_id)
        is_correct = bool(option and option.is_correct)

    t_actual_ms = getattr(payload, "t_actual_ms", None)
    if t_actual_ms is None:
        t_actual_ms = getattr(payload, "response_time_ms", None)

    answer = QuizAttemptAnswer(
        attempt_id=attempt.id,
        question_id=payload.question_id,  # type: ignore[attr-defined]
        selected_option_id=selected_option_id,
        answer_text=getattr(payload, "answer_text", None),
        is_correct=is_correct,
        hint_used=bool(getattr(payload, "hint_used", False)),
        t_actual_ms=t_actual_ms,
        points_awarded=Decimal("1") if is_correct else Decimal("0"),
    )
    db.add(answer)
    await db.flush()
    await db.refresh(answer)
    return answer


async def submit_attempt(
    db: AsyncSession,
    attempt_id: UUID,
    actor: CurrentUser,
) -> QuizAttempt:
    """Grade and finalize an attempt.

    Computes ``score_percent = score_points / question_count * 100`` and
    flips ``passed`` based on :attr:`Quiz.passing_score_percent`. The
    division uses the quiz's question count (not the answer count) so
    skipped questions count against the score — matching legacy parity.
    """
    del actor
    attempt = await _require_attempt(db, attempt_id)
    quiz = await _require_quiz(db, attempt.quiz_id)

    from sqlalchemy import func, select  # noqa: PLC0415

    answers = (
        (
            await db.execute(
                select(QuizAttemptAnswer).where(QuizAttemptAnswer.attempt_id == attempt_id)
            )
        )
        .scalars()
        .all()
    )
    question_count_row = await db.execute(
        select(func.count(QuizQuestion.id)).where(QuizQuestion.quiz_id == quiz.id)
    )
    question_count = int(question_count_row.scalar_one()) or len(answers) or 1
    score_points = sum((answer.points_awarded for answer in answers), Decimal("0"))
    score_percent = (score_points / Decimal(question_count)) * Decimal("100")

    attempt.status = "submitted"
    attempt.submitted_at = utcnow()
    attempt.time_taken_seconds = int((attempt.submitted_at - attempt.started_at).total_seconds())
    attempt.score_points = score_points
    attempt.score_percent = score_percent
    attempt.passed = score_percent >= quiz.passing_score_percent
    await db.flush()
    await db.refresh(attempt)
    return attempt


async def get_attempt_history(
    db: AsyncSession,
    quiz_id: UUID,
    actor: CurrentUser,
) -> list[QuizAttempt]:
    """Return every attempt the calling student has against ``quiz_id``."""
    from sqlalchemy import select  # noqa: PLC0415

    attempts = (
        (
            await db.execute(
                select(QuizAttempt)
                .where(
                    QuizAttempt.quiz_id == quiz_id,
                    QuizAttempt.student_id == actor.user_id,
                )
                .order_by(QuizAttempt.attempt_number)
            )
        )
        .scalars()
        .all()
    )
    return list(attempts)


__all__ = [
    "answer_attempt",
    "get_attempt_history",
    "get_published_quiz",
    "start_attempt",
    "submit_attempt",
]
