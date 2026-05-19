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

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, NotFoundError
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


_DEFAULT_FAILURE_COOLDOWN_SECONDS = 86400


class AllCardsInCooldownError(AppError):
    """Every question in the quiz has ``student_card_state.due_at > now``.

    Per thesis UC-LEARN-01 Alt 1a: a student who fails a card cannot
    retry it until the SR scheduler's failure cooldown elapses (default
    24 h, configurable via ``settings.sr_failure_cooldown_seconds``).
    When the *whole* quiz is in cooldown the router must reply HTTP 429
    with a ``Retry-After`` header — :class:`AllCardsInCooldownError`
    carries the timing payload needed to build that response.
    """

    def __init__(
        self,
        retry_available_at: datetime,
        cards_due_at: list[tuple[UUID, datetime]],
    ) -> None:
        super().__init__("All cards in cooldown")
        self.retry_available_at = retry_available_at
        self.cards_due_at = cards_due_at


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
    from sqlalchemy import func, select  # noqa: PLC0415

    stmt = select(func.coalesce(func.max(QuizAttempt.attempt_number), 0) + 1).where(
        QuizAttempt.quiz_id == quiz_id,
        QuizAttempt.student_id == student_id,
    )
    return int((await db.execute(stmt)).scalar_one())


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


async def _load_cooldown_map(
    db: AsyncSession,
    student_id: UUID,
    question_ids: list[UUID],
) -> dict[UUID, datetime]:
    """Return ``{question_id: due_at}`` for cards still in cooldown.

    A row is included only when ``student_card_state.due_at > NOW()``;
    cards the student has never attempted (no row at all) are absent
    from the map and therefore treated as available — per thesis
    UC-LEARN-01 Alt 1a, cooldown applies to *failed* cards only, not to
    cards the learner has not touched yet.

    The query reaches across the spaced-repetition feature boundary on
    purpose: services cannot import ``features/spaced_repetition/models``
    directly (Features-independent contract), so this uses raw SQL
    against the ``student_card_state`` table — mirroring the precedent
    established by T7.5.5's ``record_card_review``.
    """
    from sqlalchemy import text  # noqa: PLC0415

    if not question_ids:
        return {}
    result = await db.execute(
        text(
            "SELECT question_id, due_at FROM student_card_state "
            "WHERE student_id = :sid "
            "  AND question_id = ANY(CAST(:qids AS uuid[])) "
            "  AND due_at > NOW()"
        ),
        {"sid": str(student_id), "qids": [str(q) for q in question_ids]},
    )
    return {row[0]: row[1] for row in result.all()}


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

    **T7.5.11 — cooldown enforcement.** Before persisting the attempt,
    the service consults ``student_card_state.due_at`` for every quiz
    question (cross-feature read against the SR table — see
    :func:`_load_cooldown_map`). Three outcomes:

    * **All questions in cooldown** — raise
      :class:`AllCardsInCooldownError` carrying the earliest
      ``retry_available_at`` plus the per-question ``due_at`` list. The
      router maps this to HTTP 429 with a ``Retry-After`` header.
    * **Some questions in cooldown** — drop them from the take payload
      and stash the ``(question_id, due_at)`` pairs onto
      ``attempt.cards_in_cooldown`` so the router can echo them on the
      response (no schema change required; the field is dynamic on the
      ORM instance).
    * **None in cooldown** — proceed with the existing happy path.

    Cards the student has never attempted (no ``student_card_state``
    row) are *not* in cooldown — they are new material per thesis
    UC-LEARN-01 Alt 1a.
    """
    quiz = await _require_quiz(db, quiz_id)
    questions = await _load_quiz_questions_for_taking(db, quiz_id)

    cooldown_map = await _load_cooldown_map(
        db, actor.user_id, [question.id for question in questions]
    )
    if questions and len(cooldown_map) == len(questions):
        cards_due_at = sorted(cooldown_map.items(), key=lambda item: item[1])
        retry_available_at = cards_due_at[0][1]
        raise AllCardsInCooldownError(retry_available_at, cards_due_at)

    available_questions = [q for q in questions if q.id not in cooldown_map]

    next_number = await _next_attempt_number(db, quiz_id, actor.user_id)
    attempt = QuizAttempt(
        quiz_id=quiz_id,
        student_id=actor.user_id,
        attempt_number=next_number,
        idempotency_key=idempotency_key,
    )
    db.add(attempt)
    await flush_or_conflict(db)
    await db.refresh(attempt)

    attempt.cards_in_cooldown = [  # type: ignore[attr-defined]
        {"question_id": qid, "due_at": due_at} for qid, due_at in cooldown_map.items()
    ]

    public_quiz = QuizPublic.model_validate(quiz)
    public_questions = [
        QuizQuestionPublic.model_validate(question) for question in available_questions
    ]
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
    await flush_or_conflict(db)
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
    await flush_or_conflict(db)
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
