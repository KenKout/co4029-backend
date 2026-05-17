"""SR card-review service — atomic compose of Q → EF → scheduler → persist.

Public entrypoint: :func:`record_card_review`.

Atomicity: the service mutates ``StudentCardState`` and inserts a
``CardReview`` row inside the caller's transaction. The caller is
responsible for ``commit()``; if any step raises, the caller's enclosing
transaction rolls both writes back together.

Cross-feature contract: this module does NOT import
``QuizQuestion`` from ``features.quizzes`` — the Features-independent
import-linter contract forbids it. The required column
``quiz_questions.expected_response_time_ms`` is read via raw
``sqlalchemy.text(...)`` against the table directly. The module is
listed in ``[tool.importlinter]`` ``ignore_imports`` for the
"services do not touch SQLAlchemy" contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.spaced_repetition.models import CardReview, StudentCardState
from abridgeai.features.spaced_repetition.sm2 import (
    apply_jitter,
    derive_q,
    next_interval_days,
    update_ef,
)

from ._events import CardFailedEvent, emit_card_failed

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_DEFAULT_FAILURE_COOLDOWN_SECONDS = 86400


@dataclass(frozen=True)
class CardReviewResult:
    """Outcome of :func:`record_card_review` returned to the caller."""

    q: int
    ef_before: float
    ef_after: float
    interval_before: int
    interval_after: int
    repetition_count_after: int
    due_at: datetime
    last_q: int
    passing: bool
    retry_available_at: datetime | None
    calibration_active: bool


async def _load_quiz_question_meta(db: AsyncSession, question_id: UUID) -> tuple[int, UUID]:
    """Read T_exp + parent quiz_id for ``question_id`` via raw SQL.

    Cross-feature read: importing ``QuizQuestion`` here would violate
    the "Features are independent" contract, so we go through ``text()``.
    """
    result = await db.execute(
        text("SELECT expected_response_time_ms, quiz_id FROM quiz_questions WHERE id = :qid"),
        {"qid": str(question_id)},
    )
    row = result.first()
    if row is None:
        raise NotFoundError(f"QuizQuestion {question_id} not found")
    t_exp_ms, quiz_id = row[0], row[1]
    if t_exp_ms is None:
        raise ValueError(
            f"QuizQuestion {question_id} has no expected_response_time_ms — "
            "T_exp must be set before review (T7.5.9 publish gate)",
        )
    return int(t_exp_ms), quiz_id if isinstance(quiz_id, UUID) else UUID(str(quiz_id))


async def _load_or_init_state(
    db: AsyncSession, *, student_id: UUID, question_id: UUID
) -> tuple[StudentCardState, bool]:
    state = await db.get(StudentCardState, (student_id, question_id))
    if state is not None:
        return state, False
    state = StudentCardState(
        student_id=student_id,
        question_id=question_id,
        ef=Decimal("2.5"),
        interval_days=1,
        repetition_count=0,
        due_at=datetime.now(tz=UTC),
        last_q=None,
        last_reviewed_at=None,
        calibration_active=True,
        total_reviews=0,
    )
    db.add(state)
    return state, True


async def record_card_review(
    db: AsyncSession,
    *,
    student_id: UUID,
    question_id: UUID,
    quiz_attempt_id: UUID | None,
    t_actual_ms: int,
    correct: bool,
    hint_used: bool,
    actor_id: UUID | None = None,  # noqa: ARG001 — reserved for future explicit override
) -> CardReviewResult:
    """Record a card review atomically.

    Composes:
        derive_q (T7.5.2) → update_ef (T7.5.3) → next_interval_days +
        apply_jitter (T7.5.4) → INSERT CardReview + UPSERT
        StudentCardState in the caller's transaction.

    Failure path (q < 3) resets the SM-2 counters (n=0, interval=1) and
    pushes ``due_at`` out by the configured cooldown (default 24 h). The
    ``retry_available_at`` field on the result mirrors that timestamp so
    the quiz router (T7.5.11) can enforce a per-card retry block.

    Raises:
        NotFoundError: when ``question_id`` does not match any
            ``quiz_questions`` row.
        ValueError: when the question has no ``expected_response_time_ms``
            (T_exp must be set before scheduling — T7.5.9 publish gate).
    """
    t_exp_ms, quiz_id = await _load_quiz_question_meta(db, question_id)

    state, _was_created = await _load_or_init_state(
        db, student_id=student_id, question_id=question_id
    )

    ef_before_dec = state.ef
    ef_before = float(ef_before_dec)
    interval_before = state.interval_days
    n_before = state.repetition_count

    q = derive_q(
        correct=correct,
        hint_used=hint_used,
        t_actual_ms=t_actual_ms,
        t_exp_ms=t_exp_ms,
    )
    ef_after = update_ef(ef_before, q, n_before)

    now = datetime.now(tz=UTC)
    passing = q >= 3
    retry_available_at: datetime | None
    if passing:
        n_after = n_before + 1
        base_interval = next_interval_days(
            ef=ef_after, n=n_before, q=q, prev_interval=interval_before
        )
        interval_after = apply_jitter(base_interval, fraction=0.1)
        due_at = now + timedelta(days=interval_after)
        retry_available_at = None
    else:
        n_after = 0
        interval_after = 1
        cooldown_seconds = int(
            getattr(
                get_settings(),
                "sr_failure_cooldown_seconds",
                _DEFAULT_FAILURE_COOLDOWN_SECONDS,
            )
        )
        due_at = now + timedelta(seconds=cooldown_seconds)
        retry_available_at = due_at

    rho = (Decimal(t_actual_ms) / Decimal(t_exp_ms) if t_exp_ms > 0 else Decimal(0)).quantize(
        Decimal("0.0001")
    )

    review = CardReview(
        student_id=student_id,
        question_id=question_id,
        quiz_attempt_id=quiz_attempt_id,
        t_actual_ms=t_actual_ms,
        t_exp_ms=t_exp_ms,
        rho=rho,
        correct=correct,
        hint_used=hint_used,
        q_derived=q,
        ef_before=ef_before_dec,
        ef_after=Decimal(str(ef_after)),
        interval_before=interval_before,
        interval_after=interval_after,
        n_before=n_before,
        n_after=n_after,
    )
    db.add(review)

    state.ef = Decimal(str(ef_after))
    state.interval_days = interval_after
    state.repetition_count = n_after
    state.due_at = due_at
    state.last_q = q
    state.last_reviewed_at = now
    state.calibration_active = n_after <= 3
    state.total_reviews = state.total_reviews + 1

    await db.flush()

    if q == 0:
        await emit_card_failed(
            db,
            CardFailedEvent(
                student_id=student_id,
                question_id=question_id,
                quiz_attempt_id=quiz_attempt_id,
                quiz_id=quiz_id,
                timestamp=now,
            ),
        )

    return CardReviewResult(
        q=q,
        ef_before=ef_before,
        ef_after=ef_after,
        interval_before=interval_before,
        interval_after=interval_after,
        repetition_count_after=n_after,
        due_at=due_at,
        last_q=q,
        passing=passing,
        retry_available_at=retry_available_at,
        calibration_active=n_after <= 3,
    )


__all__ = ["CardReviewResult", "record_card_review"]
