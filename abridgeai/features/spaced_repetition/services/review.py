"""SR card-review service — atomic compose of Q → EF → scheduler → persist.

Public entrypoint: :func:`record_card_review`.

Atomicity: the service mutates ``StudentCardState`` and inserts a
``CardReview`` row inside the caller's transaction. The caller is
responsible for ``commit()``; if any step raises, the caller's enclosing
transaction rolls both writes back together.

Cross-feature contract: this module does NOT import
``QuizQuestion`` from ``features.quizzes`` directly. The required
columns (``expected_response_time_ms`` / ``quiz_id``) are read through
:mod:`features.quizzes.api.public` (Wave 4 T21) — the typed
cross-feature surface — so SR holds zero raw SQL into quiz tables for
the per-card scheduler hot path (Wave 5 T30a).

T7.5.10 + BUG-2 fix
-------------------
On q == 0 the service no longer fires the failure event at flush time.
It appends a :class:`CardFailedEvent` to ``CardReviewResult.pending_events``
and the caller is expected to ``await db.commit()`` first, then iterate
the list and dispatch
:func:`abridgeai.features.spaced_repetition.services.remediation.dispatch_remediation_for_card_failure`.
This avoids the ghost-notification race where a rolled-back review
would still trigger a side-effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.quizzes.api.public import (
    get_guess_probability,
    get_question_with_quiz_context,
    get_t_exp_for_question,
)
from abridgeai.features.spaced_repetition.models import CardReview, StudentCardState
from abridgeai.features.spaced_repetition.sm2 import (
    apply_jitter,
    derive_q,
    next_interval_days,
    update_ef,
)

from ._events import CardFailedEvent

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_DEFAULT_FAILURE_COOLDOWN_SECONDS = 86400


@dataclass(frozen=True)
class CardReviewResult:
    """Outcome of :func:`record_card_review` returned to the caller.

    ``pending_events`` carries any :class:`CardFailedEvent` instances
    queued for after-commit dispatch (see module docstring). Always a
    list (possibly empty) so callers can iterate without a ``None``
    guard.
    """

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
    pending_events: list[CardFailedEvent] = field(default_factory=list)


async def _load_quiz_question_meta(
    db: AsyncSession, question_id: UUID
) -> tuple[int, UUID, Decimal | None]:
    t_exp_ms = await get_t_exp_for_question(db, question_id)
    context = await get_question_with_quiz_context(db, question_id)
    if context is None:
        raise NotFoundError(f"QuizQuestion {question_id} not found")
    if t_exp_ms is None:
        raise ValueError(
            f"QuizQuestion {question_id} has no expected_response_time_ms — "
            "T_exp must be set before review (T7.5.9 publish gate)",
        )
    return int(t_exp_ms), context.quiz_id, context.initial_ef


#: Multiple of T_exp beyond which extra time cannot change the derived Q.
#:
#: ``derive_q`` buckets rho = t_actual / t_exp and floors at Q=3 for a correct
#: unhinted answer, so rho=1.01 and rho=60 are already indistinguishable to the
#: model. Clamping therefore costs zero fidelity while stopping pathological
#: values (student left the tab open over lunch, clock skew, a hand-crafted
#: payload) from polluting the stored ``t_actual_ms`` that analytics read.
T_ACTUAL_CAP_MULTIPLIER = 3


def _clamp_t_actual(t_actual_ms: int, t_exp_ms: int) -> int:
    """Clamp a client-reported answer time into a defensible range.

    The client measures per-question attention time and cannot apply this cap
    itself: the student-facing question payload deliberately omits
    ``expected_response_time_ms`` (it would leak how long the teacher thinks a
    question should take), so the ceiling is only knowable server-side.

    Negative values are floored at 0 — the Pydantic schema already enforces
    ``ge=0``, this is defence in depth for any non-HTTP caller.
    """
    if t_actual_ms < 0:
        return 0
    if t_exp_ms <= 0:
        return t_actual_ms
    return min(t_actual_ms, t_exp_ms * T_ACTUAL_CAP_MULTIPLIER)


async def _load_or_init_state(
    db: AsyncSession,
    *,
    student_id: UUID,
    question_id: UUID,
    initial_ef: Decimal | None = None,
) -> tuple[StudentCardState, bool]:
    state = await db.get(StudentCardState, (student_id, question_id))
    if state is not None:
        return state, False
    # Clamp the teacher-supplied initial EF to the same [1.3, 2.5] range the
    # CHECK constraint (ck_student_card_state_ef_range) and update_ef enforce.
    # The quiz settings allow initial_ef to be set without range validation,
    # so an out-of-range value must be clamped here rather than blowing up the
    # first review with an IntegrityError.
    from abridgeai.features.spaced_repetition.sm2 import EF_MAX, EF_MIN

    if initial_ef is not None:
        ef_value = min(EF_MAX, max(EF_MIN, float(initial_ef)))
        initial_ef = Decimal(str(ef_value))
    state = StudentCardState(
        student_id=student_id,
        question_id=question_id,
        ef=initial_ef if initial_ef is not None else Decimal("2.5"),
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
    t_actual_ms: int | None,
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

    On q == 0 the result includes a :class:`CardFailedEvent` in
    ``pending_events`` for the caller to dispatch **after commit** (T7.5.10
    BUG-2 fix). The service itself does not fire anything.

    ``t_actual_ms=None`` (client omitted timing) falls back to
    ``t_exp_ms`` — neutral ρ=1.0. Incorrect answers still derive Q=0
    (timing-independent), so the forgetting signal is preserved.

    New cards seed ``ef`` from ``Quiz.initial_ef`` when the teacher set
    one (FR-4.2); otherwise the SM-2 default 2.5 applies.

    Raises:
        NotFoundError: when ``question_id`` does not match any
            ``quiz_questions`` row.
        ValueError: when the question has no ``expected_response_time_ms``
            (T_exp must be set before scheduling — T7.5.9 publish gate).
    """
    t_exp_ms, quiz_id, initial_ef = await _load_quiz_question_meta(db, question_id)
    if t_actual_ms is None:
        t_actual_ms = t_exp_ms
    else:
        t_actual_ms = _clamp_t_actual(t_actual_ms, t_exp_ms)

    state, _was_created = await _load_or_init_state(
        db, student_id=student_id, question_id=question_id, initial_ef=initial_ef
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
    # Guess-channel dampening (does NOT alter q — the thesis Q is recorded as
    # derived). A correct answer on a format with a 1/N guess channel (MCQ /
    # true_false) grows EF less than genuine free recall, so a fast lucky guess
    # can't balloon the interval. Free-recall formats return 0.0 → scale 1.0 →
    # no change, preserving current behaviour for short_answer / fill_blank etc.
    guess_probability = await get_guess_probability(db, question_id)
    positive_delta_scale = 1.0 - guess_probability
    ef_after = update_ef(
        ef_before, q, n_before, positive_delta_scale=positive_delta_scale
    )

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

    pending_events: list[CardFailedEvent] = []
    if q == 0:
        pending_events.append(
            CardFailedEvent(
                student_id=student_id,
                question_id=question_id,
                quiz_attempt_id=quiz_attempt_id,
                quiz_id=quiz_id,
                timestamp=now,
            )
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
        pending_events=pending_events,
    )


__all__ = ["CardFailedEvent", "CardReviewResult", "record_card_review"]
