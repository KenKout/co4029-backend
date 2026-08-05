"""Typed cross-feature API for the spaced_repetition feature.

Sibling features (progress dashboards, admin, quizzes) import from
this module instead of issuing raw ``text(...)`` SQL or reaching into
``features.spaced_repetition.queries`` / ``services`` directly.

Reads return Pydantic DTOs (the immutable contract); ORM models
(``StudentCardState``, ``CardReview``) stay private.

Write surface (FR-4.4 learning loop)
------------------------------------
``record_card_review``
    THE SM-2 entrypoint. The quizzes answer flow calls this after
    grading so every answer updates ``student_card_state`` +
    ``card_reviews`` (Q → EF → schedule). Returns
    :class:`CardReviewResult`, a frozen dataclass (not ORM).

``dispatch_remediation_for_card_failure``
    Post-commit side-effect for each :class:`CardFailedEvent` found in
    ``CardReviewResult.pending_events`` — see the
    caller-dispatches-after-commit pattern in ``services/_events.py``.

Soft-delete: every read here uses ORM ``select()`` (or wraps an
existing query helper that does) and inherits the soft-delete
loader-criteria filter automatically. No manual ``deleted_at IS NULL``
is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, time
from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.spaced_repetition.models import CardReview, StudentCardState
from abridgeai.features.spaced_repetition.queries.published import (
    review_compliance_rate as _review_compliance_rate,
)
from abridgeai.features.spaced_repetition.queries.unlock_sql import (
    has_passing_interview_for_module,
)
from abridgeai.features.spaced_repetition.services import (
    CardFailedEvent,
    CardReviewResult,
    dispatch_remediation_for_card_failure,
    record_card_review,
)
from abridgeai.features.spaced_repetition.sm2 import (
    LessonUnlockStatus,
    check_lesson_unlock,
)

from ._dto import CardStateDTO


async def get_card_state(
    db: AsyncSession,
    *,
    student_id: UUID,
    question_id: UUID,
) -> CardStateDTO | None:
    state = await db.get(StudentCardState, (student_id, question_id))
    return CardStateDTO.model_validate(state) if state else None


async def get_due_card_count(db: AsyncSession, student_id: UUID) -> int:
    """Count a student's due, REVIEWABLE cards.

    "Reviewable" is the crux: a card is only counted if its question still
    resolves to an approved, non-deleted payload the review loop can actually
    serve. This must match the reviewability predicate in the cards-due /
    review-queue SQL (``routers/learner.py:_CARDS_DUE_SQL``) exactly —
    otherwise ``total_due`` counts cards whose question is ``pending`` / draft /
    soft-deleted and the student sees "21 due" but Start Review serves fewer,
    with a course row that has nothing behind it. A raw join (rather than the
    bare ``student_card_state`` count) is required because ``review_status`` and
    the soft-delete flags live on the quizzes-side tables.
    """
    stmt = text(
        """
        SELECT count(*)
        FROM student_card_state scs
        JOIN quiz_questions qq ON qq.id = scs.question_id
        JOIN quizzes q ON q.id = qq.quiz_id
        JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
        JOIN lessons l ON l.id = qsl.lesson_id
        WHERE scs.student_id = CAST(:student_id AS uuid)
          AND scs.due_at IS NOT NULL
          AND scs.due_at <= NOW()
          AND qq.deleted_at IS NULL
          AND q.deleted_at IS NULL
          AND l.deleted_at IS NULL
          AND qq.review_status = 'approved'
        """
    )
    return int((await db.execute(stmt, {"student_id": str(student_id)})).scalar_one())


async def get_reviews_done_today(db: AsyncSession, student_id: UUID) -> int:
    """Count ``card_reviews`` the student has recorded since UTC midnight.

    Feeds the daily-review-cap ceiling: cards already reviewed today count
    against the cap so a student can't reset it by re-entering the queue. UTC
    day boundary matches the rest of the SR scheduler (all ``due_at`` /
    ``created_at`` are UTC), so "today" is consistent with due-card maths.
    Every answer — pass or fail, review-loop or in-quiz — writes a
    ``CardReview`` row, so this counts genuine review effort.
    """
    start_of_day = datetime.combine(datetime.now(tz=UTC).date(), time.min, tzinfo=UTC)
    stmt = (
        select(func.count())
        .select_from(CardReview)
        .where(
            CardReview.student_id == student_id,
            CardReview.created_at >= start_of_day,
        )
    )
    return int((await db.execute(stmt)).scalar_one())


async def get_compliance_rate(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
) -> float | None:
    """Per-thesis review compliance ρ in [0, 1], or None if no due cards.

    Thin wrapper over the existing CTE-backed helper; the wrapper exists
    so consumers depend on the public-API surface, not on
    ``features.spaced_repetition.queries``.
    """
    return await _review_compliance_rate(db, user_id=student_id, lesson_id=lesson_id)


async def purge_card_state_for_questions(
    db: AsyncSession,
    question_ids: Sequence[UUID],
) -> int:
    """Hard-delete SM-2 card state for the given questions. Returns rows removed.

    Called by the quizzes feature when a question or quiz is (soft-)deleted:
    ``student_card_state`` is per-student SM-2 scheduling state keyed on
    ``question_id`` and is meaningless once the question no longer exists.
    Unlike the authored content (soft-deleted for audit/restore), this state is
    disposable, so we hard-delete it rather than leave orphaned rows that keep
    surfacing as perpetually-"due" cards no one can review — they can't reach
    the take/answer surface (which joins live questions) yet still inflate
    reminder counts and every teacher analytic that reads the table directly.

    Cross-feature write contract: quizzes cannot import the SR ORM model, so
    this is the authorised entrypoint. Runs in the caller's transaction (no
    commit here). ``card_reviews`` history is intentionally left intact — it is
    an immutable audit trail and is never read as "due".
    """
    if not question_ids:
        return 0
    result = await db.execute(
        delete(StudentCardState).where(StudentCardState.question_id.in_(list(question_ids)))
    )
    return int(result.rowcount or 0)


__all__ = [
    "CardFailedEvent",
    "CardReviewResult",
    "CardStateDTO",
    "check_lesson_unlock",
    "dispatch_remediation_for_card_failure",
    "get_card_state",
    "get_compliance_rate",
    "get_due_card_count",
    "get_reviews_done_today",
    "has_passing_interview_for_module",
    "LessonUnlockStatus",
    "purge_card_state_for_questions",
    "record_card_review",
]
