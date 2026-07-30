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
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.spaced_repetition.models import StudentCardState
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
    stmt = (
        select(func.count())
        .select_from(StudentCardState)
        .where(
            StudentCardState.student_id == student_id,
            StudentCardState.due_at <= func.now(),
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
    "has_passing_interview_for_module",
    "LessonUnlockStatus",
    "purge_card_state_for_questions",
    "record_card_review",
]
