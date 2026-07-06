"""Typed cross-feature read API for the spaced_repetition feature.

Sibling features (progress dashboards, admin) import from this module
instead of issuing raw ``text(...)`` SQL or reaching into
``features.spaced_repetition.queries`` directly. SR is mostly a
*consumer* of other features (courses, quizzes); the public surface
exported here is small by design -- only the three reads that
external dashboards actually need.

Reads return Pydantic DTOs (the immutable contract); ORM models
(``StudentCardState``, ``CardReview``) stay private.

Soft-delete: every read here uses ORM ``select()`` (or wraps an
existing query helper that does) and inherits the soft-delete
loader-criteria filter automatically. No manual ``deleted_at IS NULL``
is needed.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.spaced_repetition.models import StudentCardState
from abridgeai.features.spaced_repetition.queries.published import (
    review_compliance_rate as _review_compliance_rate,
)
from abridgeai.features.spaced_repetition.queries.unlock_sql import (
    has_passing_interview_for_module as _has_passing_interview_for_module,
)
from abridgeai.features.spaced_repetition.sm2.lesson_unlock import (
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


async def has_passing_interview_for_module(
    db: AsyncSession,
    *,
    student_id: UUID,
    module_id: UUID,
) -> bool:
    """True iff the student has a completed interview session with
    ``pass_verdict = TRUE`` for the module's interview config (FR-5.3).

    Thin wrapper so cross-feature consumers (quizzes attempt-start lock,
    courses lesson gating) depend on the public-API surface.
    """
    return await _has_passing_interview_for_module(db, student_id=student_id, module_id=module_id)


__all__ = [
    "CardStateDTO",
    "LessonUnlockStatus",
    "check_lesson_unlock",
    "get_card_state",
    "get_compliance_rate",
    "get_due_card_count",
    "has_passing_interview_for_module",
]
