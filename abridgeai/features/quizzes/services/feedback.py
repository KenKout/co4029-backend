"""Grade-band feedback service (Phase 8).

Teacher-facing grade bands map a score range to feedback text shown to the
student after submit. Bands are edited as a set (wholesale replace, mirroring
how question options are replaced). Overlaps are rejected at write time; gaps
are allowed. ``select_overall_feedback`` picks the band matching a score.

Layering: owns its own DB reads/writes (precedent: services/taking.py,
services/regrade.py), so routers call here rather than touching queries.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, select

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.features.quizzes.models import Quiz, QuizFeedback
from abridgeai.features.quizzes.schemas.feedback import FeedbackBandIn

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _require_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz:
    quiz = (await db.execute(select(Quiz).where(Quiz.id == quiz_id))).scalar_one_or_none()
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    return quiz


def _validate_no_overlap(bands: list[FeedbackBandIn]) -> None:
    """Reject overlapping bands; gaps are allowed. Raises AppError (→ 422)."""
    for b in bands:
        if b.min_grade >= b.max_grade:
            raise AppError("Each band requires min_grade < max_grade")
    ordered = sorted(bands, key=lambda b: b.min_grade)
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        if nxt.min_grade < prev.max_grade:
            raise AppError("Grade bands must not overlap")


async def list_bands(db: AsyncSession, quiz_id: UUID) -> list[QuizFeedback]:
    """All grade bands for a quiz, ordered by ``min_grade`` ascending."""
    rows = (
        await db.execute(
            select(QuizFeedback)
            .where(QuizFeedback.quiz_id == quiz_id)
            .order_by(QuizFeedback.min_grade)
        )
    ).scalars().all()
    return list(rows)


async def set_feedback_bands(
    db: AsyncSession, *, quiz_id: UUID, bands: list[FeedbackBandIn]
) -> list[QuizFeedback]:
    """Wholesale-replace a quiz's grade bands (delete-all-then-insert).

    Validates min<max and no overlap before touching the DB. Returns the new set.
    """
    await _require_quiz(db, quiz_id)
    _validate_no_overlap(bands)

    await db.execute(delete(QuizFeedback).where(QuizFeedback.quiz_id == quiz_id))
    for b in bands:
        db.add(
            QuizFeedback(
                quiz_id=quiz_id,
                min_grade=b.min_grade,
                max_grade=b.max_grade,
                feedback_text=b.feedback_text,
                feedback_format=b.feedback_format,
            )
        )
    await flush_or_conflict(db)
    return await list_bands(db, quiz_id)


async def select_overall_feedback(
    db: AsyncSession, *, quiz_id: UUID, score_percent: Decimal | None
) -> QuizFeedback | None:
    """Return the band matching ``score_percent`` (``min <= s < max``; top band
    inclusive at 100), or None when no band matches / score is None."""
    if score_percent is None:
        return None
    bands = await list_bands(db, quiz_id)
    for band in bands:
        top_inclusive = band.max_grade >= Decimal("100")
        if band.min_grade <= score_percent and (
            score_percent < band.max_grade
            or (top_inclusive and score_percent <= band.max_grade)
        ):
            return band
    return None


__all__ = ["list_bands", "select_overall_feedback", "set_feedback_bands"]
