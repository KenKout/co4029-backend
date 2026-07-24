"""Gradebook service (Phase 9): materialised grade-of-record.

A student's grade of record for a quiz is derived from their completed attempts
via the quiz's ``grading_method`` (highest / average / first / last) and stored
in ``quiz_grades`` (whole-quiz row = ``grade_item_id IS NULL``). It is refreshed
whenever an attempt is submitted or a regrade is committed, so the gradebook
never drifts from the underlying attempts.

Layering: owns its own DB reads/writes (precedent: services/taking.py). The
reducer ``_compute_final_grade`` is pure and unit-tested in isolation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.models import Quiz, QuizAttempt, QuizGrade

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class AttemptScore:
    attempt_id: uuid.UUID
    attempt_number: int
    score_percent: Decimal
    score_points: Decimal


@dataclass(frozen=True)
class FinalGrade:
    grade_percent: Decimal
    grade_points: Decimal
    based_on_attempt_id: uuid.UUID | None
    attempts_counted: int


def _compute_final_grade(attempts: list[AttemptScore], method: str) -> FinalGrade | None:
    """Reduce per-attempt scores to a single grade via ``grading_method``.

    Pure — mirrors Moodle's grade_calculator. Returns None for no attempts.
    """
    if not attempts:
        return None
    if method == "highest":
        top = max(attempts, key=lambda a: a.score_percent)
        return FinalGrade(top.score_percent, top.score_points, top.attempt_id, len(attempts))
    if method == "first":
        a = min(attempts, key=lambda a: a.attempt_number)
        return FinalGrade(a.score_percent, a.score_points, a.attempt_id, len(attempts))
    if method == "last":
        a = max(attempts, key=lambda a: a.attempt_number)
        return FinalGrade(a.score_percent, a.score_points, a.attempt_id, len(attempts))
    if method == "average":
        n = len(attempts)
        avg_pct = sum((a.score_percent for a in attempts), Decimal(0)) / n
        avg_pts = sum((a.score_points for a in attempts), Decimal(0)) / n
        return FinalGrade(
            avg_pct.quantize(Decimal("0.01")),
            avg_pts.quantize(Decimal("0.01")),
            None,
            n,
        )
    raise ValueError(f"unknown grading_method: {method}")


async def recompute_final_grade(
    db: AsyncSession, quiz: Quiz, student_id: uuid.UUID
) -> QuizGrade | None:
    """Refresh the whole-quiz grade-of-record for one student.

    Participates in the caller's transaction (submit / regrade) — does NOT
    commit. Deletes the grade row when the student has no completed attempts.
    """
    rows = (
        await db.execute(
            select(QuizAttempt)
            .where(
                QuizAttempt.quiz_id == quiz.id,
                QuizAttempt.student_id == student_id,
                QuizAttempt.status.in_(["submitted", "graded"]),
            )
            .order_by(QuizAttempt.attempt_number)
        )
    ).scalars().all()

    scores = [
        AttemptScore(
            attempt_id=a.id,
            attempt_number=a.attempt_number,
            score_percent=a.score_percent if a.score_percent is not None else Decimal("0"),
            score_points=a.score_points if a.score_points is not None else Decimal("0"),
        )
        for a in rows
    ]
    final = _compute_final_grade(scores, quiz.grading_method)

    existing = (
        await db.execute(
            select(QuizGrade).where(
                QuizGrade.quiz_id == quiz.id,
                QuizGrade.student_id == student_id,
                QuizGrade.grade_item_id.is_(None),
            )
        )
    ).scalar_one_or_none()

    if final is None:
        if existing is not None:
            await db.execute(
                delete(QuizGrade).where(QuizGrade.id == existing.id)
            )
            await flush_or_conflict(db)
        return None

    passed = final.grade_percent >= quiz.passing_score_percent
    if existing is None:
        existing = QuizGrade(quiz_id=quiz.id, student_id=student_id, grade_item_id=None)
        db.add(existing)
    existing.grade_percent = final.grade_percent
    existing.grade_points = final.grade_points
    existing.passed = passed
    existing.grading_method = quiz.grading_method
    existing.based_on_attempt_id = final.based_on_attempt_id
    existing.attempts_counted = final.attempts_counted
    existing.computed_at = utcnow()
    await flush_or_conflict(db)
    return existing


async def list_quiz_grades(db: AsyncSession, quiz_id: uuid.UUID) -> list[QuizGrade]:
    """All whole-quiz grade-of-record rows for a quiz (teacher gradebook)."""
    rows = (
        await db.execute(
            select(QuizGrade).where(
                QuizGrade.quiz_id == quiz_id,
                QuizGrade.grade_item_id.is_(None),
            )
        )
    ).scalars().all()
    return list(rows)


__all__ = [
    "AttemptScore",
    "FinalGrade",
    "list_quiz_grades",
    "recompute_final_grade",
]
