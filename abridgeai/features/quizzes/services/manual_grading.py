"""Manual grading service (Phase 4).

Open-response answers (``code`` always; ``short_answer`` / ``fill_blank`` when
the exact-match auto-grader missed) are flagged ``needs_manual_grade=True`` at
answer time. This service backs the teacher grading queue:

* :func:`list_needs_grading` — the queue: answers awaiting a human, for a quiz.
* :func:`grade_answer_manually` — record a teacher mark + feedback on one answer,
  clear the flag, recompute the parent attempt's headline score (reusing the
  shared helper from ``taking.py`` — DRY), and, when no answers on the attempt
  still need grading, flip the attempt status to ``graded``.

Layering: owns its own DB reads/writes (same precedent as ``services/taking.py``
and ``services/regrade.py``), so routers call here rather than touching queries.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
    QuizQuestion,
)
from abridgeai.features.quizzes.services.taking import _recompute_attempt_score

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_needs_grading(
    db: AsyncSession, *, quiz_id: UUID
) -> list[tuple[QuizAttemptAnswer, QuizQuestion, QuizAttempt]]:
    """Return ``(answer, question, attempt)`` triples awaiting manual grading.

    Scoped to one quiz; only ``needs_manual_grade=True`` answers on
    ``submitted``/``graded`` attempts. Ordered oldest attempt first so a teacher
    works through them in a stable sequence.
    """
    rows = (
        await db.execute(
            select(QuizAttemptAnswer, QuizQuestion, QuizAttempt)
            .join(QuizQuestion, QuizQuestion.id == QuizAttemptAnswer.question_id)
            .join(QuizAttempt, QuizAttempt.id == QuizAttemptAnswer.attempt_id)
            .where(
                QuizAttempt.quiz_id == quiz_id,
                QuizAttemptAnswer.needs_manual_grade.is_(True),
                QuizAttempt.status.in_(["submitted", "graded"]),
            )
            .order_by(QuizAttempt.started_at, QuizAttemptAnswer.id)
        )
    ).all()
    return [(r[0], r[1], r[2]) for r in rows]


async def grade_answer_manually(
    db: AsyncSession,
    *,
    quiz_id: UUID,
    answer_id: UUID,
    score: Decimal,
    feedback: str | None,
    grader_id: UUID | None,
) -> QuizAttemptAnswer:
    """Record a teacher mark on one open-response answer and recompute the attempt.

    * Sets ``manual_score`` / ``manual_feedback`` / ``graded_by`` / ``graded_at``.
    * Mirrors the mark into the scoring fields the attempt sums:
      ``points_awarded = score`` and ``is_correct = score > 0``.
    * Clears ``needs_manual_grade``.
    * Recomputes the parent attempt's headline score (shared helper — DRY).
    * Flips the attempt to ``graded`` once nothing on it still needs grading.

    Guards: the answer must exist and belong to ``quiz_id`` (else 404); a negative
    score is rejected (400 upstream via :class:`AppError`).
    """
    if score < 0:
        raise AppError("Manual score must be >= 0")

    answer = (
        await db.execute(
            select(QuizAttemptAnswer)
            .join(QuizAttempt, QuizAttempt.id == QuizAttemptAnswer.attempt_id)
            .where(
                QuizAttemptAnswer.id == answer_id,
                QuizAttempt.quiz_id == quiz_id,
            )
        )
    ).scalar_one_or_none()
    if answer is None:
        raise NotFoundError(f"Answer {answer_id} not found for quiz {quiz_id}")

    answer.manual_score = score
    answer.manual_feedback = feedback
    answer.graded_by = grader_id
    answer.graded_at = utcnow()
    answer.points_awarded = score
    answer.is_correct = score > 0
    answer.needs_manual_grade = False
    await flush_or_conflict(db)

    attempt = await db.get(QuizAttempt, answer.attempt_id)
    quiz = await db.get(Quiz, quiz_id)
    if attempt is not None and quiz is not None:
        score_points, score_percent, _correct, _count = await _recompute_attempt_score(
            db, attempt, quiz
        )
        attempt.score_points = score_points
        attempt.score_percent = score_percent
        attempt.passed = score_percent >= quiz.passing_score_percent

        # If nothing on this attempt still needs a human, mark it graded.
        remaining = (
            await db.execute(
                select(QuizAttemptAnswer.id).where(
                    QuizAttemptAnswer.attempt_id == attempt.id,
                    QuizAttemptAnswer.needs_manual_grade.is_(True),
                )
            )
        ).first()
        if remaining is None:
            attempt.status = "graded"
        await flush_or_conflict(db)

        # Phase 13: correctness-bearing audit event in the same transaction.
        from abridgeai.features.quizzes.services.audit import (  # noqa: PLC0415
            record_event,
        )

        await record_event(
            db,
            event_name="attempt_manually_graded",
            quiz_id=quiz_id,
            actor_user_id=grader_id,
            subject_attempt_id=attempt.id,
            subject_user_id=attempt.student_id,
            payload={"answer_id": str(answer_id), "score": str(score)},
        )

    await db.refresh(answer)
    return answer


__all__ = ["grade_answer_manually", "list_needs_grading"]
