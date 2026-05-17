"""Student-self performance metric reads (T7.5.7).

Each function returns a single learner's view of their own progress through a
lesson. Cross-feature reads use raw ``text(...)`` against table names so the
``Features are independent`` import-linter contract is upheld — we never
import ``QuizQuestion``, ``Lesson``, or ``QuizSourceLesson`` here.

Cache integration
-----------------
* :func:`knowledge_retention_estimate` is wrapped with ``@cached(KR_ESTIMATE)``
  (5 min TTL) — value drifts slowly per review and the dashboard fetches it
  on every page load.
* :func:`review_compliance_rate` is wrapped with ``@cached(COMPLIANCE)``
  (1 hour TTL) — the SQL is heavier (CTE + correlated EXISTS) and the value
  changes hourly at most.
* :func:`progression_readiness` and :func:`student_lesson_summary` are NOT
  cached at this layer; readiness already reads through the
  ``LESSON_UNLOCK`` cache via :func:`check_lesson_unlock`, and the summary
  composes the cached metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.sql.elements import TextClause

from abridgeai.core.cache.decorators import cached
from abridgeai.core.cache.keys import COMPLIANCE, KR_ESTIMATE

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_SQL_DIR = resources.files("abridgeai.features.spaced_repetition.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_KR_ESTIMATE_SQL = _load("kr_estimate.sql")
_COMPLIANCE_SQL = _load("compliance_rate.sql")
_LESSON_CARDS_DUE_NOW_SQL = text(
    """
    SELECT
        COUNT(*) AS cards_total,
        COUNT(*) FILTER (
            WHERE scs.due_at IS NOT NULL AND scs.due_at <= NOW()
        ) AS cards_due_now
    FROM quiz_questions qq
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    LEFT JOIN student_card_state scs
        ON scs.question_id = qq.id
        AND scs.student_id = CAST(:student_id AS uuid)
    WHERE qsl.lesson_id = CAST(:lesson_id AS uuid)
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
    """
)


@cached(KR_ESTIMATE)
async def knowledge_retention_estimate(
    db: AsyncSession,
    *,
    user_id: UUID,
    lesson_id: UUID,
) -> float:
    """Per-thesis Knowledge-Retention estimate R̂ in [0, 1].

    ``user_id`` is the student; the parameter name matches the
    ``KR_ESTIMATE`` cache-key placeholder.
    """
    row = (
        await db.execute(
            _KR_ESTIMATE_SQL,
            {"student_id": str(user_id), "lesson_id": str(lesson_id)},
        )
    ).one()
    return float(row[0] or 0.0)


async def progression_readiness(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
) -> bool:
    """Whether the lesson is unlocked for the student (delegates to T7.5.6)."""
    from abridgeai.features.spaced_repetition.sm2.lesson_unlock import (
        check_lesson_unlock,
    )

    status = await check_lesson_unlock(db, student_id=student_id, lesson_id=lesson_id)
    return bool(status.eligible)


@cached(COMPLIANCE)
async def review_compliance_rate(
    db: AsyncSession,
    *,
    user_id: UUID,
    lesson_id: UUID,
    grace_window_seconds: int = 86400,
) -> float | None:
    """Per-thesis review compliance ρ in [0, 1], or None if no due cards.

    ``user_id`` matches the ``COMPLIANCE`` cache-key placeholder.
    """
    row = (
        await db.execute(
            _COMPLIANCE_SQL,
            {
                "student_id": str(user_id),
                "lesson_id": str(lesson_id),
                "grace_window_seconds": int(grace_window_seconds),
            },
        )
    ).one()
    reviewed_in_window = int(row[0] or 0)
    due_total = int(row[1] or 0)
    if due_total == 0:
        return None
    return reviewed_in_window / due_total


@dataclass(frozen=True)
class StudentLessonSummary:
    """Composite of the three thesis metrics for a (student, lesson)."""

    kr_estimate: float
    progression_ready: bool
    compliance_rate: float | None
    cards_total: int
    cards_due_now: int


async def student_lesson_summary(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
) -> StudentLessonSummary:
    """Compose the dashboard summary for one (student, lesson) pair."""
    kr = await knowledge_retention_estimate(db, user_id=student_id, lesson_id=lesson_id)
    ready = await progression_readiness(db, student_id=student_id, lesson_id=lesson_id)
    compliance = await review_compliance_rate(db, user_id=student_id, lesson_id=lesson_id)

    counts = (
        await db.execute(
            _LESSON_CARDS_DUE_NOW_SQL,
            {"student_id": str(student_id), "lesson_id": str(lesson_id)},
        )
    ).one()
    return StudentLessonSummary(
        kr_estimate=kr,
        progression_ready=ready,
        compliance_rate=compliance,
        cards_total=int(counts[0] or 0),
        cards_due_now=int(counts[1] or 0),
    )


__all__ = [
    "StudentLessonSummary",
    "knowledge_retention_estimate",
    "progression_readiness",
    "review_compliance_rate",
    "student_lesson_summary",
]
