"""Teacher class-wide analytics queries (T7.5.7).

These views aggregate across all active enrollments in a course. They are
intentionally NOT cached — teacher dashboards are infrequent and the SQL
already does the heavy lifting in one round-trip.

Cross-feature isolation: cards live in ``features.quizzes``, lessons in
``features.courses``, enrollments in ``features.enrollments``. We touch only
their tables via raw SQL, never their ORM classes, to satisfy the
``Features are independent`` contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.sql.elements import TextClause

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_SQL_DIR = resources.files("abridgeai.features.spaced_repetition.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_CLASS_KR_DISTRIBUTION_SQL = _load("class_kr_distribution.sql")
_AT_RISK_SQL = _load("at_risk.sql")

_CLASS_CARD_DIFFICULTY_SQL = text(
    """
    SELECT
        qq.id AS question_id,
        qq.quiz_id AS quiz_id,
        AVG(scs.ef)::float AS mean_ef,
        COUNT(DISTINCT scs.student_id) AS student_count
    FROM quiz_questions qq
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    JOIN student_card_state scs ON scs.question_id = qq.id
    WHERE qsl.lesson_id = CAST(:lesson_id AS uuid)
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
    GROUP BY qq.id, qq.quiz_id
    HAVING COUNT(DISTINCT scs.student_id) > 0
    ORDER BY mean_ef ASC, student_count DESC
    LIMIT :top_n
    """
)


@dataclass(frozen=True)
class ClassKRDistribution:
    """Class-wide R̂ distribution for a single lesson."""

    lesson_id: UUID
    student_count: int
    histogram: list[tuple[float, int]]
    mean_kr: float
    median_kr: float


async def class_kr_distribution(
    db: AsyncSession,
    *,
    course_id: UUID,
    lesson_id: UUID,
) -> ClassKRDistribution:
    """Histogram + mean/median R̂ across all active students in a course."""
    row = (
        await db.execute(
            _CLASS_KR_DISTRIBUTION_SQL,
            {"course_id": str(course_id), "lesson_id": str(lesson_id)},
        )
    ).one()
    raw_histogram = row[3]
    if isinstance(raw_histogram, str):
        raw_histogram = json.loads(raw_histogram)
    histogram = [
        (float(bucket["bucket_lower"]), int(bucket["count"])) for bucket in (raw_histogram or [])
    ]
    return ClassKRDistribution(
        lesson_id=lesson_id,
        student_count=int(row[0] or 0),
        histogram=histogram,
        mean_kr=float(row[1] or 0.0),
        median_kr=float(row[2] or 0.0),
    )


@dataclass(frozen=True)
class DifficultCard:
    """A card whose mean EF across the cohort signals quality issues."""

    question_id: UUID
    quiz_id: UUID
    mean_ef: float
    student_count: int


async def class_card_difficulty(
    db: AsyncSession,
    *,
    lesson_id: UUID,
    top_n: int = 10,
) -> list[DifficultCard]:
    """Top-N hardest cards in the lesson (lowest mean EF across cohort)."""
    rows = (
        await db.execute(
            _CLASS_CARD_DIFFICULTY_SQL,
            {"lesson_id": str(lesson_id), "top_n": int(top_n)},
        )
    ).all()
    return [
        DifficultCard(
            question_id=_as_uuid(row[0]),
            quiz_id=_as_uuid(row[1]),
            mean_ef=float(row[2] or 0.0),
            student_count=int(row[3] or 0),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class AtRiskStudent:
    """A student flagged on at least one at-risk signal (UC-COURSE-04)."""

    student_id: UUID
    name: str
    low_compliance: bool
    frozen_kr: bool
    high_theory_practice_gap: bool
    last_active_at: datetime | None


async def at_risk_students(
    db: AsyncSession,
    *,
    course_id: UUID,
) -> list[AtRiskStudent]:
    """Students in the course flagged by any at-risk signal."""
    rows = (await db.execute(_AT_RISK_SQL, {"course_id": str(course_id)})).all()
    return [
        AtRiskStudent(
            student_id=_as_uuid(row[0]),
            name=str(row[1]),
            low_compliance=bool(row[2]),
            frozen_kr=bool(row[3]),
            high_theory_practice_gap=bool(row[4]),
            last_active_at=row[5],
        )
        for row in rows
    ]


def _as_uuid(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


__all__ = [
    "AtRiskStudent",
    "ClassKRDistribution",
    "DifficultCard",
    "at_risk_students",
    "class_card_difficulty",
    "class_kr_distribution",
]
