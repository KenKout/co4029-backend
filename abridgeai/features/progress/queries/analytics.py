from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from importlib import resources
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_AT_RISK_SQL = text(
    resources.files("abridgeai.features.progress.queries.sql")
    .joinpath("at_risk_students.sql")
    .read_text(encoding="utf-8")
)


_COURSE_PROGRESS_SQL = text(
    resources.files("abridgeai.features.progress.queries.sql")
    .joinpath("course_progress_summary.sql")
    .read_text(encoding="utf-8")
)


@dataclass(frozen=True)
class CourseProgressSummaryRow:
    course_id: UUID
    student_count: int
    avg_completion_percent: float


async def summarize_progress_by_course(
    db: AsyncSession, course_ids: Sequence[UUID]
) -> dict[UUID, CourseProgressSummaryRow]:
    """Average lesson completion per course, keyed by course id.

    Courses with no active enrolments are absent from the mapping rather
    than present at 0%: "nobody enrolled" and "everybody at zero" are
    different facts and the caller renders them differently.
    """
    if not course_ids:
        return {}
    rows = await db.execute(_COURSE_PROGRESS_SQL, {"course_ids": list(course_ids)})
    return {
        row.course_id: CourseProgressSummaryRow(
            course_id=row.course_id,
            student_count=int(row.student_count),
            avg_completion_percent=float(row.avg_completion_percent),
        )
        for row in rows.all()
    }


@dataclass(frozen=True)
class AtRiskRow:
    """One at-risk roster row.

    ``course_id`` is carried on the row because the statement now spans a
    set of courses: the teacher dashboard aggregates across every course a
    teacher authors, and without the id it could not attribute a signal
    back to a course.
    """

    course_id: UUID
    user_id: UUID
    enrolled_at: datetime
    last_engagement_at: datetime | None
    completion_percent: Decimal
    days_since_last_engagement: float | None
    days_since_enrolled: float


async def list_at_risk_rows_for_courses(
    db: AsyncSession,
    course_ids: Sequence[UUID],
    *,
    inactivity_days: int,
    low_completion_percent: int,
    grace_period_days: int,
) -> list[AtRiskRow]:
    """At-risk rows across ``course_ids``.

    Thresholds are passed in rather than read here so the query layer stays
    free of settings lookups; the caller (``services.monitoring``) resolves
    them once and reuses the values for both the SQL filter and the
    human-readable reason text, which is what keeps the two in agreement.

    An empty ``course_ids`` short-circuits: ``= ANY('{}')`` is valid but
    round-tripping to Postgres to learn there is nothing to score is waste.
    """
    if not course_ids:
        return []
    rows = await db.execute(
        _AT_RISK_SQL,
        {
            "course_ids": list(course_ids),
            "inactivity_days": inactivity_days,
            "low_completion_percent": low_completion_percent,
            "grace_period_days": grace_period_days,
        },
    )
    return [
        AtRiskRow(
            course_id=row.course_id,
            user_id=row.user_id,
            enrolled_at=row.enrolled_at,
            last_engagement_at=row.last_engagement_at,
            completion_percent=row.completion_percent,
            days_since_last_engagement=(
                float(row.days_since_last_engagement)
                if row.days_since_last_engagement is not None
                else None
            ),
            days_since_enrolled=float(row.days_since_enrolled),
        )
        for row in rows.all()
    ]


async def list_at_risk_rows(
    db: AsyncSession,
    course_id: UUID,
    *,
    inactivity_days: int,
    low_completion_percent: int,
    grace_period_days: int,
) -> list[AtRiskRow]:
    """Single-course convenience wrapper over the multi-course statement."""
    return await list_at_risk_rows_for_courses(
        db,
        [course_id],
        inactivity_days=inactivity_days,
        low_completion_percent=low_completion_percent,
        grace_period_days=grace_period_days,
    )


__all__ = [
    "AtRiskRow",
    "CourseProgressSummaryRow",
    "list_at_risk_rows",
    "list_at_risk_rows_for_courses",
    "summarize_progress_by_course",
]
