"""Public, typed cross-feature read API for the progress feature.

This module is the *only* path other features may use to reach into
progress. Sibling modules (``models``, ``queries``, ``services``) remain
feature-internal.

Reads return Pydantic DTOs (the immutable contract); ORM models stay
private. The at-risk aggregation re-uses the file-backed SQL in
``progress/queries/sql/at_risk_students.sql`` so the cross-feature
projection cannot drift from the dashboard projection.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.progress.models import LessonProgress
from abridgeai.features.progress.queries.analytics import list_at_risk_rows
from abridgeai.features.progress.services import reporting

from ._dto import AtRiskStudentDTO, LessonProgressDTO


async def get_lesson_progress(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
) -> LessonProgressDTO | None:
    """Return the progress row for a single (student, lesson) pair.

    Returns ``None`` if no row exists yet (i.e. the student has never
    engaged with this lesson). The unique constraint
    ``uq_lesson_progress_user_lesson`` guarantees at most one match.
    """
    stmt = select(LessonProgress).where(
        LessonProgress.user_id == student_id,
        LessonProgress.lesson_id == lesson_id,
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    return LessonProgressDTO.model_validate(row) if row is not None else None


async def get_at_risk_students(
    db: AsyncSession,
    course_id: UUID,
) -> list[AtRiskStudentDTO]:
    """Return the at-risk roster for ``course_id``.

    Wraps :func:`abridgeai.features.progress.queries.analytics.list_at_risk_rows`
    (which reads ``progress/queries/sql/at_risk_students.sql``) and
    re-projects each row through :class:`AtRiskStudentDTO`. The
    underlying SQL stays the source of truth for the at-risk
    definition; this surface is purely a typed pass-through.
    """
    rows = await list_at_risk_rows(db, course_id)
    return [
        AtRiskStudentDTO(
            user_id=row.user_id,
            last_engagement_at=row.last_engagement_at,
            completion_percent=row.completion_percent,
            days_since_last_engagement=row.days_since_last_engagement,
        )
        for row in rows
    ]


async def get_course_progress_for_user(
    db: AsyncSession,
    *,
    user_id: UUID,
    course_id: UUID,
) -> dict[str, object]:
    """Per-user course progress summary for a (course, student) pair.

    Cross-feature read backing the manager/HOD user-detail page: how far a
    student is through a course (completed / in-progress / not-started
    lessons, completion percent, last activity). The learner endpoint
    serves the signed-in student; this is the same projection for any
    user, exposed through the public API so sibling features never reach
    into progress internals.
    """
    summary = await reporting.get_my_course_progress_summary(
        db, user_id=user_id, course_id=course_id
    )
    return summary.model_dump()


__all__ = [
    "AtRiskStudentDTO",
    "LessonProgressDTO",
    "get_at_risk_students",
    "get_course_progress_for_user",
    "get_lesson_progress",
]
