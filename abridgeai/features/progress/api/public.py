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

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.progress.models import LessonProgress
from abridgeai.features.progress.services import monitoring, reporting

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

    Delegates to :func:`progress.services.monitoring.get_at_risk_students`
    rather than to the raw query, so cross-feature callers get rows scored
    against the SAME administrator-tunable thresholds and grace period as
    the teacher-facing surfaces. Calling the query directly would mean
    picking thresholds here, which is exactly the drift this module exists
    to prevent.
    """
    result = await monitoring.get_at_risk_students(db, course_id)
    return [
        AtRiskStudentDTO(
            user_id=student.user_id,
            completion_percent=student.completion_percent,
            days_since_last_engagement=(
                float(student.days_since_last_engagement)
                if student.days_since_last_engagement is not None
                else None
            ),
            primary_reason=student.reasons[0].detail if student.reasons else None,
            signal_count=len(student.reasons),
        )
        for student in result.students
    ]


async def count_students_needing_attention(
    db: AsyncSession,
    course_ids: Sequence[UUID],
) -> int:
    """DISTINCT students at risk across ``course_ids``.

    Backs the teacher dashboard's headline "students needing attention"
    figure. Exposed as a count rather than as rows because the caller needs
    a number and shipping the roster across a feature boundary would leak
    per-student data no aggregate tile can use.
    """
    return await monitoring.count_students_needing_attention(db, course_ids)


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
    "count_students_needing_attention",
    "get_at_risk_students",
    "get_course_progress_for_user",
    "get_lesson_progress",
]
