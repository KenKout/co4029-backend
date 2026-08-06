from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import and_, select, text

from abridgeai.features.enrollments.models import Enrollment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_user_enrollment_for_course(
    db: AsyncSession, user_id: UUID, course_id: UUID
) -> Enrollment | None:
    result = await db.execute(
        select(Enrollment).where(
            and_(
                Enrollment.student_id == user_id,
                Enrollment.course_id == course_id,
            )
        )
    )
    return result.scalar_one_or_none()


async def list_my_enrollments(db: AsyncSession, user_id: UUID) -> list[Enrollment]:
    result = await db.execute(
        select(Enrollment)
        .where(Enrollment.student_id == user_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return list(result.scalars().all())


_COURSE_COMPLETION_PERCENT_SQL = text(
    """
    WITH course_lessons AS (
        SELECT l.id AS lesson_id
        FROM modules m
        JOIN lessons l ON l.module_id = m.id
            AND l.deleted_at IS NULL
            AND l.status = 'published'
        WHERE m.course_id = :course_id
          AND m.deleted_at IS NULL
    )
    SELECT
        COUNT(cl.lesson_id) AS lesson_count,
        COALESCE(AVG(COALESCE(lp.completion_percent, 0)), 0)::float AS completion_percent
    FROM course_lessons cl
    LEFT JOIN lesson_progress lp
        ON lp.lesson_id = cl.lesson_id
        AND lp.user_id = :student_id
    """
)


async def get_course_completion_percent(
    db: AsyncSession, *, course_id: UUID, student_id: UUID
) -> tuple[int, float]:
    """``(published_lesson_count, mean completion percent)`` for one student.

    Mirrors the averaging the career-path progress SQL already does, so the
    D2 completion writer and the pathway progress read can never disagree
    about whether a course is finished. A course with **no** published
    lessons returns ``(0, 0.0)`` — the caller must not treat that as 100%
    complete (an empty course is not an achievement).
    """
    row = (
        await db.execute(
            _COURSE_COMPLETION_PERCENT_SQL,
            {"course_id": course_id, "student_id": student_id},
        )
    ).one()
    return int(row.lesson_count), float(row.completion_percent)


__all__ = [
    "get_course_completion_percent",
    "get_user_enrollment_for_course",
    "list_my_enrollments",
]
