from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text

from abridgeai.features.career_paths.models import StudentCareerEnrollment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_LIST_MY_CAREER_ENROLLMENTS_SQL = text(
    """
    SELECT sce.career_path_id, sce.status, sce.started_at, sce.completed_at,
           cp.slug, cp.name
    FROM student_career_enrollments sce
    JOIN career_paths cp ON cp.id = sce.career_path_id
    WHERE sce.student_id = :student_id
      AND sce.deleted_at IS NULL
      AND cp.deleted_at IS NULL
    ORDER BY sce.started_at DESC
    """
)


async def list_my_career_enrollments(db: AsyncSession, student_id: UUID) -> list[dict[str, Any]]:
    rows = (
        await db.execute(_LIST_MY_CAREER_ENROLLMENTS_SQL, {"student_id": student_id})
    ).mappings()
    return [dict(row) for row in rows]


async def get_my_career_enrollment(
    db: AsyncSession, *, student_id: UUID, career_path_id: UUID
) -> StudentCareerEnrollment | None:
    stmt = select(StudentCareerEnrollment).where(
        StudentCareerEnrollment.student_id == student_id,
        StudentCareerEnrollment.career_path_id == career_path_id,
        StudentCareerEnrollment.deleted_at.is_(None),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


_PATH_COURSE_PROGRESS_SQL = text(
    """
    WITH path_courses AS (
        SELECT cci.course_id, cci.position, c.slug, c.title, c.status
        FROM career_course_items cci
        JOIN courses c ON c.id = cci.course_id
        WHERE cci.career_path_id = :career_path_id
          AND c.status = 'published'
          AND c.deleted_at IS NULL
    ),
    course_lessons AS (
        SELECT pc.course_id, l.id AS lesson_id
        FROM path_courses pc
        JOIN modules m ON m.course_id = pc.course_id
            AND m.deleted_at IS NULL
        JOIN lessons l ON l.module_id = m.id
            AND l.deleted_at IS NULL
            AND l.status = 'published'
    ),
    lesson_completion AS (
        SELECT cl.course_id, cl.lesson_id,
               COALESCE(lp.completion_percent, 0) AS completion_percent
        FROM course_lessons cl
        LEFT JOIN lesson_progress lp
            ON lp.lesson_id = cl.lesson_id
            AND lp.user_id = :student_id
    ),
    course_progress AS (
        SELECT pc.course_id, pc.slug, pc.title, pc.status, pc.position,
               COALESCE(AVG(lc.completion_percent), 0)::float AS completion_percent
        FROM path_courses pc
        LEFT JOIN lesson_completion lc ON lc.course_id = pc.course_id
        GROUP BY pc.course_id, pc.slug, pc.title, pc.status, pc.position
    )
    SELECT course_id, slug, title, status, position, completion_percent
    FROM course_progress
    ORDER BY position
    """
)


async def get_path_course_progress(
    db: AsyncSession, *, career_path_id: UUID, student_id: UUID
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _PATH_COURSE_PROGRESS_SQL,
            {"career_path_id": career_path_id, "student_id": student_id},
        )
    ).mappings()
    return [dict(row) for row in rows]


_ROSTER_PROGRESS_SQL = text(
    """
    WITH path_courses AS (
        SELECT cci.course_id
        FROM career_course_items cci
        JOIN courses c ON c.id = cci.course_id
        WHERE cci.career_path_id = :career_path_id
          AND c.status = 'published'
          AND c.deleted_at IS NULL
    ),
    enrolled_students AS (
        SELECT sce.student_id, u.primary_email
        FROM student_career_enrollments sce
        JOIN users u ON u.id = sce.student_id
        WHERE sce.career_path_id = :career_path_id
          AND sce.deleted_at IS NULL
    ),
    course_lessons AS (
        SELECT pc.course_id, l.id AS lesson_id
        FROM path_courses pc
        JOIN modules m ON m.course_id = pc.course_id
            AND m.deleted_at IS NULL
        JOIN lessons l ON l.module_id = m.id
            AND l.deleted_at IS NULL
            AND l.status = 'published'
    ),
    student_course_progress AS (
        SELECT es.student_id, es.primary_email, cl.course_id,
               COALESCE(AVG(COALESCE(lp.completion_percent, 0)), 0)::float
                 AS course_percent
        FROM enrolled_students es
        CROSS JOIN course_lessons cl
        LEFT JOIN lesson_progress lp
            ON lp.lesson_id = cl.lesson_id
            AND lp.user_id = es.student_id
        GROUP BY es.student_id, es.primary_email, cl.course_id
    ),
    aggregated AS (
        SELECT student_id, primary_email,
               COALESCE(AVG(course_percent), 0)::float AS overall_percent,
               SUM(CASE WHEN course_percent >= 100 THEN 1 ELSE 0 END) AS completed_courses,
               COUNT(course_id) AS course_count
        FROM student_course_progress
        GROUP BY student_id, primary_email
    )
    SELECT student_id, primary_email, overall_percent,
           completed_courses, course_count
    FROM aggregated
    UNION ALL
    SELECT es.student_id, es.primary_email, 0::float AS overall_percent,
           0 AS completed_courses, 0 AS course_count
    FROM enrolled_students es
    WHERE NOT EXISTS (SELECT 1 FROM aggregated a WHERE a.student_id = es.student_id)
    ORDER BY primary_email
    """
)


async def get_roster_path_progress(db: AsyncSession, career_path_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_ROSTER_PROGRESS_SQL, {"career_path_id": career_path_id})).mappings()
    return [dict(row) for row in rows]


__all__ = [
    "get_my_career_enrollment",
    "get_path_course_progress",
    "get_roster_path_progress",
    "list_my_career_enrollments",
]
