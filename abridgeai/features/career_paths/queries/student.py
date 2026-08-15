from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text

from abridgeai.features.career_paths.models import (
    StudentCareerEnrollment,
    StudentStageProgress,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_LIST_MY_CAREER_ENROLLMENTS_SQL = text(
    """
    SELECT sce.career_path_id, sce.status, sce.started_at, sce.completed_at,
           sce.version_id, cpv.version_no, cp.slug, cp.name
    FROM student_career_enrollments sce
    JOIN career_paths cp ON cp.id = sce.career_path_id
    JOIN career_path_versions cpv ON cpv.id = sce.version_id
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


async def list_active_enrollee_student_ids(db: AsyncSession, career_path_id: UUID) -> list[UUID]:
    """Student ids of every ``active`` enrollee of the path (backfill target)."""
    stmt = select(StudentCareerEnrollment.student_id).where(
        StudentCareerEnrollment.career_path_id == career_path_id,
        StudentCareerEnrollment.status == "active",
        StudentCareerEnrollment.deleted_at.is_(None),
    )
    return list((await db.execute(stmt)).scalars().all())


_PATH_COURSE_PROGRESS_SQL = text(
    """
    WITH path_courses AS (
        SELECT cci.course_id, cci.position, cci.stage_id, cci.is_required,
               cci.satisfied_by, c.slug, c.title, c.status
        FROM career_course_items cci
        JOIN career_path_stages s ON s.id = cci.stage_id
            AND s.deleted_at IS NULL
        JOIN courses c ON c.id = cci.course_id
        WHERE cci.version_id = :version_id
          AND c.status = 'published'
          AND c.deleted_at IS NULL
    ),
    -- Per-course UNIT tally, matching enrollments.queries.completion_units
    -- exactly. `completion_percent` has to be measured the same way
    -- `satisfied` is decided, or the bar reads 100% on a course the stage
    -- gate still considers unfinished (which is what happened while this
    -- averaged lesson progress and the D2 writer counted quizzes).
    lesson_units AS (
        -- Lessons join through modules, NOT module_items: module_items is the
        -- ordering table and a published lesson can have no row in it (one
        -- course in this database has 4 published lessons, 3 module_items).
        -- Quizzes/interviews have no other membership link, so they must.
        SELECT pc.course_id,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE lp.status = 'completed') AS done
        FROM path_courses pc
        JOIN modules m ON m.course_id = pc.course_id AND m.deleted_at IS NULL
        JOIN lessons l ON l.module_id = m.id
            AND l.deleted_at IS NULL
            AND l.status = 'published'
        LEFT JOIN lesson_progress lp
            ON lp.lesson_id = l.id AND lp.user_id = :student_id
        GROUP BY pc.course_id
    ),
    quiz_pop AS (
        SELECT pc.course_id, q.id AS quiz_id,
               CASE WHEN q.allow_retakes THEN q.max_attempts ELSE 1 END AS eff_max
        FROM path_courses pc
        JOIN modules m ON m.course_id = pc.course_id AND m.deleted_at IS NULL
        JOIN module_items mi ON mi.module_id = m.id
            AND mi.item_type = 'quiz'
            AND mi.deleted_at IS NULL
        JOIN quizzes q ON q.id = mi.quiz_id
            AND q.deleted_at IS NULL
            AND q.status = 'published'
    ),
    quiz_units AS (
        SELECT qp.course_id,
               COUNT(*) AS total,
               COUNT(*) FILTER (
                   WHERE COALESCE(g.passed, FALSE)
                      OR (qp.eff_max IS NOT NULL
                          AND COALESCE(a.used, 0) >= qp.eff_max
                          AND COALESCE(a.in_flight, 0) = 0)
               ) AS done
        FROM quiz_pop qp
        LEFT JOIN (
            SELECT quiz_id, passed FROM quiz_grades
            WHERE student_id = :student_id AND grade_item_id IS NULL
        ) g ON g.quiz_id = qp.quiz_id
        LEFT JOIN (
            SELECT quiz_id, COUNT(*) AS used,
                   COUNT(*) FILTER (WHERE status = 'in_progress') AS in_flight
            FROM quiz_attempts WHERE student_id = :student_id
            GROUP BY quiz_id
        ) a ON a.quiz_id = qp.quiz_id
        GROUP BY qp.course_id
    ),
    interview_units AS (
        SELECT pc.course_id,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE COALESCE(s.passed, FALSE)) AS done
        FROM path_courses pc
        JOIN modules m ON m.course_id = pc.course_id AND m.deleted_at IS NULL
        JOIN module_items mi ON mi.module_id = m.id
            AND mi.item_type = 'interview'
            AND mi.deleted_at IS NULL
        JOIN interview_configs ic ON ic.id = mi.interview_config_id
            AND ic.deleted_at IS NULL
            AND ic.status = 'published'
        LEFT JOIN (
            SELECT interview_config_id, BOOL_OR(pass_verdict IS TRUE) AS passed
            FROM interview_sessions
            WHERE student_id = :student_id
            GROUP BY interview_config_id
        ) s ON s.interview_config_id = ic.id
        GROUP BY pc.course_id
    ),
    course_progress AS (
        SELECT pc.course_id, pc.slug, pc.title, pc.status, pc.position,
               pc.stage_id, pc.is_required, pc.satisfied_by,
               COALESCE(lu.total, 0) + COALESCE(qu.total, 0)
                   + COALESCE(iu.total, 0) AS unit_total,
               COALESCE(lu.done, 0) + COALESCE(qu.done, 0)
                   + COALESCE(iu.done, 0) AS unit_done
        FROM path_courses pc
        LEFT JOIN lesson_units lu ON lu.course_id = pc.course_id
        LEFT JOIN quiz_units qu ON qu.course_id = pc.course_id
        LEFT JOIN interview_units iu ON iu.course_id = pc.course_id
    )
    SELECT cp.course_id, cp.slug, cp.title, cp.status, cp.position,
           cp.stage_id, cp.is_required, cp.satisfied_by,
           -- A course with no gradeable unit reports 0, never 100: it cannot
           -- be completed, and the D2 writer refuses to promote it.
           CASE WHEN cp.unit_total = 0 THEN 0.0
                ELSE ROUND(cp.unit_done * 100.0 / cp.unit_total, 2)
           END::float AS completion_percent,
           cp.unit_total, cp.unit_done,
           -- satisfied is course_enrollments.status = 'completed' (D2), NOT
           -- completion_percent >= 100. The two can still differ: a course the
           -- student was never enrolled in has no status row at all, and the
           -- writer only fires for enrolled students.
           COALESCE(ce.status = 'completed', FALSE) AS satisfied,
           (ce.id IS NOT NULL) AS is_enrolled
    FROM course_progress cp
    LEFT JOIN course_enrollments ce
        ON ce.course_id = cp.course_id
        AND ce.student_id = :student_id
    ORDER BY cp.stage_id, cp.position
    """
)


async def get_path_course_progress(
    db: AsyncSession, *, version_id: UUID, student_id: UUID
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _PATH_COURSE_PROGRESS_SQL,
            {"version_id": version_id, "student_id": student_id},
        )
    ).mappings()
    return [dict(row) for row in rows]


async def list_latched_stage_ids(db: AsyncSession, enrollment_id: UUID) -> set[UUID]:
    """Stage ids this enrollment has ever completed (append-only latch)."""
    stmt = select(StudentStageProgress.stage_id).where(
        StudentStageProgress.enrollment_id == enrollment_id
    )
    return set((await db.execute(stmt)).scalars().all())


async def latch_stage_complete(db: AsyncSession, *, enrollment_id: UUID, stage_id: UUID) -> bool:
    """Insert the latch row for a stage that just evaluated complete.

    Idempotent via ``ON CONFLICT DO NOTHING`` on the
    ``(enrollment_id, stage_id)`` unique constraint — two concurrent reads
    both evaluating a stage complete must not raise. Returns ``True`` iff
    this call wrote the row (so the caller knows whether to commit).

    Never UPDATEs and never DELETEs: the table is append-only, which is
    exactly what makes stage completion irreversible.
    """
    result = await db.execute(
        _INSERT_STAGE_LATCH_SQL,
        {"enrollment_id": enrollment_id, "stage_id": stage_id},
    )
    return result.rowcount > 0


_INSERT_STAGE_LATCH_SQL = text(
    """
    INSERT INTO student_stage_progress (enrollment_id, stage_id)
    VALUES (:enrollment_id, :stage_id)
    ON CONFLICT (enrollment_id, stage_id) DO NOTHING
    """
)


_ROSTER_PROGRESS_SQL = text(
    """
    WITH path_courses AS (
        SELECT cci.course_id
        FROM career_course_items cci
        JOIN courses c ON c.id = cci.course_id
        WHERE cci.version_id = :version_id
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


async def get_roster_path_progress(db: AsyncSession, version_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_ROSTER_PROGRESS_SQL, {"version_id": version_id})).mappings()
    return [dict(row) for row in rows]


__all__ = [
    "get_my_career_enrollment",
    "get_path_course_progress",
    "get_roster_path_progress",
    "latch_stage_complete",
    "list_active_enrollee_student_ids",
    "list_latched_stage_ids",
    "list_my_career_enrollments",
]
