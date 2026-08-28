-- Average lesson completion per course, over actively enrolled students.
--
-- Mirrors the completion arm of ``at_risk_students.sql`` on purpose: the
-- "Avg progress" column on Course Health and the low-completion risk rule
-- must be computed the same way, or a course can read 45% average while
-- the risk engine calls most of its roster behind.
--
-- Students with no lesson_progress rows count as 0, not as missing: a
-- cohort where half have not started is at 50%-of-the-starters only if you
-- silently drop the half who did nothing, which flatters the number
-- exactly where attention is most needed.
WITH per_student AS (
    SELECT
        ce.course_id AS course_id,
        ce.student_id AS user_id,
        COALESCE(AVG(lp.completion_percent), 0) AS completion_percent
    FROM course_enrollments ce
    LEFT JOIN modules m ON m.course_id = ce.course_id AND m.deleted_at IS NULL
    LEFT JOIN lessons l ON l.module_id = m.id AND l.deleted_at IS NULL
    LEFT JOIN lesson_progress lp ON lp.lesson_id = l.id AND lp.user_id = ce.student_id
    WHERE ce.course_id = ANY(:course_ids)
      AND ce.status = 'active'
    GROUP BY ce.course_id, ce.student_id
)
SELECT
    course_id,
    COUNT(*) AS student_count,
    COALESCE(AVG(completion_percent), 0) AS avg_completion_percent
FROM per_student
GROUP BY course_id
