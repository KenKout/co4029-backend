-- Full teacher-roster projection: enrollment + progress + risk, one row
-- per active-course-enrollment student. Composes the same lesson-progress
-- aggregation as progress/queries/sql/roster_progress.sql and the same
-- inactivity/completion thresholds as progress/queries/sql/at_risk_students.sql
-- so the "Students" page's numbers never drift from the "Progress" page's.
--
-- at_risk_level buckets (mirrors the FE's RISK_META keys):
--   high   — no engagement ever, or 14+ days inactive
--   medium — 7-14 days inactive, or completion < 30%
--   low    — some incompleteness but recent activity
--   none   — on track
WITH progress AS (
    SELECT
        ce.student_id,
        COUNT(DISTINCT l.id) AS total_lessons,
        COUNT(DISTINCT lp.id) FILTER (WHERE lp.status = 'completed') AS completed_lessons,
        COALESCE(AVG(lp.completion_percent), 0) AS completion_percent,
        MAX(lp.last_activity_at) AS last_activity_at
    FROM course_enrollments ce
    LEFT JOIN modules m ON m.course_id = ce.course_id AND m.deleted_at IS NULL
    LEFT JOIN lessons l ON l.module_id = m.id AND l.deleted_at IS NULL
    LEFT JOIN lesson_progress lp ON lp.lesson_id = l.id AND lp.user_id = ce.student_id
    WHERE ce.course_id = :course_id
    GROUP BY ce.student_id
)
SELECT
    ce.id AS enrollment_id,
    ce.student_id,
    ce.status AS enrollment_status,
    ce.enrolled_at,
    ce.completed_at,
    ce.dropped_at,
    u.primary_email,
    COALESCE(p.display_name, u.primary_email) AS display_name,
    COALESCE(pr.completion_percent, 0) AS progress_percent,
    pr.last_activity_at,
    CASE
        WHEN ce.status != 'active' THEN 'none'
        WHEN pr.last_activity_at IS NULL THEN 'high'
        WHEN pr.last_activity_at < NOW() - INTERVAL '14 days' THEN 'high'
        WHEN pr.last_activity_at < NOW() - INTERVAL '7 days' THEN 'medium'
        WHEN COALESCE(pr.completion_percent, 0) < 30 THEN 'medium'
        WHEN COALESCE(pr.completion_percent, 0) < 60 THEN 'low'
        ELSE 'none'
    END AS at_risk_level
FROM course_enrollments ce
JOIN users u ON u.id = ce.student_id
LEFT JOIN user_profiles p ON p.user_id = u.id
LEFT JOIN progress pr ON pr.student_id = ce.student_id
WHERE ce.course_id = :course_id
ORDER BY ce.enrolled_at DESC
