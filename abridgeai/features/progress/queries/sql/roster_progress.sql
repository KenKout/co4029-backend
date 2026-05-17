SELECT
    u.id AS user_id,
    COUNT(DISTINCT l.id) AS total_lessons,
    COUNT(DISTINCT lp.id) FILTER (WHERE lp.status = 'completed') AS completed_lessons,
    COUNT(DISTINCT lp.id) FILTER (WHERE lp.status = 'in_progress') AS in_progress_lessons,
    COUNT(DISTINCT l.id) - COUNT(DISTINCT lp.id) AS not_started_lessons,
    COALESCE(AVG(lp.completion_percent), 0) AS completion_percent,
    COALESCE(SUM(lp.total_time_seconds), 0) AS total_time_seconds
FROM course_enrollments ce
JOIN users u ON u.id = ce.student_id
LEFT JOIN modules m ON m.course_id = ce.course_id AND m.deleted_at IS NULL
LEFT JOIN lessons l ON l.module_id = m.id AND l.deleted_at IS NULL
LEFT JOIN lesson_progress lp ON lp.lesson_id = l.id AND lp.user_id = u.id
WHERE ce.course_id = :course_id
  AND ce.status = 'active'
GROUP BY u.id
ORDER BY completion_percent ASC, u.id ASC
