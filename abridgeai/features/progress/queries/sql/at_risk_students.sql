WITH user_engagement AS (
    SELECT
        ce.student_id AS user_id,
        MAX(me.created_at) AS last_engagement_at,
        COALESCE(AVG(lp.completion_percent), 0) AS completion_percent
    FROM course_enrollments ce
    LEFT JOIN modules m ON m.course_id = ce.course_id AND m.deleted_at IS NULL
    LEFT JOIN lessons l ON l.module_id = m.id AND l.deleted_at IS NULL
    LEFT JOIN learning_materials lm ON lm.lesson_id = l.id AND lm.deleted_at IS NULL
    LEFT JOIN learning_material_versions lmv ON lmv.material_id = lm.id
        AND lmv.deleted_at IS NULL
    LEFT JOIN material_engagement me ON me.material_version_id = lmv.id
        AND me.user_id = ce.student_id
    LEFT JOIN lesson_progress lp ON lp.lesson_id = l.id AND lp.user_id = ce.student_id
    WHERE ce.course_id = :course_id
      AND ce.status = 'active'
    GROUP BY ce.student_id
)
SELECT
    user_id,
    last_engagement_at,
    completion_percent,
    CASE
        WHEN last_engagement_at IS NULL THEN NULL
        ELSE EXTRACT(EPOCH FROM (NOW() - last_engagement_at)) / 86400.0
    END AS days_since_last_engagement
FROM user_engagement
WHERE last_engagement_at IS NULL
   OR last_engagement_at < NOW() - INTERVAL '7 days'
   OR completion_percent < 30
ORDER BY user_id ASC
