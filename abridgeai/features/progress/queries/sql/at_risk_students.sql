-- At-risk roster rows for one or more courses.
--
-- Single source of truth for the "at risk" definition: the per-course
-- teacher view, the cross-course teacher dashboard and the cross-feature
-- public API all run THIS statement, so the three surfaces cannot drift
-- apart on what counts as at risk.
--
-- Thresholds arrive as bind parameters rather than literals because they
-- are administrator-tunable (see ``progress.at_risk_*`` in
-- ``core.settings_registry``). ``:inactivity_days`` and
-- ``:low_completion_percent`` set the risk bar; ``:grace_period_days``
-- suppresses risk entirely for students who enrolled too recently to have
-- had a fair chance to engage -- without it every new enrolment is flagged
-- inactive the moment it crosses the inactivity bar.
--
-- Grace is measured from ``enrolled_at``, so it protects a genuinely new
-- student without ever hiding a long-standing one.
WITH user_engagement AS (
    SELECT
        ce.course_id AS course_id,
        ce.student_id AS user_id,
        MAX(ce.enrolled_at) AS enrolled_at,
        MAX(me.created_at) AS last_engagement_at,
        COALESCE(AVG(lp.completion_percent), 0) AS completion_percent,
        (
            SELECT COUNT(*)
            FROM quiz_attempts qa2
            JOIN quizzes q2 ON q2.id = qa2.quiz_id
            WHERE q2.course_id = ce.course_id
              AND qa2.student_id = ce.student_id
              AND qa2.passed = FALSE
        ) AS failed_quiz_attempts,
        (
            SELECT COUNT(*)
            FROM quiz_attempts qa2
            JOIN quizzes q2 ON q2.id = qa2.quiz_id
            WHERE q2.course_id = ce.course_id
              AND qa2.student_id = ce.student_id
              AND qa2.status = 'submitted'
        ) AS ungraded_quiz_attempts,
        (
            SELECT COUNT(*)
            FROM interview_sessions isx2
            JOIN interview_configs ic2 ON ic2.id = isx2.interview_config_id
            WHERE ic2.course_id = ce.course_id
              AND isx2.student_id = ce.student_id
              AND isx2.pass_verdict = FALSE
        ) AS failed_interview_sessions,
        (
            SELECT COUNT(*)
            FROM interview_sessions isx2
            JOIN interview_configs ic2 ON ic2.id = isx2.interview_config_id
            WHERE ic2.course_id = ce.course_id
              AND isx2.student_id = ce.student_id
              AND isx2.pass_verdict IS NULL
              AND isx2.ended_at IS NOT NULL
        ) AS pending_interview_sessions
    FROM course_enrollments ce
    LEFT JOIN modules m ON m.course_id = ce.course_id AND m.deleted_at IS NULL
    LEFT JOIN lessons l ON l.module_id = m.id AND l.deleted_at IS NULL
    LEFT JOIN learning_materials lm ON lm.lesson_id = l.id AND lm.deleted_at IS NULL
    LEFT JOIN learning_material_versions lmv ON lmv.material_id = lm.id
        AND lmv.deleted_at IS NULL
    LEFT JOIN material_engagement me ON me.material_version_id = lmv.id
        AND me.user_id = ce.student_id
    LEFT JOIN lesson_progress lp ON lp.lesson_id = l.id AND lp.user_id = ce.student_id
    WHERE ce.course_id = ANY(:course_ids)
      AND ce.status = 'active'
    GROUP BY ce.course_id, ce.student_id
)
SELECT
    course_id,
    user_id,
    enrolled_at,
    last_engagement_at,
    completion_percent,
    failed_quiz_attempts,
    ungraded_quiz_attempts,
    failed_interview_sessions,
    pending_interview_sessions,
    CASE
        WHEN last_engagement_at IS NULL THEN NULL
        ELSE EXTRACT(EPOCH FROM (NOW() - last_engagement_at)) / 86400.0
    END AS days_since_last_engagement,
    EXTRACT(EPOCH FROM (NOW() - enrolled_at)) / 86400.0 AS days_since_enrolled
FROM user_engagement
-- Grace period first: a student inside it is never at risk, whatever the
-- other signals say.
WHERE enrolled_at <= NOW() - make_interval(days => :grace_period_days)
  AND (
        last_engagement_at IS NULL
     OR last_engagement_at < NOW() - make_interval(days => :inactivity_days)
     OR completion_percent < :low_completion_percent
  )
ORDER BY course_id ASC, user_id ASC
