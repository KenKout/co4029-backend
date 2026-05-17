-- At-risk students per course (UC-COURSE-04 composite signal).
--
-- A student is "at risk" when ANY of three signals trips:
--
-- 1. low_compliance: ρ < 0.5 across the whole course (cards reviewed within
--    24h grace / cards due ever). Computed inline rather than calling
--    compliance_rate.sql per (student, lesson) to keep this single-shot.
--
-- 2. frozen_kr: the student's last 7 days of card_reviews show no EF
--    movement on any card in the course (i.e., either no reviews at all
--    in 7 days, or every review left ef_after == ef_before — the SM-2
--    update_ef formula returns the same value when q=3 and ef stays
--    capped, but in practice 7 days of zero EF deltas means the student
--    has stopped engaging).
--
-- 3. high_theory_practice_gap: average quiz score - average interview
--    pass-verdict score > 30. Interviews don't have a numeric score, so
--    we treat ``pass_verdict=true => 100``, ``pass_verdict=false => 0``,
--    NULL excluded. Quiz score uses ``score_percent`` from quiz_attempts
--    (0..100). Gap > 30 means the student passes paper but flunks oral.
--
-- ``last_active_at`` is the most recent card_review or quiz_attempt
-- across the course; NULL if neither.
WITH class_students AS (
    SELECT ce.student_id
    FROM course_enrollments ce
    WHERE ce.course_id = CAST(:course_id AS uuid)
      AND ce.status = 'active'
),
course_questions AS (
    SELECT DISTINCT qq.id AS question_id
    FROM quiz_questions qq
    JOIN quizzes q ON q.id = qq.quiz_id
    WHERE q.course_id = CAST(:course_id AS uuid)
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
),
compliance_per_student AS (
    SELECT
        cs.student_id,
        COUNT(*) FILTER (WHERE scs.due_at <= NOW()) AS due_total,
        COUNT(*) FILTER (
            WHERE scs.due_at <= NOW()
            AND EXISTS (
                SELECT 1 FROM card_reviews cr
                WHERE cr.student_id = cs.student_id
                  AND cr.question_id = scs.question_id
                  AND cr.created_at >= scs.due_at
                  AND cr.created_at <= scs.due_at + INTERVAL '86400 seconds'
            )
        ) AS reviewed_in_window
    FROM class_students cs
    LEFT JOIN student_card_state scs ON scs.student_id = cs.student_id
    LEFT JOIN course_questions cq ON cq.question_id = scs.question_id
    WHERE cq.question_id IS NOT NULL OR scs.question_id IS NULL
    GROUP BY cs.student_id
),
recent_ef_movement AS (
    SELECT
        cs.student_id,
        BOOL_OR(cr.ef_after <> cr.ef_before) AS any_movement
    FROM class_students cs
    LEFT JOIN card_reviews cr
        ON cr.student_id = cs.student_id
        AND cr.created_at >= NOW() - INTERVAL '7 days'
        AND cr.question_id IN (SELECT question_id FROM course_questions)
    GROUP BY cs.student_id
),
quiz_avg AS (
    SELECT
        cs.student_id,
        AVG(qa.score_percent)::float AS avg_quiz_score
    FROM class_students cs
    LEFT JOIN quiz_attempts qa
        ON qa.student_id = cs.student_id
        AND qa.score_percent IS NOT NULL
    LEFT JOIN quizzes q ON q.id = qa.quiz_id AND q.course_id = CAST(:course_id AS uuid)
    WHERE qa.id IS NULL OR q.id IS NOT NULL
    GROUP BY cs.student_id
),
interview_avg AS (
    SELECT
        cs.student_id,
        AVG(CASE WHEN ist.pass_verdict THEN 100.0 ELSE 0.0 END) AS avg_interview_score
    FROM class_students cs
    LEFT JOIN interview_sessions ist
        ON ist.student_id = cs.student_id
        AND ist.pass_verdict IS NOT NULL
    LEFT JOIN interview_configs ic
        ON ic.id = ist.interview_config_id
    LEFT JOIN modules m ON m.id = ic.module_id AND m.course_id = CAST(:course_id AS uuid)
    WHERE ist.id IS NULL OR m.id IS NOT NULL
    GROUP BY cs.student_id
),
last_active AS (
    SELECT
        cs.student_id,
        GREATEST(
            COALESCE((
                SELECT MAX(cr.created_at)
                FROM card_reviews cr
                WHERE cr.student_id = cs.student_id
                  AND cr.question_id IN (SELECT question_id FROM course_questions)
            ), 'epoch'::timestamptz),
            COALESCE((
                SELECT MAX(qa.started_at)
                FROM quiz_attempts qa
                JOIN quizzes q ON q.id = qa.quiz_id
                WHERE qa.student_id = cs.student_id
                  AND q.course_id = CAST(:course_id AS uuid)
            ), 'epoch'::timestamptz)
        ) AS last_active_at_raw
    FROM class_students cs
)
SELECT
    cs.student_id,
    COALESCE(up.display_name, u.primary_email) AS name,
    (
        cps.due_total > 0
        AND (cps.reviewed_in_window::float / cps.due_total) < 0.5
    ) AS low_compliance,
    (
        ref.any_movement IS NULL
        OR ref.any_movement = false
    ) AS frozen_kr,
    (
        qa.avg_quiz_score IS NOT NULL
        AND ia.avg_interview_score IS NOT NULL
        AND (qa.avg_quiz_score - ia.avg_interview_score) > 30.0
    ) AS high_theory_practice_gap,
    NULLIF(la.last_active_at_raw, 'epoch'::timestamptz) AS last_active_at
FROM class_students cs
JOIN users u ON u.id = cs.student_id
LEFT JOIN user_profiles up ON up.user_id = cs.student_id
LEFT JOIN compliance_per_student cps ON cps.student_id = cs.student_id
LEFT JOIN recent_ef_movement ref ON ref.student_id = cs.student_id
LEFT JOIN quiz_avg qa ON qa.student_id = cs.student_id
LEFT JOIN interview_avg ia ON ia.student_id = cs.student_id
LEFT JOIN last_active la ON la.student_id = cs.student_id
WHERE
    (cps.due_total > 0 AND (cps.reviewed_in_window::float / cps.due_total) < 0.5)
    OR (ref.any_movement IS NULL OR ref.any_movement = false)
    OR (
        qa.avg_quiz_score IS NOT NULL
        AND ia.avg_interview_score IS NOT NULL
        AND (qa.avg_quiz_score - ia.avg_interview_score) > 30.0
    )
ORDER BY name;
