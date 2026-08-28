-- Quiz outcome + activity metrics per course, for the Course Health table.
--
-- Pass rate is computed over each student's BEST attempt per quiz, not
-- over all attempts. Counting every attempt lets a course that encourages
-- retries look worse than one that forbids them, which inverts the signal:
-- the question a teacher is asking is "are my students getting there",
-- not "how many tries did it take".
--
-- Only published quizzes count. A draft quiz's attempts are the author's
-- own testing, and folding those into the cohort's pass rate would let a
-- teacher move the number by failing their own draft.
--
-- ``pass_sample`` is returned alongside the rate so the UI can withhold a
-- percentage computed from three attempts. A rate without its denominator
-- is the failure mode FR-054 calls out.
WITH best_attempt AS (
    SELECT DISTINCT ON (q.course_id, qa.quiz_id, qa.student_id)
        q.course_id AS course_id,
        qa.passed AS passed
    FROM quiz_attempts qa
    JOIN quizzes q ON q.id = qa.quiz_id
    WHERE q.course_id = ANY(:course_ids)
      AND q.status = 'published'
      AND q.deleted_at IS NULL
      AND qa.status IN ('submitted', 'graded')
      AND qa.score_percent IS NOT NULL
    ORDER BY q.course_id, qa.quiz_id, qa.student_id, qa.score_percent DESC
),
pass_rates AS (
    SELECT
        course_id,
        COUNT(*) AS pass_sample,
        AVG(CASE WHEN passed THEN 1.0 ELSE 0.0 END) * 100 AS pass_rate_percent
    FROM best_attempt
    GROUP BY course_id
),
-- Last activity spans both arms of "the course is alive": reading the
-- material and sitting the assessments. Taking only one would call a
-- revision-heavy course dormant.
engagement_activity AS (
    SELECT m.course_id AS course_id, MAX(me.created_at) AS last_at
    FROM material_engagement me
    JOIN learning_material_versions lmv ON lmv.id = me.material_version_id
    JOIN learning_materials lm ON lm.id = lmv.material_id
    JOIN lessons l ON l.id = lm.lesson_id
    JOIN modules m ON m.id = l.module_id
    WHERE m.course_id = ANY(:course_ids)
    GROUP BY m.course_id
),
attempt_activity AS (
    SELECT q.course_id AS course_id, MAX(qa.submitted_at) AS last_at
    FROM quiz_attempts qa
    JOIN quizzes q ON q.id = qa.quiz_id
    WHERE q.course_id = ANY(:course_ids)
      AND qa.submitted_at IS NOT NULL
    GROUP BY q.course_id
),
activity AS (
    SELECT course_id, MAX(last_at) AS last_activity_at
    FROM (
        SELECT course_id, last_at FROM engagement_activity
        UNION ALL
        SELECT course_id, last_at FROM attempt_activity
    ) merged
    GROUP BY course_id
)
SELECT
    c.id AS course_id,
    pr.pass_rate_percent,
    COALESCE(pr.pass_sample, 0) AS pass_sample,
    a.last_activity_at
FROM courses c
LEFT JOIN pass_rates pr ON pr.course_id = c.id
LEFT JOIN activity a ON a.course_id = c.id
WHERE c.id = ANY(:course_ids)
