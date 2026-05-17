-- Class-wide R̂ histogram per (course, lesson).
--
-- For every active student in the course, compute the same R̂ formula as
-- kr_estimate.sql (per-card EF normalised to [0,1], averaged across the
-- lesson's cards), then bucket into 10 equal-width bins [0.0, 0.1) ..
-- [0.9, 1.0]. Bin upper bound 1.0 is included in the last bucket.
--
-- ``mean_kr`` and ``median_kr`` are computed over the same per-student R̂
-- distribution (PERCENTILE_CONT for the linearly interpolated median, since
-- the discrete distribution is small enough that a continuous interpolation
-- is the cheaper code path than width_bucket sampling).
WITH lesson_cards AS (
    SELECT qq.id AS question_id
    FROM quiz_questions qq
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    WHERE qsl.lesson_id = CAST(:lesson_id AS uuid)
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
),
class_students AS (
    SELECT student_id
    FROM course_enrollments
    WHERE course_id = CAST(:course_id AS uuid)
      AND status = 'active'
),
per_student_kr AS (
    SELECT
        cs.student_id,
        COALESCE(
            AVG((COALESCE(scs.ef, 1.3) - 1.3) / (2.5 - 1.3)),
            0.0
        ) AS r_hat
    FROM class_students cs
    CROSS JOIN lesson_cards lc
    LEFT JOIN student_card_state scs
        ON scs.student_id = cs.student_id
        AND scs.question_id = lc.question_id
    GROUP BY cs.student_id
),
buckets AS (
    SELECT
        LEAST(FLOOR(r_hat * 10)::int, 9) AS bucket_idx
    FROM per_student_kr
),
histogram AS (
    SELECT
        bucket_idx,
        COUNT(*) AS bucket_count
    FROM buckets
    GROUP BY bucket_idx
)
SELECT
    (SELECT COUNT(*) FROM per_student_kr) AS student_count,
    COALESCE((SELECT AVG(r_hat) FROM per_student_kr), 0.0) AS mean_kr,
    COALESCE(
        (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r_hat) FROM per_student_kr),
        0.0
    ) AS median_kr,
    COALESCE(
        (
            SELECT json_agg(
                json_build_object(
                    'bucket_lower', b.bucket_lower,
                    'count', COALESCE(h.bucket_count, 0)
                )
                ORDER BY b.bucket_lower
            )
            FROM (
                SELECT generate_series(0, 9) AS bucket_idx,
                       generate_series(0, 9) * 0.1 AS bucket_lower
            ) b
            LEFT JOIN histogram h USING (bucket_idx)
        ),
        '[]'::json
    ) AS histogram_json;
