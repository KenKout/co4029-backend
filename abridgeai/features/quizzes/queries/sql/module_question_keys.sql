WITH siblings AS (
    SELECT q.id AS quiz_id
    FROM quizzes q
    WHERE q.deleted_at IS NULL
      AND q.module_id = (
          SELECT module_id FROM quizzes WHERE id = :quiz_id AND deleted_at IS NULL
      )
)
SELECT
    encode(
        sha256(
            (
                qq.prompt_text
                || md5(coalesce(qq.source_refs::text, '[]'))
            )::bytea
        ),
        'hex'
    ) AS question_key
FROM quiz_questions qq
JOIN siblings s ON s.quiz_id = qq.quiz_id
WHERE qq.deleted_at IS NULL
