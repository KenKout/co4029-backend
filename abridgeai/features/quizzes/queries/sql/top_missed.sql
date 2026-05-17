SELECT
    qq.id AS question_id,
    qq.prompt_text AS prompt,
    AVG(CASE WHEN qaa.is_correct THEN 1.0 ELSE 0.0 END) AS correctness_rate,
    COUNT(qaa.id) AS attempt_count
FROM quiz_questions qq
JOIN quizzes q ON q.id = qq.quiz_id
JOIN quiz_attempt_answers qaa ON qaa.question_id = qq.id
JOIN quiz_attempts qa ON qa.id = qaa.attempt_id
WHERE q.course_id = :course_id
  AND qq.deleted_at IS NULL
  AND q.deleted_at IS NULL
  AND qa.status IN ('submitted', 'graded')
GROUP BY qq.id, qq.prompt_text
HAVING COUNT(qaa.id) >= 5
ORDER BY correctness_rate ASC, qq.id ASC
LIMIT :limit
