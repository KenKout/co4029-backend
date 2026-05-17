-- Knowledge-Retention estimate per (student, lesson).
--
-- Thesis §5.x:
--   R̂ = (1 / |C_l|) × Σ (EF_sc - 1.3) / (2.5 - 1.3)   for c in C_l
--
-- C_l = cards (quiz_questions) belonging to quizzes that source from the lesson
-- (link via ``quiz_source_lessons``). Cards never reviewed by the student have
-- no row in ``student_card_state`` and therefore no EF; the LEFT JOIN keeps them
-- in the denominator with a NULL EF, and ``COALESCE(EF, 1.3)`` clamps unseen
-- cards to the floor (R̂ contribution = 0). Empty lesson collapses to 0.0 via
-- the outer COALESCE.
--
-- Result is bounded in [0, 1] by the EF >= 1.3 / EF <= 2.5 invariants enforced
-- on ``student_card_state.ef`` (CHECK constraint) and ``update_ef`` clamp.
SELECT COALESCE(
    AVG((COALESCE(scs.ef, 1.3) - 1.3) / (2.5 - 1.3)),
    0.0
) AS r_hat
FROM quiz_questions qq
JOIN quizzes q ON q.id = qq.quiz_id
JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
LEFT JOIN student_card_state scs
    ON scs.question_id = qq.id
    AND scs.student_id = CAST(:student_id AS uuid)
WHERE qsl.lesson_id = CAST(:lesson_id AS uuid)
  AND qq.deleted_at IS NULL
  AND q.deleted_at IS NULL;
