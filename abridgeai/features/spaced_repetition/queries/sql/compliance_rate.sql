-- Review compliance rate per (student, lesson) within grace window.
--
-- Thesis §5.x:
--   ρ = count(reviewed within due_at + grace) / count(due_at <= NOW())
--
-- "Reviewed within window" means a CardReview row exists for the card whose
-- created_at falls in [due_at, due_at + grace_window]. We materialise the set
-- of due cards once (CTE) and probe ``card_reviews`` per card via EXISTS so
-- the planner can short-circuit on the (student_id, question_id) composite
-- index covering ``ix_card_reviews_student_created``.
--
-- Card scope: quiz_questions sourced from the lesson (via quiz_source_lessons),
-- restricted to cards the student has actually started (row in
-- ``student_card_state``) AND with due_at <= NOW(). Cards that aren't due
-- yet are excluded from both numerator and denominator.
--
-- :grace_window_seconds is bound as plain integer; we convert to interval in
-- SQL to avoid driver-specific INTERVAL parameter quirks.
WITH due_cards AS (
    SELECT scs.question_id, scs.due_at
    FROM student_card_state scs
    JOIN quiz_questions qq ON qq.id = scs.question_id
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    WHERE scs.student_id = CAST(:student_id AS uuid)
      AND qsl.lesson_id = CAST(:lesson_id AS uuid)
      AND scs.due_at <= NOW()
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
)
SELECT
    COUNT(*) FILTER (
        WHERE EXISTS (
            SELECT 1
            FROM card_reviews cr
            WHERE cr.student_id = CAST(:student_id AS uuid)
              AND cr.question_id = due_cards.question_id
              AND cr.created_at >= due_cards.due_at
              AND cr.created_at <= due_cards.due_at + (CAST(:grace_window_seconds AS integer) * INTERVAL '1 second')
        )
    ) AS reviewed_in_window,
    COUNT(*) AS due_total
FROM due_cards;
