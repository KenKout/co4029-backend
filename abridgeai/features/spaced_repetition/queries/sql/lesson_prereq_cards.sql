-- T7.5.6 (redesign): EF aggregation over a lesson's PREREQUISITE cards.
--
-- The unlock gate means "understand A to unlock B": lesson B's EF gate
-- counts the SM-2 cards of B's prerequisite lessons (both edge kinds —
-- ``lesson_prerequisites`` rows and ``module_prerequisites`` expansions,
-- both resolved by the caller into this id list), NOT B's own quiz. A
-- lesson's own quiz never gates its own lesson; it feeds the NEXT
-- lesson's gate. A lesson with no prerequisites therefore has no card
-- evidence to demand and opens by construction (total = 0).
--
-- Card traversal mirrors ``lesson_unlock.sql``: a quiz's cards belong to
-- a lesson when the quiz sources it (``quiz_source_lessons``) AND hangs
-- in that lesson's module (``module_items`` in the same module). Like the
-- single-lesson aggregate, learner-visibility rules are NOT applied (a
-- draft quiz still contributes so authors can preview the gate).
--
-- LEFT JOIN onto ``student_card_state`` so cards never reviewed by the
-- student count as EF=0 (i.e. blocking) — that is the intended
-- "learn A, take A's quiz, then B opens" progression pressure.
--
-- Bind parameters:
--   :student_id      UUID     — learner whose state we read.
--   :lesson_ids      UUID[]   — prerequisite lesson ids to aggregate over
--                               (empty list short-circuits in Python).
--   :ef_min          float    — EF threshold from the GATED lesson's
--                               ``ef_min_unlock`` (B's bar for A's cards).
--   :blocking_limit  int      — cap on returned blocking_card payload.
--
-- Returned columns: (passing BIGINT, total BIGINT, blocking JSONB).
WITH prereq_cards AS (
    SELECT DISTINCT
        qq.id AS question_id,
        qq.quiz_id,
        qq.source_refs,
        COALESCE(scs.ef::float8, 0.0) AS ef
    FROM quiz_questions qq
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    JOIN lessons l ON l.id = qsl.lesson_id
    JOIN module_items mi ON mi.quiz_id = q.id AND mi.module_id = l.module_id
    LEFT JOIN student_card_state scs
        ON scs.question_id = qq.id
        AND scs.student_id = :student_id
    WHERE qsl.lesson_id = ANY(:lesson_ids)
        AND qq.deleted_at IS NULL
        AND q.deleted_at IS NULL
        AND mi.deleted_at IS NULL
        AND l.deleted_at IS NULL
),
blocking_subset AS (
    SELECT question_id, quiz_id, source_refs, ef
    FROM prereq_cards
    WHERE ef < :ef_min
    ORDER BY ef ASC, question_id ASC
    LIMIT :blocking_limit
)
SELECT
    (SELECT COUNT(*) FROM prereq_cards WHERE ef >= :ef_min)::BIGINT AS passing,
    (SELECT COUNT(*) FROM prereq_cards)::BIGINT AS total,
    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'question_id', question_id,
                    'current_ef', ef,
                    'quiz_id', quiz_id,
                    'source_chunk_ids', COALESCE(source_refs, '[]'::jsonb)
                )
                ORDER BY ef ASC, question_id ASC
            )
            FROM blocking_subset
        ),
        '[]'::jsonb
    ) AS blocking;
