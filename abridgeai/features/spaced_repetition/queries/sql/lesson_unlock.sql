-- T7.5.6: Lesson unlock EF gate aggregation (single shot, indexed).
--
-- Aggregates the SM-2 EasinessFactor (EF) state across every quiz card
-- attached to a lesson and returns the four primitives the unlock-gate
-- needs:
--
--   * passing  — count of cards whose stored EF >= :ef_min
--   * total    — count of cards in the lesson (NOT NULL, may be 0)
--   * blocking — JSONB array describing every card whose EF is below
--                :ef_min (limited to :blocking_limit rows; the
--                pre-filter LIMITS the inner CTE so jsonb_agg only walks
--                the truncated set).
--
-- Lesson → cards traversal:
-- Per baseline schema (T3.x), quizzes attach to a lesson via the
-- ``module_items`` table where ``item_type = 'quiz'`` and the parent
-- module also hosts the lesson row. The intermediary join is therefore:
--
--     lessons l
--       └─ modules m            (l.module_id = m.id)
--            └─ module_items mi (mi.module_id = m.id, mi.quiz_id NOT NULL)
--                 └─ quizzes q  (q.id = mi.quiz_id)
--                      └─ quiz_questions qq (qq.quiz_id = q.id)
--
-- Soft-delete: every layer respects ``deleted_at IS NULL`` and the
-- learner-visibility rules are NOT applied here (the unlock gate cares
-- about authoring-visibility, not learner-publish — a quiz being draft
-- still contributes to the EF gate so authors can preview the gate
-- behaviour).
--
-- LEFT JOIN onto ``student_card_state`` so cards never reviewed by the
-- student count as EF=0 (i.e. blocking).
--
-- Bind parameters:
--   :student_id      UUID  — learner whose state we read.
--   :lesson_id       UUID  — lesson whose cards we aggregate.
--   :ef_min          float — EF threshold from lesson.ef_min_unlock.
--   :blocking_limit  int   — cap on returned blocking_card payload size.
--
-- Returned columns: (passing BIGINT, total BIGINT, blocking JSONB).
WITH lesson_cards AS (
    SELECT
        qq.id AS question_id,
        qq.quiz_id,
        qq.source_refs,
        COALESCE(scs.ef::float8, 0.0) AS ef
    FROM quiz_questions qq
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN module_items mi ON mi.quiz_id = q.id
    JOIN modules m ON m.id = mi.module_id
    JOIN lessons l ON l.module_id = m.id
    LEFT JOIN student_card_state scs
        ON scs.question_id = qq.id
        AND scs.student_id = :student_id
    WHERE l.id = :lesson_id
        AND qq.deleted_at IS NULL
        AND q.deleted_at IS NULL
        AND mi.deleted_at IS NULL
        AND m.deleted_at IS NULL
        AND l.deleted_at IS NULL
),
blocking_subset AS (
    SELECT question_id, quiz_id, source_refs, ef
    FROM lesson_cards
    WHERE ef < :ef_min
    ORDER BY ef ASC, question_id ASC
    LIMIT :blocking_limit
)
SELECT
    (SELECT COUNT(*) FROM lesson_cards WHERE ef >= :ef_min)::BIGINT AS passing,
    (SELECT COUNT(*) FROM lesson_cards)::BIGINT AS total,
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
