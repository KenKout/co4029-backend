-- T3.4: Published course content tree (student-facing).
--
-- Returns the course → modules → module_items tree filtered to PUBLISHED
-- only at every level (and respecting soft-delete via deleted_at IS NULL).
--
-- DRAFT_VISIBILITY rule (plan §4153):
-- A module item that points at a draft / soft-deleted lesson is EXCLUDED
-- entirely from the result, never returned with target=null. The
-- visible_items CTE filters them out before the json_agg roll-up.
--
-- Forward-reference branches (Phase 5 / Phase 6):
-- Items with item_type='quiz' or item_type='interview' are gated behind
-- AND FALSE so the tree shape stays consistent. T5.x / T6.x will swap
-- those branches for real predicates joining quizzes / interview_configs.
--
-- Bind parameters:
--   :course_id  UUID — course to fetch.
--
-- Returns a single row with three JSON columns:
--   course   — the published course (or NULL if not found / not published)
--   modules  — JSON array of published modules (ordered by position)
--   items    — JSON array of visible module_items (ordered by position),
--              each carrying its module_id and lesson payload
WITH course_root AS (
    SELECT
        c.id,
        c.organization_id,
        c.org_unit_id,
        c.owner_user_id,
        c.slug,
        c.title,
        c.description,
        c.status,
        c.level,
        c.thumbnail_object_id,
        c.estimated_minutes,
        c.expected_completion_days,
        c.enrollment_cap,
        c.created_at,
        c.updated_at
    FROM courses c
    WHERE c.id = :course_id
      AND c.status = 'published'
      AND c.deleted_at IS NULL
),
published_modules AS (
    SELECT
        m.id,
        m.course_id,
        m.title,
        m.description,
        m.position,
        m.status,
        m.estimated_minutes,
        m.requires_all_lessons_unlocked,
        m.created_at,
        m.updated_at
    FROM modules m
    JOIN course_root c ON m.course_id = c.id
    WHERE m.status = 'published'
      AND m.deleted_at IS NULL
),
candidate_items AS (
    SELECT
        mi.id,
        mi.module_id,
        mi.item_type,
        mi.lesson_id,
        mi.quiz_id,
        mi.interview_config_id,
        mi.position,
        mi.unlock_rule_json
    FROM module_items mi
    JOIN published_modules m ON mi.module_id = m.id
    WHERE mi.deleted_at IS NULL
),
published_lessons AS (
    SELECT
        l.id,
        l.module_id,
        l.slug,
        l.title,
        l.summary,
        l.notes_markdown,
        l.lesson_type,
        l.difficulty,
        l.estimated_minutes,
        l.status,
        l.ef_min_unlock,
        l.tau_unlock,
        l.requires_interview_pass,
        l.created_at,
        l.updated_at
    FROM lessons l
    JOIN published_modules m ON l.module_id = m.id
    WHERE l.status = 'published'
      AND l.deleted_at IS NULL
),
visible_items AS (
    SELECT
        i.id,
        i.module_id,
        i.item_type,
        i.lesson_id,
        i.quiz_id,
        i.interview_config_id,
        i.position,
        i.unlock_rule_json,
        l.id AS lesson_pub_id
    FROM candidate_items i
    LEFT JOIN published_lessons l ON l.id = i.lesson_id
    WHERE
        (i.item_type = 'lesson' AND l.id IS NOT NULL)
        -- Phase 5 swaps the FALSE branch for an EXISTS clause against
        -- published quizzes joined on i.quiz_id.
        OR (i.item_type = 'quiz' AND FALSE)
        -- Phase 6 swaps the FALSE branch for an EXISTS clause against
        -- published interview_configs joined on i.interview_config_id.
        OR (i.item_type = 'interview' AND FALSE)
)
SELECT
    (SELECT row_to_json(c) FROM course_root c) AS course,
    (
        SELECT COALESCE(json_agg(row_to_json(m) ORDER BY m.position), '[]'::json)
        FROM published_modules m
    ) AS modules,
    (
        SELECT COALESCE(
            json_agg(
                json_build_object(
                    'id', vi.id,
                    'module_id', vi.module_id,
                    'item_type', vi.item_type,
                    'lesson_id', vi.lesson_id,
                    'quiz_id', vi.quiz_id,
                    'interview_config_id', vi.interview_config_id,
                    'position', vi.position,
                    'unlock_rule_json', vi.unlock_rule_json,
                    'lesson', (
                        SELECT row_to_json(pl)
                        FROM published_lessons pl
                        WHERE pl.id = vi.lesson_id
                    )
                )
                ORDER BY vi.module_id, vi.position
            ),
            '[]'::json
        )
        FROM visible_items vi
    ) AS items;
