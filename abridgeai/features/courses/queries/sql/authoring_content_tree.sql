-- T3.4: Authoring course content tree (teacher-facing).
--
-- Mirrors the shape of `published_content_tree.sql` but does NOT apply the
-- ``status = 'published'`` filter — draft + archived rows are visible to the
-- author. Soft-delete is still honoured (`deleted_at IS NULL`) because
-- soft-deleted rows must never resurface in any UI surface.
--
-- ``:include_archived`` toggle (plan §4119, §4122):
--   FALSE (default) → exclude archived courses, modules, lessons.
--   TRUE            → include them.
-- Drafts are ALWAYS included (the whole point of the authoring view).
--
-- Bind parameters:
--   :course_id          UUID    — course to fetch.
--   :include_archived   BOOLEAN — see toggle above.
--
-- Returns a single row with three JSON columns: course / modules / items
-- (same shape as the published tree so service layer can compose either).
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
      AND c.deleted_at IS NULL
      AND (:include_archived OR c.status <> 'archived')
),
authoring_modules AS (
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
    WHERE m.deleted_at IS NULL
      AND (:include_archived OR m.status <> 'archived')
),
authoring_items AS (
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
    JOIN authoring_modules m ON mi.module_id = m.id
    WHERE mi.deleted_at IS NULL
),
authoring_lessons AS (
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
    JOIN authoring_modules m ON l.module_id = m.id
    WHERE l.deleted_at IS NULL
      AND (:include_archived OR l.status <> 'archived')
)
SELECT
    (SELECT row_to_json(c) FROM course_root c) AS course,
    (
        SELECT COALESCE(json_agg(row_to_json(m) ORDER BY m.position), '[]'::json)
        FROM authoring_modules m
    ) AS modules,
    (
        SELECT COALESCE(
            json_agg(
                json_build_object(
                    'id', ai.id,
                    'module_id', ai.module_id,
                    'item_type', ai.item_type,
                    'lesson_id', ai.lesson_id,
                    'quiz_id', ai.quiz_id,
                    'interview_config_id', ai.interview_config_id,
                    'position', ai.position,
                    'unlock_rule_json', ai.unlock_rule_json,
                    'lesson', (
                        SELECT row_to_json(al)
                        FROM authoring_lessons al
                        WHERE al.id = ai.lesson_id
                    )
                )
                ORDER BY ai.module_id, ai.position
            ),
            '[]'::json
        )
        FROM authoring_items ai
    ) AS items;
