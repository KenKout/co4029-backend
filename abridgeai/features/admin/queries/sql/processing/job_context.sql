-- Everything about one job that is NOT on the processing_jobs row itself:
-- who it belongs to, how long it waited, how long it ran (PRD ADM-013).
--
-- Ownership is resolved HERE, at read time, rather than read from a
-- denormalized column. ``processing_jobs.entity_id`` is polymorphic with no
-- foreign key, so a stored organization_id would have to be populated by every
-- enqueue site and would silently rot the first time one forgot. For a single
-- job detail the walk is five indexed lookups and cannot drift.
--
-- The five paths, one per entity_type:
--   material_version  -> learning_material_versions -> learning_materials
--                        -> lessons -> modules -> courses
--   lesson            -> lessons -> modules -> courses
--   quiz              -> quizzes -> courses
--   interview_config  -> interview_configs -> courses
--   generation_run    -> generation_runs.course_id -> courses  (nullable:
--                        a run can be scoped to a lesson or module instead,
--                        and course_id is then NULL — the caller renders that
--                        as unknown rather than inventing an owner)
--
-- Timings are derived, not stored:
--   queue_wait_seconds -- created_at -> started_at. The number that says
--                         whether a slow job was slow or merely queued behind
--                         other work, which duration alone cannot answer.
--   duration_seconds   -- started_at -> finished_at, or -> :now while running,
--                         so an in-flight job reports how long it has been
--                         going instead of NULL.
--
-- :job_id (uuid), :now (timestamptz)
WITH job AS (
    SELECT pj.id, pj.entity_type, pj.entity_id, pj.created_at,
           pj.started_at, pj.finished_at
    FROM processing_jobs pj
    WHERE pj.id = CAST(:job_id AS uuid)
),
resolved AS (
    SELECT
        j.id,
        CASE j.entity_type
            WHEN 'material_version' THEN (
                SELECT m.course_id
                FROM learning_material_versions lmv
                JOIN learning_materials lm ON lm.id = lmv.material_id
                JOIN lessons l ON l.id = lm.lesson_id
                JOIN modules m ON m.id = l.module_id
                WHERE lmv.id = j.entity_id
            )
            WHEN 'lesson' THEN (
                SELECT m.course_id
                FROM lessons l
                JOIN modules m ON m.id = l.module_id
                WHERE l.id = j.entity_id
            )
            WHEN 'quiz' THEN (
                SELECT q.course_id FROM quizzes q WHERE q.id = j.entity_id
            )
            WHEN 'interview_config' THEN (
                SELECT ic.course_id
                FROM interview_configs ic
                WHERE ic.id = j.entity_id
            )
            WHEN 'generation_run' THEN (
                SELECT gr.course_id
                FROM generation_runs gr
                WHERE gr.id = j.entity_id
            )
        END AS course_id
    FROM job j
)
SELECT
    j.id,
    r.course_id,
    c.title            AS course_title,
    c.slug             AS course_slug,
    c.organization_id  AS organization_id,
    o.name             AS organization_name,
    -- Whole seconds: sub-second precision on a queue wait is noise, and the
    -- UI renders these as coarse durations anyway.
    CASE
        WHEN j.started_at IS NULL THEN NULL
        ELSE EXTRACT(EPOCH FROM (j.started_at - j.created_at))::bigint
    END AS queue_wait_seconds,
    CASE
        WHEN j.started_at IS NULL THEN NULL
        ELSE EXTRACT(
                 EPOCH FROM (
                     COALESCE(j.finished_at, CAST(:now AS timestamptz))
                     - j.started_at
                 )
             )::bigint
    END AS duration_seconds,
    -- NULL, not 0, while the job has not started: "has not run" and "ran for
    -- no time" are different states.
    (j.finished_at IS NULL AND j.started_at IS NOT NULL) AS is_running
FROM job j
JOIN resolved r ON r.id = j.id
LEFT JOIN courses c ON c.id = r.course_id
LEFT JOIN organizations o ON o.id = c.organization_id
;
