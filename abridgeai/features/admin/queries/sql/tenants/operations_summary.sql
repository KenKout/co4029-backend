-- Tenant operations summary for ONE organization (PRD ADM-042).
--
-- Organization detail showed Info / Domains / Units / Memberships — an
-- identity record with nothing operational in it. This adds what an operator
-- actually asks about a tenant: how much of it there is, what it is costing,
-- and whether its background work is healthy.
--
-- The interesting part is the job section. ``processing_jobs`` has no
-- organization column and its ``entity_id`` is polymorphic with no foreign
-- key, which is why platform-wide job aggregates are still global. For ONE
-- organization the walk runs in the other direction and is perfectly
-- tractable: collect the entity ids that belong to this tenant's courses, then
-- match jobs against them. ``ix_processing_jobs_entity_id`` covers the join.
--
-- That reverse walk is deliberately not generalised into a platform-wide
-- GROUP BY. It is bounded here because one organization's entity set is small;
-- across every tenant it would be a full cross-product and is the case that
-- genuinely needs a denormalized column.
--
-- :organization_id (uuid), :now (timestamptz), :window_days (int)
WITH bounds AS (
    SELECT
        CAST(:now AS timestamptz) AS as_of,
        CAST(:now AS timestamptz)
            - make_interval(days => CAST(:window_days AS int)) AS since
),
org_courses AS (
    SELECT c.id
    FROM courses c
    WHERE c.organization_id = CAST(:organization_id AS uuid)
      AND c.deleted_at IS NULL
),
-- Every entity this tenant's jobs could reference, tagged with the
-- entity_type processing_jobs records it under.
org_entities AS (
    SELECT lmv.id AS entity_id, 'material_version' AS entity_type
    FROM learning_material_versions lmv
    JOIN learning_materials lm ON lm.id = lmv.material_id
    JOIN lessons l ON l.id = lm.lesson_id
    JOIN modules m ON m.id = l.module_id
    JOIN org_courses oc ON oc.id = m.course_id

    UNION ALL
    SELECT l.id, 'lesson'
    FROM lessons l
    JOIN modules m ON m.id = l.module_id
    JOIN org_courses oc ON oc.id = m.course_id

    UNION ALL
    SELECT q.id, 'quiz'
    FROM quizzes q
    JOIN org_courses oc ON oc.id = q.course_id

    UNION ALL
    SELECT ic.id, 'interview_config'
    FROM interview_configs ic
    JOIN org_courses oc ON oc.id = ic.course_id

    UNION ALL
    SELECT gr.id, 'generation_run'
    FROM generation_runs gr
    JOIN org_courses oc ON oc.id = gr.course_id
),
org_jobs AS (
    SELECT pj.status, pj.updated_at
    FROM processing_jobs pj
    JOIN org_entities oe
      ON oe.entity_id = pj.entity_id
     AND oe.entity_type = pj.entity_type
)
SELECT
    b.as_of,
    ------------------------------------------------------------------ people
    (
        SELECT COUNT(*)
        FROM organization_memberships om
        JOIN users u ON u.id = om.user_id
        WHERE om.organization_id = CAST(:organization_id AS uuid)
          AND om.deleted_at IS NULL
          AND om.status = 'active'
          AND u.status = 'active'
    ) AS active_members,
    (
        SELECT COUNT(*)
        FROM organization_memberships om
        JOIN users u ON u.id = om.user_id
        WHERE om.organization_id = CAST(:organization_id AS uuid)
          AND om.deleted_at IS NULL
          AND u.last_login_at >= b.since
    ) AS members_active_in_window,
    --------------------------------------------------------------- inventory
    (SELECT COUNT(*) FROM org_courses) AS course_count,
    (
        SELECT COUNT(*)
        FROM org_courses oc
        JOIN courses c ON c.id = oc.id
        WHERE c.status = 'published'
    ) AS published_course_count,
    (
        SELECT COUNT(*)
        FROM learning_materials lm
        JOIN lessons l ON l.id = lm.lesson_id
        JOIN modules m ON m.id = l.module_id
        JOIN org_courses oc ON oc.id = m.course_id
        WHERE lm.deleted_at IS NULL
    ) AS material_count,
    (
        -- Bytes held by this tenant's material versions. NULL sizes are
        -- treated as 0 rather than poisoning the sum; a version with no
        -- recorded size is an ingest gap, not unbounded storage.
        SELECT COALESCE(SUM(so.size_bytes), 0)
        FROM learning_material_versions lmv
        JOIN learning_materials lm ON lm.id = lmv.material_id
        JOIN lessons l ON l.id = lm.lesson_id
        JOIN modules m ON m.id = l.module_id
        JOIN org_courses oc ON oc.id = m.course_id
        JOIN storage_objects so ON so.id = lmv.storage_object_id
    ) AS storage_bytes,
    ----------------------------------------------------------------- jobs
    (
        SELECT COUNT(*) FROM org_jobs j, bounds bb
        WHERE j.updated_at >= bb.since
          AND j.status IN ('completed', 'failed', 'cancelled')
    ) AS jobs_terminal_window,
    (
        SELECT COUNT(*) FROM org_jobs j, bounds bb
        WHERE j.updated_at >= bb.since AND j.status = 'failed'
    ) AS jobs_failed_window,
    (
        -- In-flight is as-of, not windowed: a job pending since last week is
        -- still in this tenant's queue now.
        SELECT COUNT(*) FROM org_jobs j
        WHERE j.status IN ('pending', 'running')
    ) AS jobs_in_flight,
    ------------------------------------------------------------ configuration
    (
        SELECT COUNT(*)
        FROM system_settings s
        WHERE s.organization_id = CAST(:organization_id AS uuid)
    ) AS config_overrides
FROM bounds b;
