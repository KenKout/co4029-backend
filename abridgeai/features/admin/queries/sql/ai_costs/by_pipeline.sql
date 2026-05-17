-- AI cost by pipeline_run_id (T0.27).
--
-- Groups ai_model_calls by pipeline_run_id (the umbrella id assigned by a
-- pipeline driver, distinct from generation_run_id which is a per-step run
-- and may be NULL for raw worker tasks). Joins generation_runs for context
-- (course_id, generation_type) by matching pipeline_run_id against the most
-- recent generation_run that shares a row in the same audit; we use
-- amc.generation_run_id as the bridge.
--
-- Stages breakdown is built as a JSONB array per pipeline so we can return
-- a one-row-per-pipeline shape without a second query.
--
-- :since   timestamptz lower bound on called_at.
-- :top_n   row cap.
WITH bounded AS (
    SELECT
        amc.pipeline_run_id,
        amc.generation_run_id,
        amc.stage_name,
        amc.estimated_cost_usd,
        amc.total_tokens,
        amc.called_at
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
      AND amc.pipeline_run_id IS NOT NULL
),
per_pipeline AS (
    SELECT
        b.pipeline_run_id,
        COUNT(*)::bigint AS call_count,
        COALESCE(SUM(b.total_tokens), 0)::bigint AS total_tokens,
        COALESCE(SUM(b.estimated_cost_usd), 0)::numeric(18, 6) AS total_usd,
        MIN(b.called_at) AS started_at
    FROM bounded b
    GROUP BY b.pipeline_run_id
),
per_pipeline_stages AS (
    SELECT
        b.pipeline_run_id,
        jsonb_agg(jsonb_build_object(
            'stage_name', COALESCE(b.stage_name, 'unknown'),
            'tokens', tokens,
            'usd', usd
        ) ORDER BY usd DESC, b.stage_name ASC) AS stages_breakdown
    FROM (
        SELECT
            pipeline_run_id,
            stage_name,
            COALESCE(SUM(total_tokens), 0)::bigint AS tokens,
            COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS usd
        FROM bounded
        GROUP BY pipeline_run_id, stage_name
    ) b
    GROUP BY b.pipeline_run_id
),
per_pipeline_run_ref AS (
    SELECT DISTINCT ON (b.pipeline_run_id)
        b.pipeline_run_id,
        gr.generation_type,
        gr.course_id
    FROM bounded b
    LEFT JOIN generation_runs gr ON gr.id = b.generation_run_id
    ORDER BY b.pipeline_run_id, gr.created_at DESC NULLS LAST
)
SELECT
    p.pipeline_run_id AS pipeline_run_id,
    r.generation_type AS generation_type,
    r.course_id AS course_id,
    p.started_at AS started_at,
    p.call_count AS call_count,
    p.total_tokens AS total_tokens,
    p.total_usd AS total_usd,
    COALESCE(s.stages_breakdown, '[]'::jsonb) AS stages_breakdown
FROM per_pipeline p
LEFT JOIN per_pipeline_stages s ON s.pipeline_run_id = p.pipeline_run_id
LEFT JOIN per_pipeline_run_ref r ON r.pipeline_run_id = p.pipeline_run_id
ORDER BY p.total_usd DESC, p.call_count DESC
LIMIT :top_n;
