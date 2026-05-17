-- AI cost by user (T0.27): top spenders since :since.
--
-- Walks two attribution paths because ai_model_calls may be parented by
-- either generation_runs (pipelines that own a run) or processing_jobs
-- (one-off worker tasks):
--   path A: amc.processing_job_id -> processing_jobs.entity_id
--           -> generation_runs.requested_by (entity_type='generation_run')
--   path B: amc.generation_run_id -> generation_runs.requested_by directly
-- We coalesce both paths in a single CTE so each call attributes to at most
-- one user.
--
-- :since   timestamptz lower bound on called_at.
-- :top_n   row cap (caller validates 1..200).
WITH bounded AS (
    SELECT
        amc.id,
        amc.generation_run_id,
        amc.processing_job_id,
        amc.estimated_cost_usd,
        amc.total_tokens
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
),
attributed AS (
    SELECT
        b.id,
        b.estimated_cost_usd,
        b.total_tokens,
        COALESCE(gr_direct.requested_by, gr_via_job.requested_by) AS user_id
    FROM bounded b
    LEFT JOIN generation_runs gr_direct
           ON gr_direct.id = b.generation_run_id
    LEFT JOIN processing_jobs pj
           ON pj.id = b.processing_job_id
          AND pj.entity_type = 'generation_run'
    LEFT JOIN generation_runs gr_via_job
           ON gr_via_job.id = pj.entity_id
)
SELECT
    a.user_id AS user_id,
    COALESCE(up.display_name, u.primary_email, '') AS display_name,
    COUNT(*)::bigint AS call_count,
    COALESCE(SUM(a.total_tokens), 0)::bigint AS total_tokens,
    COALESCE(SUM(a.estimated_cost_usd), 0)::numeric(18, 6) AS total_usd
FROM attributed a
LEFT JOIN users u ON u.id = a.user_id
LEFT JOIN user_profiles up ON up.user_id = a.user_id
WHERE a.user_id IS NOT NULL
GROUP BY a.user_id, up.display_name, u.primary_email
ORDER BY total_usd DESC, call_count DESC
LIMIT :top_n;
