-- Per-stage rollup of the AI calls one job made (PRD ADM-014).
--
-- Answers "where did this pipeline spend its time and money", which is the
-- first question asked of a slow or expensive job. The companion
-- ``job_ai_calls.sql`` answers the second one -- which individual call failed.
--
-- ``pipeline_stage`` was renamed to ``stage_name`` in migration 0005, so only
-- ``stage_name`` exists. Falls back to ``role`` so session-runtime calls, which
-- attribute via role alone, still appear in the totals instead of vanishing
-- out of a rollup that claims to cover the job.
--
-- :job_id (uuid)
SELECT
    COALESCE(amc.stage_name, amc.role, 'unknown')  AS stage,
    COUNT(*)                                       AS call_count,
    COUNT(*) FILTER (WHERE amc.status = 'failed')  AS failed_count,
    COALESCE(SUM(amc.estimated_cost_usd), 0)       AS spend_usd,
    COALESCE(SUM(amc.total_tokens), 0)             AS tokens,
    -- NULL when nothing in the stage recorded a latency, rather than 0.
    MAX(amc.latency_ms)                            AS max_latency_ms
FROM ai_model_calls amc
WHERE amc.processing_job_id = CAST(:job_id AS uuid)
   OR amc.generation_run_id = CAST(:job_id AS uuid)
GROUP BY COALESCE(amc.stage_name, amc.role, 'unknown')
ORDER BY spend_usd DESC NULLS LAST, call_count DESC;
