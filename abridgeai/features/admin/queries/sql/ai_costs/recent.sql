-- Recent expensive ai_model_calls (T0.27) sorted by cost DESC then time DESC.
-- :limit   row cap; caller validates 1..500.
SELECT
    amc.id AS id,
    amc.role AS role,
    amc.tier AS tier,
    amc.stage_name AS stage_name,
    amc.model_name AS model,
    amc.total_tokens AS tokens,
    amc.estimated_cost_usd AS usd,
    amc.latency_ms AS latency_ms,
    amc.called_at AS called_at,
    amc.pipeline_run_id AS pipeline_run_id
FROM ai_model_calls amc
ORDER BY amc.estimated_cost_usd DESC NULLS LAST, amc.called_at DESC
LIMIT :limit;
