-- AI cost + efficiency grouped by model_name (T0.27 Tier 3 enhancement).
--
-- One row per model with rolled-up spend, tokens, call count, latency
-- percentiles (p50/p95 via percentile_cont), and the ACTUAL blended
-- cost-per-1k-tokens (total_usd / total_tokens * 1000). The last column lets
-- an operator spot models that are expensive per unit of work, independent of
-- the admin-configured pricing table.
--
-- Shares the same optional NULL-safe filter binds as summary/by_category.
--
-- :since   timestamptz lower bound on called_at (required to bound scans).
-- :top_n   row cap.
WITH bounded AS (
    SELECT
        COALESCE(amc.model_name, 'unknown') AS model_name,
        amc.estimated_cost_usd,
        amc.total_tokens,
        amc.latency_ms
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
      AND (CAST(:f_model AS text) IS NULL OR amc.model_name = CAST(:f_model AS text))
      AND (CAST(:f_role AS text) IS NULL OR amc.role = CAST(:f_role AS text))
      AND (CAST(:f_operation AS text) IS NULL OR amc.operation = CAST(:f_operation AS text))
      AND (CAST(:f_status AS text) IS NULL OR amc.status = CAST(:f_status AS text))
)
SELECT
    model_name AS model_name,
    COUNT(*)::bigint AS call_count,
    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
    COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS total_usd,
    COALESCE(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms), 0
    )::bigint AS latency_p50_ms,
    COALESCE(
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0
    )::bigint AS latency_p95_ms,
    CASE
        WHEN COALESCE(SUM(total_tokens), 0) > 0
        THEN (SUM(estimated_cost_usd) / SUM(total_tokens) * 1000)::numeric(18, 6)
        ELSE 0
    END AS usd_per_1k_tokens
FROM bounded
GROUP BY model_name
ORDER BY total_usd DESC, call_count DESC
LIMIT :top_n;
