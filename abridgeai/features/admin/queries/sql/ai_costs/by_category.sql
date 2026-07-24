-- AI cost grouped by a single caller-chosen dimension (T0.27 enhancement).
--
-- The grouping column is NOT a bind parameter (Postgres cannot bind an
-- identifier); the query layer validates the dimension name against a fixed
-- allowlist and substitutes the real column name into {dimension_col} before
-- executing. Never interpolate raw user input here.
--
-- Bucketing/context columns available for grouping: operation, role, tier,
-- stage_name, model_name, status. Each row is one distinct value of that
-- column with rolled-up spend, tokens (split into input/output/cached), and
-- call count.
--
-- :since   timestamptz lower bound on called_at (required to bound scans).
-- :top_n   row cap.
WITH bounded AS (
    SELECT
        COALESCE(CAST({dimension_col} AS text), 'unknown') AS dimension_value,
        amc.estimated_cost_usd,
        amc.total_tokens,
        amc.input_tokens,
        amc.output_tokens,
        amc.cached_input_tokens
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
      AND (CAST(:f_model AS text) IS NULL OR amc.model_name = CAST(:f_model AS text))
      AND (CAST(:f_role AS text) IS NULL OR amc.role = CAST(:f_role AS text))
      AND (CAST(:f_operation AS text) IS NULL OR amc.operation = CAST(:f_operation AS text))
      AND (CAST(:f_status AS text) IS NULL OR amc.status = CAST(:f_status AS text))
)
SELECT
    dimension_value AS dimension_value,
    COUNT(*)::bigint AS call_count,
    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
    COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
    COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
    COALESCE(SUM(cached_input_tokens), 0)::bigint AS cached_tokens,
    COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS total_usd
FROM bounded
GROUP BY dimension_value
ORDER BY total_usd DESC, call_count DESC
LIMIT :top_n;
