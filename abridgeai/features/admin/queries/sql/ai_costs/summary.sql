-- AI cost summary (T0.27): totals + breakdowns by role / stage / time bucket.
--
-- Aggregates ai_model_calls within [:since, NOW()]. Bucketing is driven by
-- :period (one of 'day', 'week', 'month'). Caller normalises invalid values
-- before binding.
--
-- Returns one row containing several aggregates as JSON arrays so we can
-- avoid N round-trips. The router/service flattens these back into typed
-- response models.
--
-- :since   timestamptz lower bound on called_at (required to bound scans).
-- :period  one of 'day' | 'week' | 'month' (validated upstream).
WITH bounded AS (
    SELECT
        amc.role,
        amc.stage_name,
        amc.estimated_cost_usd,
        amc.total_tokens,
        amc.called_at
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
),
totals AS (
    SELECT
        COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
        COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS total_usd,
        COUNT(*)::bigint AS call_count
    FROM bounded
),
by_role AS (
    SELECT
        COALESCE(role, 'unknown') AS role,
        COALESCE(SUM(total_tokens), 0)::bigint AS tokens,
        COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS usd
    FROM bounded
    GROUP BY COALESCE(role, 'unknown')
    ORDER BY usd DESC, role ASC
),
by_stage AS (
    SELECT
        COALESCE(stage_name, 'unknown') AS stage_name,
        COALESCE(SUM(total_tokens), 0)::bigint AS tokens,
        COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS usd
    FROM bounded
    GROUP BY COALESCE(stage_name, 'unknown')
    ORDER BY usd DESC, stage_name ASC
),
buckets AS (
    SELECT
        date_trunc(:period, called_at) AS bucket_start_ts,
        COALESCE(SUM(total_tokens), 0)::bigint AS tokens,
        COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS usd
    FROM bounded
    GROUP BY date_trunc(:period, called_at)
    ORDER BY bucket_start_ts ASC
)
SELECT
    (SELECT total_tokens FROM totals) AS total_tokens,
    (SELECT total_usd FROM totals) AS total_usd,
    (SELECT call_count FROM totals) AS call_count,
    COALESCE(
        (SELECT jsonb_agg(jsonb_build_object(
            'role', role, 'tokens', tokens, 'usd', usd
        ) ORDER BY usd DESC, role ASC) FROM by_role),
        '[]'::jsonb
    ) AS by_role,
    COALESCE(
        (SELECT jsonb_agg(jsonb_build_object(
            'stage_name', stage_name, 'tokens', tokens, 'usd', usd
        ) ORDER BY usd DESC, stage_name ASC) FROM by_stage),
        '[]'::jsonb
    ) AS by_stage,
    COALESCE(
        (SELECT jsonb_agg(jsonb_build_object(
            'bucket_start_ts', bucket_start_ts, 'tokens', tokens, 'usd', usd
        ) ORDER BY bucket_start_ts ASC) FROM buckets),
        '[]'::jsonb
    ) AS buckets;
