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
        amc.input_tokens,
        amc.output_tokens,
        amc.cached_input_tokens,
        amc.status,
        amc.called_at
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
      AND (CAST(:f_model AS text) IS NULL OR amc.model_name = CAST(:f_model AS text))
      AND (CAST(:f_role AS text) IS NULL OR amc.role = CAST(:f_role AS text))
      AND (CAST(:f_operation AS text) IS NULL OR amc.operation = CAST(:f_operation AS text))
      AND (CAST(:f_status AS text) IS NULL OR amc.status = CAST(:f_status AS text))
),
failed AS (
    SELECT
        COUNT(*)::bigint AS failed_call_count,
        COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS failed_usd
    FROM bounded
    WHERE status = 'failed'
),
totals AS (
    SELECT
        COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
        COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
        COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
        COALESCE(SUM(cached_input_tokens), 0)::bigint AS cached_tokens,
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
    -- Gap-filled series. Grouping the rows alone omitted zero-spend days
    -- entirely, so the x-axis jumped (e.g. Jun 28 -> Jul 12), compressing the
    -- timeline and making later spikes look sharper than they were. Generate
    -- every bucket in range and LEFT JOIN the aggregates so quiet days plot as
    -- an explicit 0.
    SELECT
        gs.bucket_start_ts,
        COALESCE(agg.tokens, 0)::bigint AS tokens,
        COALESCE(agg.usd, 0)::numeric(18, 6) AS usd
    FROM generate_series(
        date_trunc(:period, CAST(:since AS timestamptz)),
        date_trunc(:period, NOW()),
        CASE :period
            WHEN 'week' THEN INTERVAL '1 week'
            WHEN 'month' THEN INTERVAL '1 month'
            ELSE INTERVAL '1 day'
        END
    ) AS gs(bucket_start_ts)
    LEFT JOIN (
        SELECT
            date_trunc(:period, called_at) AS bucket_start_ts,
            COALESCE(SUM(total_tokens), 0)::bigint AS tokens,
            COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS usd
        FROM bounded
        GROUP BY date_trunc(:period, called_at)
    ) AS agg ON agg.bucket_start_ts = gs.bucket_start_ts
    ORDER BY gs.bucket_start_ts ASC
)
SELECT
    (SELECT total_tokens FROM totals) AS total_tokens,
    (SELECT input_tokens FROM totals) AS input_tokens,
    (SELECT output_tokens FROM totals) AS output_tokens,
    (SELECT cached_tokens FROM totals) AS cached_tokens,
    (SELECT total_usd FROM totals) AS total_usd,
    (SELECT call_count FROM totals) AS call_count,
    (SELECT failed_call_count FROM failed) AS failed_call_count,
    (SELECT failed_usd FROM failed) AS failed_usd,
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
