-- Daily API latency trend: per-calendar-day p50 / p95 latency and request
-- volume, from the same source as api_reliability.sql (http_audit_log —
-- written by the audit middleware for every non-exempt request).
--
-- Contract mirrors top-of-file conventions there: percentiles are NULL on a
-- zero-traffic day (the caller renders a flat line, not a fabricated 0ms),
-- and one row is returned per day INCLUDING zero-traffic days so the chart
-- is continuous.
--
-- Windowed by explicit bounds rather than a day count, so the chart plots
-- exactly the span the page filter selected. A day count cannot express a
-- range that ENDS in the past — "Aug 1–Aug 8" would have been rendered as
-- "the last 8 days", quietly plotting up to today instead. The bounds come
-- from ``_window_bounds`` in the stats service, the same helper the dashboard
-- rollup uses, so the chart and the KPI above it describe one span.
--
-- :window_start (timestamptz) -- inclusive lower bound.
-- :window_end   (timestamptz) -- EXCLUSIVE upper bound (the day after the last
--   day the user picked, so a range through Aug 29 covers all of Aug 29).
--
-- Scope: GLOBAL. http_audit_log records the acting user but not their
-- organization, so this metric cannot be tenant-filtered.
SELECT
    d::date AS day,
    COUNT(h.id)                                                        AS requests_total,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY h.latency_ms)         AS p50_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY h.latency_ms)         AS p95_latency_ms
FROM generate_series(
    date_trunc('day', CAST(:window_start AS timestamptz)),
    -- window_end is exclusive, so the last SERIES day is the one before it.
    -- Without the step back, a range ending Aug 29 would emit an empty Aug 30.
    date_trunc('day', CAST(:window_end AS timestamptz) - INTERVAL '1 microsecond'),
    INTERVAL '1 day'
) d
LEFT JOIN http_audit_log h
    ON h.created_at >= d
    AND h.created_at < d + INTERVAL '1 day'
    AND h.created_at >= CAST(:window_start AS timestamptz)
    AND h.created_at < CAST(:window_end AS timestamptz)
GROUP BY d::date
ORDER BY d::date;
