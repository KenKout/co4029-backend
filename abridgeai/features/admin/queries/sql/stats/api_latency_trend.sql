-- Daily API latency trend: per-calendar-day p50 / p95 latency and request
-- volume, from the same source as api_reliability.sql (http_audit_log —
-- written by the audit middleware for every non-exempt request).
--
-- Contract mirrors top-of-file conventions there: percentiles are NULL on a
-- zero-traffic day (the caller renders a flat line, not a fabricated 0ms),
-- and one row is returned per day INCLUDING zero-traffic days so the chart
-- is continuous.
--
-- :days (int)          -- lookback window in calendar days.
-- :now (timestamptz)   -- reference timestamp; tests pin this.
--
-- Scope: GLOBAL. http_audit_log records the acting user but not their
-- organization, so this metric cannot be tenant-filtered.
SELECT
    d::date AS day,
    COUNT(h.id)                                                        AS requests_total,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY h.latency_ms)         AS p50_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY h.latency_ms)         AS p95_latency_ms
FROM generate_series(
    CAST(:now AS timestamptz) - (CAST(:days AS int) - 1) * INTERVAL '1 day',
    CAST(:now AS timestamptz),
    INTERVAL '1 day'
) d
LEFT JOIN http_audit_log h
    ON h.created_at::date = d::date
    AND h.created_at < CAST(:now AS timestamptz)
GROUP BY d::date
ORDER BY d::date;