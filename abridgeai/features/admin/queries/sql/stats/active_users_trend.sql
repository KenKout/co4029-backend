-- Daily active users trend: distinct users who created an auth session on
-- each calendar day (a session row is created on every login — see
-- identity/services/login.py::_issue_tokens), so this is the honest
-- "how many people logged in that day" curve behind the DAU/WAU/MAU numbers.
--
-- Windowed by explicit bounds rather than a day count, for the same reason as
-- api_latency_trend.sql: a day count cannot express a range that ENDS in the
-- past, so a picker range of "Aug 1–Aug 8" would have plotted the last 8 days
-- up to today instead. Bounds come from ``_window_bounds`` in the stats
-- service, the same helper the dashboard rollup uses.
--
-- :organization_id (uuid | NULL) -- when NULL, global; otherwise scoped via
--   organization_memberships (matches active_users.sql).
-- :window_start (timestamptz)    -- inclusive lower bound.
-- :window_end   (timestamptz)    -- EXCLUSIVE upper bound (the day after the
--   last day the user picked).
--
-- Returns one row per day (including zero-activity days, so the chart is
-- continuous): `day` (date) and `count` (distinct users).
SELECT
    d::date AS day,
    COUNT(DISTINCT s.user_id) AS count
FROM generate_series(
    date_trunc('day', CAST(:window_start AS timestamptz)),
    -- window_end is exclusive, so the last SERIES day is the one before it.
    date_trunc('day', CAST(:window_end AS timestamptz) - INTERVAL '1 microsecond'),
    INTERVAL '1 day'
) d
LEFT JOIN auth_sessions s
    ON s.created_at >= d
    AND s.created_at < d + INTERVAL '1 day'
    AND s.created_at >= CAST(:window_start AS timestamptz)
    AND s.created_at < CAST(:window_end AS timestamptz)
    AND (CAST(:organization_id AS uuid) IS NULL
         OR EXISTS (
             SELECT 1
             FROM organization_memberships om
             WHERE om.user_id = s.user_id
               AND om.organization_id = CAST(:organization_id AS uuid)
               AND om.deleted_at IS NULL
         ))
GROUP BY d::date
ORDER BY d::date;
