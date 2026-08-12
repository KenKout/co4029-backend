-- Daily active users trend: distinct users who created an auth session on
-- each calendar day (a session row is created on every login — see
-- identity/services/login.py::_issue_tokens), so this is the honest
-- "how many people logged in that day" curve behind the DAU/WAU/MAU numbers.
--
-- :organization_id (uuid | NULL) -- when NULL, global; otherwise scoped via
--   organization_memberships (matches active_users.sql).
-- :days (int)                    -- lookback window in calendar days.
-- :now (timestamptz)             -- reference timestamp; tests pin this.
--
-- Returns one row per day (including zero-activity days, so the chart is
-- continuous): `day` (date) and `count` (distinct users).
SELECT
    d::date AS day,
    COUNT(DISTINCT s.user_id) AS count
FROM generate_series(
    CAST(:now AS timestamptz) - (CAST(:days AS int) - 1) * INTERVAL '1 day',
    CAST(:now AS timestamptz),
    INTERVAL '1 day'
) d
LEFT JOIN auth_sessions s
    ON s.created_at::date = d::date
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
