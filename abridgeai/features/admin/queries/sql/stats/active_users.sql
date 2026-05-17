-- DAU / WAU / MAU based on users.last_login_at.
--
-- :organization_id (uuid | NULL) -- when NULL, global; otherwise scoped via
--   organization_memberships.
-- :now (timestamptz)             -- evaluation reference timestamp; tests pin this
--   so windows are deterministic.
--
-- Returns one row with three integer counts: DAU (24h), WAU (7d), MAU (30d).
SELECT
    SUM(CASE WHEN u.last_login_at >= CAST(:now AS timestamptz) - INTERVAL '1 day'
             THEN 1 ELSE 0 END) AS dau,
    SUM(CASE WHEN u.last_login_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
             THEN 1 ELSE 0 END) AS wau,
    SUM(CASE WHEN u.last_login_at >= CAST(:now AS timestamptz) - INTERVAL '30 days'
             THEN 1 ELSE 0 END) AS mau
FROM users u
WHERE u.last_login_at IS NOT NULL
  AND (CAST(:organization_id AS uuid) IS NULL
       OR EXISTS (
           SELECT 1
           FROM organization_memberships om
           WHERE om.user_id = u.id
             AND om.organization_id = CAST(:organization_id AS uuid)
             AND om.deleted_at IS NULL
       ));
