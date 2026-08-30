-- Security & access rollup for the operator dashboard (PRD ADM-020).
--
-- Every figure here is a COUNT with an explicit definition, not a scored
-- "alert". Alert rules, severities and review state need decisions this
-- deployment has not made yet (open decision D-03), and inventing thresholds
-- would produce a number nobody can act on or trust. What is countable today
-- is counted; the thresholds arrive with the rules.
--
-- Definitions, all bounded by :since so the scans stay indexed
-- (ix_http_audit_log_created_at):
--
--   failed_logins       -- failed Google OAuth callback requests only. Token
--                          refresh/logout traffic is deliberately excluded:
--                          it is session maintenance, not a login attempt.
--   denied_requests     -- 403 anywhere OUTSIDE the auth surface. This is
--                          authorization, not authentication: a signed-in user
--                          reaching for something that is not theirs. Kept
--                          separate from failed_logins because the response is
--                          different — one is a password problem, the other is
--                          a permissions problem or probing.
--   distinct_failed_ips -- how many sources produced those failures. One IP
--                          failing 40 times is somebody's stale password; 40
--                          IPs failing once each is not.
--   role_changes        -- role assignments granted, modified or revoked.
--   privileged_accounts -- ACTIVE users currently holding admin / manager /
--                          hod at any scope. A point-in-time inventory, not a
--                          windowed count, hence no :since.
--
-- Scope: GLOBAL for the http_audit_log figures — the table records the acting
-- user but not their organization, so these cannot be tenant-filtered.
-- :organization_id narrows role changes and privileged accounts only, and the
-- router reports that split honestly.
--
-- :now             (timestamptz) -- as-of reference; tests pin it.
-- :since           (timestamptz) -- lower bound for the windowed counts.
-- :organization_id (uuid | NULL) -- org filter where the edge exists.
WITH auth_failures AS (
    SELECT h.ip_address, h.user_id
    FROM http_audit_log h
    WHERE h.created_at >= CAST(:since AS timestamptz)
      AND h.created_at < CAST(:now AS timestamptz)
      AND h.status_code >= 400
      AND h.path = '/api/v1/auth/google/callback'
)
SELECT
    CAST(:now AS timestamptz) AS as_of,
    (SELECT COUNT(*) FROM auth_failures) AS failed_logins,
    (
        -- NULL, not 0, when nothing failed: "no failures" and "failures from
        -- an unknown number of sources" are different, and a 0 here would read
        -- as the latter.
        SELECT NULLIF(COUNT(DISTINCT ip_address), 0) FROM auth_failures
    ) AS distinct_failed_ips,
    (
        SELECT COUNT(*)
        FROM http_audit_log h
        WHERE h.created_at >= CAST(:since AS timestamptz)
          AND h.created_at < CAST(:now AS timestamptz)
          AND h.status_code = 403
          AND h.path <> '/api/v1/auth/google/callback'
    ) AS denied_requests,
    (
        SELECT COUNT(*)
        FROM user_role_assignments ura
        WHERE ura.updated_at >= CAST(:since AS timestamptz)
          AND ura.updated_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR ura.organization_id = CAST(:organization_id AS uuid))
    ) AS role_changes,
    (
        -- Revocations specifically: a role being taken away is the half of
        -- role churn most worth a second look during an incident.
        SELECT COUNT(*)
        FROM user_role_assignments ura
        WHERE ura.deleted_at >= CAST(:since AS timestamptz)
          AND ura.deleted_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR ura.organization_id = CAST(:organization_id AS uuid))
    ) AS role_revocations,
    (
        SELECT COUNT(DISTINCT ura.user_id)
        FROM user_role_assignments ura
        JOIN roles r ON r.id = ura.role_id
        JOIN users u ON u.id = ura.user_id
        WHERE r.code IN ('admin', 'manager', 'hod')
          AND ura.deleted_at IS NULL
          AND u.status = 'active'
          AND (ura.active_until IS NULL
               OR ura.active_until > CAST(:now AS timestamptz))
          AND (CAST(:organization_id AS uuid) IS NULL
               OR ura.organization_id = CAST(:organization_id AS uuid))
    ) AS privileged_accounts,
    (
        -- Sessions still valid right now. A spike here alongside failed logins
        -- is a different story from a spike on its own.
        SELECT COUNT(*)
        FROM auth_sessions s
        WHERE s.revoked_at IS NULL
          AND s.expires_at > CAST(:now AS timestamptz)
    ) AS active_sessions
;
