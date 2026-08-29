-- HTTP audit log search -- consumes T0.23's http_audit_log table when present.
-- Bounded by :since (required at the service layer) and optional user / path filters.
SELECT
    h.id            AS id,
    h.request_id    AS request_id,
    h.user_id       AS user_id,
    h.session_id    AS session_id,
    h.method        AS method,
    h.path          AS path,
    h.status_code   AS status_code,
    CASE
        WHEN h.path = '/api/v1/auth/google/callback' AND h.status_code = 401
            THEN 'authentication_rejected'
        WHEN h.path = '/api/v1/auth/google/callback' AND h.status_code = 403
            THEN 'account_not_provisioned'
        WHEN h.path = '/api/v1/auth/google/callback' AND h.status_code >= 400
            THEN 'login_request_failed'
        WHEN h.status_code = 403 THEN 'authorization_denied'
        WHEN h.status_code >= 500 THEN 'server_error'
        WHEN h.status_code >= 400 THEN 'client_error'
        ELSE NULL
    END             AS failure_reason,
    h.latency_ms    AS latency_ms,
    h.ip_address::text AS ip_address,
    h.user_agent    AS user_agent,
    h.created_at    AS created_at
FROM http_audit_log h
WHERE h.created_at >= CAST(:since AS timestamptz)
  AND (CAST(:until AS timestamptz) IS NULL OR h.created_at < CAST(:until AS timestamptz))
  AND (CAST(:user_id AS uuid) IS NULL OR h.user_id = CAST(:user_id AS uuid))
  AND (CAST(:request_id AS uuid) IS NULL OR h.request_id = CAST(:request_id AS uuid))
  AND (CAST(:path_pattern AS text) IS NULL OR h.path LIKE CAST(:path_pattern AS text))
  AND (
      CAST(:event_kind AS text) IS NULL
      OR (
          CAST(:event_kind AS text) = 'login_failure'
          AND h.path = '/api/v1/auth/google/callback'
          AND h.status_code >= 400
      )
      OR (
          CAST(:event_kind AS text) = 'denied'
          AND h.status_code = 403
          AND h.path <> '/api/v1/auth/google/callback'
      )
  )
ORDER BY h.created_at DESC
LIMIT :limit;
