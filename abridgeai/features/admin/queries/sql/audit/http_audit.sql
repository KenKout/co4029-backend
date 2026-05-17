-- HTTP audit log search -- consumes T0.23's http_audit_log table when present.
-- Bounded by :since (required at the service layer) and optional user / path filters.
SELECT
    h.id            AS id,
    h.user_id       AS user_id,
    h.session_id    AS session_id,
    h.method        AS method,
    h.path          AS path,
    h.status_code   AS status_code,
    h.latency_ms    AS latency_ms,
    h.ip_address::text AS ip_address,
    h.user_agent    AS user_agent,
    h.created_at    AS created_at
FROM http_audit_log h
WHERE h.created_at >= CAST(:since AS timestamptz)
  AND (CAST(:user_id AS uuid) IS NULL OR h.user_id = CAST(:user_id AS uuid))
  AND (CAST(:path_pattern AS text) IS NULL OR h.path LIKE CAST(:path_pattern AS text))
ORDER BY h.created_at DESC
LIMIT :limit;
