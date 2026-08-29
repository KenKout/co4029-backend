-- Active (non-revoked, non-expired) auth sessions for a single user.
SELECT
    s.id           AS id,
    s.expires_at   AS expires_at,
    s.revoked_at   AS revoked_at,
    s.ip_address   AS ip_address,
    s.user_agent   AS user_agent,
    s.created_at   AS created_at
FROM auth_sessions s
WHERE s.user_id = CAST(:user_id AS uuid)
  AND s.revoked_at IS NULL
  AND (s.expires_at IS NULL OR s.expires_at > NOW())
ORDER BY s.created_at DESC;
