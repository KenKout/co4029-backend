-- Data-change LIST for user accounts: every row updated within the window,
-- newest first. Same uniform projection as the single-lookup variant
-- (entity_id / title / status / primary_email / *_by / *_at /
-- organization_id).
SELECT
    u.id             AS entity_id,
    COALESCE(p.display_name, u.primary_email) AS title,
    u.status         AS status,
    u.primary_email  AS primary_email,
    CAST(NULL AS uuid) AS created_by,
    CAST(NULL AS uuid) AS updated_by,
    CAST(NULL AS uuid) AS deleted_by,
    u.created_at     AS created_at,
    u.updated_at     AS updated_at,
    CAST(NULL AS timestamptz) AS deleted_at,
    CAST(NULL AS uuid) AS organization_id
FROM users u
LEFT JOIN user_profiles p ON p.user_id = u.id
WHERE u.updated_at >= CAST(:since AS timestamptz)
ORDER BY u.updated_at DESC
LIMIT :limit;
