-- Data-change lookup for a single user account. Users carry only
-- created_at / updated_at (TimestampMixin, no AuditedByMixin / SoftDelete),
-- so created_by / updated_by / deleted_by / deleted_at are surfaced as NULL
-- to keep the response shape uniform with the other entity kinds. The
-- profile display name is joined in for a human-readable label; users are
-- global (no organization_id column) so organization_id is NULL.
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
WHERE u.id = CAST(:entity_id AS uuid);
