-- Data-change LIST for user_role_assignments: every row updated within the
-- window, newest first. Same uniform projection as the single-lookup
-- variant (entity_id / title / status / scope_kind / subject_user_id /
-- *_by / *_at / organization_id).
SELECT
    ra.id              AS entity_id,
    r.code             AS title,
    CASE WHEN ra.deleted_at IS NOT NULL THEN 'revoked' ELSE 'active' END AS status,
    ra.scope_kind      AS scope_kind,
    ra.user_id         AS subject_user_id,
    ra.granted_by      AS created_by,
    ra.updated_by      AS updated_by,
    ra.deleted_by      AS deleted_by,
    ra.created_at      AS created_at,
    ra.updated_at      AS updated_at,
    ra.deleted_at      AS deleted_at,
    ra.organization_id AS organization_id
FROM user_role_assignments ra
JOIN roles r ON r.id = ra.role_id
WHERE ra.updated_at >= CAST(:since AS timestamptz)
  AND (CAST(:until AS timestamptz) IS NULL OR updated_at < CAST(:until AS timestamptz))
ORDER BY ra.updated_at DESC
LIMIT :limit;
