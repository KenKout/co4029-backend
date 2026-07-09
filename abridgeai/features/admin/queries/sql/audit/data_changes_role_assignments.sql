-- Data-change lookup for a single user_role_assignments row -- who granted
-- the role, when it became active, and whether it has been revoked
-- (soft-deleted). The role code is joined in for a human-readable label.
-- ``granted_by`` is the assignment's actor; it is surfaced as ``created_by``
-- so the response shape stays uniform across entity kinds.
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
WHERE ra.id = CAST(:entity_id AS uuid);
