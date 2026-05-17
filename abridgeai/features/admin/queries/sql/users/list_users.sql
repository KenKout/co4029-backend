-- Paginated user list with optional status / role / org filters.
-- :status_filter (text | NULL)            -- match users.status, NULL = any.
-- :role_code     (text | NULL)            -- filter to users with at least one active assignment of role.
-- :organization_id (uuid | NULL)          -- when NULL: global; when set: must have a non-deleted org membership.
-- :limit, :offset                         -- pagination.
SELECT
    u.id              AS user_id,
    u.primary_email   AS primary_email,
    u.status          AS status,
    u.last_login_at   AS last_login_at,
    u.created_at      AS created_at,
    u.updated_at      AS updated_at
FROM users u
WHERE (:status_filter IS NULL OR u.status = :status_filter)
  AND (CAST(:organization_id AS uuid) IS NULL
       OR EXISTS (
           SELECT 1 FROM organization_memberships om
           WHERE om.user_id = u.id
             AND om.organization_id = CAST(:organization_id AS uuid)
             AND om.deleted_at IS NULL
       ))
  AND (:role_code IS NULL
       OR EXISTS (
           SELECT 1
           FROM user_role_assignments ura
           JOIN roles r ON r.id = ura.role_id
           WHERE ura.user_id = u.id
             AND ura.deleted_at IS NULL
             AND ura.active_from <= NOW()
             AND (ura.active_until IS NULL OR ura.active_until > NOW())
             AND r.code = :role_code
             AND (CAST(:organization_id AS uuid) IS NULL
                  OR ura.scope_kind = 'global'
                  OR ura.organization_id = CAST(:organization_id AS uuid))
       ))
ORDER BY u.created_at DESC, u.id
LIMIT :limit OFFSET :offset;
