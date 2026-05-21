-- Paginated user list with optional status / role / org / search filters.
-- :status_filter (text | NULL)            -- match users.status, NULL = any.
-- :role_code     (text | NULL)            -- filter to users with at least one active assignment of role.
-- :organization_id (uuid | NULL)          -- when NULL: global; when set: must have a non-deleted org membership.
-- :q             (text | NULL)            -- substring match against primary_email or profile display_name (case-insensitive).
-- :limit, :offset                         -- pagination.
SELECT
    u.id              AS user_id,
    u.primary_email   AS primary_email,
    u.status          AS status,
    p.display_name    AS display_name,
    u.last_login_at   AS last_login_at,
    u.created_at      AS created_at,
    u.updated_at      AS updated_at
FROM users u
LEFT JOIN user_profiles p
       ON p.user_id = u.id
      AND p.deleted_at IS NULL
WHERE (CAST(:status_filter AS text) IS NULL OR u.status = CAST(:status_filter AS text))
  AND (CAST(:organization_id AS uuid) IS NULL
       OR EXISTS (
           SELECT 1 FROM organization_memberships om
           WHERE om.user_id = u.id
             AND om.organization_id = CAST(:organization_id AS uuid)
             AND om.deleted_at IS NULL
       ))
  AND (CAST(:role_code AS text) IS NULL
       OR EXISTS (
           SELECT 1
           FROM user_role_assignments ura
           JOIN roles r ON r.id = ura.role_id
           WHERE ura.user_id = u.id
             AND ura.deleted_at IS NULL
             AND ura.active_from <= NOW()
             AND (ura.active_until IS NULL OR ura.active_until > NOW())
             AND r.code = CAST(:role_code AS text)
             AND (CAST(:organization_id AS uuid) IS NULL
                  OR ura.scope_kind = 'global'
                  OR ura.organization_id = CAST(:organization_id AS uuid))
       ))
  AND (CAST(:q AS text) IS NULL
       OR u.primary_email ILIKE '%' || CAST(:q AS text) || '%'
       OR p.display_name ILIKE '%' || CAST(:q AS text) || '%')
ORDER BY u.created_at DESC, u.id
LIMIT :limit OFFSET :offset;
