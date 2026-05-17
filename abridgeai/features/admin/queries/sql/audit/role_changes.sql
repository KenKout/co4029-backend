-- UserRoleAssignment audit lookup -- who granted/revoked roles since :since.
-- Bounded by :since to prevent unbounded scans (caller must pass a value).
SELECT
    ura.id                AS assignment_id,
    ura.user_id           AS user_id,
    ura.role_id           AS role_id,
    r.code                AS role_code,
    ura.scope_kind        AS scope_kind,
    ura.organization_id   AS organization_id,
    ura.org_unit_id       AS org_unit_id,
    ura.course_id         AS course_id,
    ura.granted_by        AS granted_by,
    ura.active_from       AS active_from,
    ura.active_until      AS active_until,
    ura.deleted_at        AS deleted_at,
    ura.deleted_by        AS deleted_by,
    ura.created_at        AS created_at,
    ura.updated_at        AS updated_at
FROM user_role_assignments ura
JOIN roles r ON r.id = ura.role_id
WHERE ura.updated_at >= CAST(:since AS timestamptz)
  AND (CAST(:organization_id AS uuid) IS NULL
       OR ura.organization_id = CAST(:organization_id AS uuid))
ORDER BY ura.updated_at DESC
LIMIT :limit;
