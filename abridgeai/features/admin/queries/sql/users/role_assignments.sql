-- Active role assignments for a single user (used by user detail view).
SELECT
    ura.id              AS assignment_id,
    ura.role_id         AS role_id,
    r.code              AS role_code,
    ura.scope_kind      AS scope_kind,
    ura.organization_id AS organization_id,
    ura.org_unit_id     AS org_unit_id,
    ura.course_id       AS course_id,
    ura.active_from     AS active_from,
    ura.active_until    AS active_until,
    ura.created_at      AS created_at
FROM user_role_assignments ura
JOIN roles r ON r.id = ura.role_id
WHERE ura.user_id = CAST(:user_id AS uuid)
  AND ura.deleted_at IS NULL
ORDER BY ura.created_at DESC;
