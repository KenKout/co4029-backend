-- Active role assignments for a single user (used by user detail view).
-- Scope FK ids are resolved to human-readable names (LEFT JOIN so an
-- assignment whose org/unit/course was since deleted still returns, with a
-- NULL name the UI falls back on). The admin surface already reads across
-- feature tables via raw SQL, so these joins match the existing pattern.
SELECT
    ura.id              AS assignment_id,
    ura.role_id         AS role_id,
    r.code              AS role_code,
    r.name              AS role_name,
    ura.scope_kind      AS scope_kind,
    ura.organization_id AS organization_id,
    o.name              AS organization_name,
    ura.org_unit_id     AS org_unit_id,
    ou.name             AS org_unit_name,
    ura.course_id       AS course_id,
    c.title             AS course_title,
    ura.active_from     AS active_from,
    ura.active_until    AS active_until,
    ura.created_at      AS created_at
FROM user_role_assignments ura
JOIN roles r ON r.id = ura.role_id
LEFT JOIN organizations o ON o.id = ura.organization_id
LEFT JOIN org_units ou ON ou.id = ura.org_unit_id
LEFT JOIN courses c ON c.id = ura.course_id
WHERE ura.user_id = CAST(:user_id AS uuid)
  AND ura.deleted_at IS NULL
ORDER BY ura.created_at DESC;
