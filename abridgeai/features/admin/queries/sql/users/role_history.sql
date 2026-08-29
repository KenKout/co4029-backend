-- Full role-assignment history for one account, including revoked rows and
-- the actors who granted/revoked them. This is separate from the active-role
-- projection used by the editor so history never changes authorization UI.
SELECT
    ura.id              AS assignment_id,
    r.code              AS role_code,
    r.name              AS role_name,
    ura.scope_kind      AS scope_kind,
    ura.organization_id AS organization_id,
    o.name              AS organization_name,
    ura.org_unit_id     AS org_unit_id,
    ou.name             AS org_unit_name,
    ura.course_id       AS course_id,
    c.title             AS course_title,
    ura.granted_by      AS granted_by,
    grantor.primary_email AS granted_by_email,
    ura.deleted_by      AS revoked_by,
    revoker.primary_email AS revoked_by_email,
    ura.active_from     AS active_from,
    ura.active_until    AS active_until,
    ura.created_at      AS created_at,
    ura.updated_at      AS updated_at,
    ura.deleted_at      AS revoked_at
FROM user_role_assignments ura
JOIN roles r ON r.id = ura.role_id
LEFT JOIN organizations o ON o.id = ura.organization_id
LEFT JOIN org_units ou ON ou.id = ura.org_unit_id
LEFT JOIN courses c ON c.id = ura.course_id
LEFT JOIN users grantor ON grantor.id = ura.granted_by
LEFT JOIN users revoker ON revoker.id = ura.deleted_by
WHERE ura.user_id = CAST(:user_id AS uuid)
ORDER BY COALESCE(ura.deleted_at, ura.updated_at, ura.created_at) DESC;
