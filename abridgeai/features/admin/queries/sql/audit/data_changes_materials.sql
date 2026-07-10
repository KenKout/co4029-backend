-- Data-change lookup for a single learning material -- who created / last
-- updated / soft-deleted. Organisation is resolved through the owning
-- lesson -> module -> course chain (materials carry no org_id column).
SELECT
    m.id            AS entity_id,
    m.title         AS title,
    m.material_type AS material_type,
    CASE WHEN m.deleted_at IS NOT NULL THEN 'deleted' ELSE 'active' END AS status,
    m.created_by    AS created_by,
    m.updated_by    AS updated_by,
    m.deleted_by    AS deleted_by,
    m.created_at    AS created_at,
    m.updated_at    AS updated_at,
    m.deleted_at    AS deleted_at,
    c.organization_id AS organization_id,
    m.lesson_id     AS lesson_id
FROM learning_materials m
JOIN lessons l  ON l.id = m.lesson_id
JOIN modules mo ON mo.id = l.module_id
JOIN courses c  ON c.id = mo.course_id
WHERE m.id = CAST(:entity_id AS uuid);
