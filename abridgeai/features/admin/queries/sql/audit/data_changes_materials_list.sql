-- Data-change LIST for learning materials: every row updated within the
-- window, newest first. Same uniform projection as the single-lookup
-- variant (entity_id / title / material_type / status / *_by / *_at /
-- organization_id / lesson_id).
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
WHERE m.updated_at >= CAST(:since AS timestamptz)
  AND (CAST(:until AS timestamptz) IS NULL OR updated_at < CAST(:until AS timestamptz))
ORDER BY m.updated_at DESC
LIMIT :limit;
