-- Data-change LIST for courses: every row updated within the window,
-- newest first. Same uniform projection as the single-lookup variant
-- (entity_id / title / slug / status / *_by / *_at / organization_id).
SELECT
    c.id           AS entity_id,
    c.title        AS title,
    c.slug         AS slug,
    c.status       AS status,
    c.created_by   AS created_by,
    c.updated_by   AS updated_by,
    c.deleted_by   AS deleted_by,
    c.created_at   AS created_at,
    c.updated_at   AS updated_at,
    c.deleted_at   AS deleted_at,
    c.organization_id AS organization_id
FROM courses c
WHERE c.updated_at >= CAST(:since AS timestamptz)
  AND (CAST(:until AS timestamptz) IS NULL OR updated_at < CAST(:until AS timestamptz))
ORDER BY c.updated_at DESC
LIMIT :limit;
