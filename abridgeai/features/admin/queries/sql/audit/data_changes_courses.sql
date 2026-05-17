-- Data-change lookup for a single course -- who created / last updated / soft-deleted.
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
WHERE c.id = CAST(:entity_id AS uuid);
