-- Content + AI processing breakdown.
-- Returns three result sets via array_agg:
--   * courses by status
--   * materials by type
--   * processing_jobs by status
SELECT
    (
        SELECT COALESCE(jsonb_agg(jsonb_build_object('status', t.status, 'count', t.cnt) ORDER BY t.status), '[]'::jsonb)
        FROM (
            SELECT c.status AS status, COUNT(*) AS cnt
            FROM courses c
            WHERE c.deleted_at IS NULL
              AND (CAST(:organization_id AS uuid) IS NULL
                   OR c.organization_id = CAST(:organization_id AS uuid))
            GROUP BY c.status
        ) t
    ) AS courses_by_status,
    (
        SELECT COALESCE(jsonb_agg(jsonb_build_object('material_type', t.material_type, 'count', t.cnt) ORDER BY t.material_type), '[]'::jsonb)
        FROM (
            SELECT lm.material_type AS material_type, COUNT(*) AS cnt
            FROM learning_materials lm
            JOIN lessons l ON l.id = lm.lesson_id AND l.deleted_at IS NULL
            JOIN modules m ON m.id = l.module_id AND m.deleted_at IS NULL
            JOIN courses c ON c.id = m.course_id AND c.deleted_at IS NULL
            WHERE CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid)
            GROUP BY lm.material_type
        ) t
    ) AS materials_by_type,
    (
        SELECT COALESCE(jsonb_agg(jsonb_build_object('status', t.status, 'count', t.cnt) ORDER BY t.status), '[]'::jsonb)
        FROM (
            SELECT pj.status AS status, COUNT(*) AS cnt
            FROM processing_jobs pj
            GROUP BY pj.status
        ) t
    ) AS processing_jobs_by_status,
    (
        SELECT COUNT(*)
        FROM courses c
        WHERE c.deleted_at IS NULL
          AND c.created_at >= now() - interval '7 days'
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS courses_created_7d,
    (
        SELECT COUNT(*)
        FROM learning_materials lm
        JOIN lessons l ON l.id = lm.lesson_id AND l.deleted_at IS NULL
        JOIN modules m ON m.id = l.module_id AND m.deleted_at IS NULL
        JOIN courses c ON c.id = m.course_id AND c.deleted_at IS NULL
        WHERE lm.created_at >= now() - interval '7 days'
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS materials_created_7d,
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.created_at >= date_trunc('day', now())
    ) AS processing_jobs_created_today;
