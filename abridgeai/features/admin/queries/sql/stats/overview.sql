-- Aggregate counters for the admin overview tile.
--
-- :organization_id (uuid | NULL) -- when NULL, returns global aggregates;
-- when provided, filters every count down to data scoped to that organization.
--
-- Tables touched:
--   * users (hard-delete; org-scoped via organization_memberships)
--   * courses (soft-delete; organization_id direct column)
--   * course_enrollments (no soft-delete; reaches org via courses)
--   * learning_materials (no organization_id; reaches org via lessons -> modules -> courses)
--   * quiz_attempts (no organization_id; reaches org via quizzes -> modules -> courses)
--
-- All single-row scalar aggregations -- safe under a Manager's org-scoped pass.
SELECT
    (
        SELECT COUNT(*)
        FROM users u
        WHERE CAST(:organization_id AS uuid) IS NULL
           OR EXISTS (
               SELECT 1
               FROM organization_memberships om
               WHERE om.user_id = u.id
                 AND om.organization_id = CAST(:organization_id AS uuid)
                 AND om.deleted_at IS NULL
           )
    ) AS total_users,
    (
        SELECT COUNT(*)
        FROM courses c
        WHERE c.deleted_at IS NULL
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS total_courses,
    (
        SELECT COUNT(*)
        FROM course_enrollments ce
        JOIN courses c ON c.id = ce.course_id AND c.deleted_at IS NULL
        WHERE CAST(:organization_id AS uuid) IS NULL
           OR c.organization_id = CAST(:organization_id AS uuid)
    ) AS total_enrollments,
    (
        SELECT COUNT(*)
        FROM learning_materials lm
        JOIN lessons l ON l.id = lm.lesson_id AND l.deleted_at IS NULL
        JOIN modules m ON m.id = l.module_id AND m.deleted_at IS NULL
        JOIN courses c ON c.id = m.course_id AND c.deleted_at IS NULL
        WHERE CAST(:organization_id AS uuid) IS NULL
           OR c.organization_id = CAST(:organization_id AS uuid)
    ) AS total_materials,
    (
        SELECT COUNT(*)
        FROM quiz_attempts qa
        JOIN quizzes q ON q.id = qa.quiz_id AND q.deleted_at IS NULL
        JOIN modules m ON m.id = q.module_id AND m.deleted_at IS NULL
        JOIN courses c ON c.id = m.course_id AND c.deleted_at IS NULL
        WHERE CAST(:organization_id AS uuid) IS NULL
           OR c.organization_id = CAST(:organization_id AS uuid)
    ) AS total_quiz_attempts;
