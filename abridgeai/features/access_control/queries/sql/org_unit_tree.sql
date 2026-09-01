-- Ancestor walk for FIX-CRIT-2: the faculty hosting :course_id and
-- every parent unit. A HOD assigned at unit Y matches courses in Y or
-- any descendant. UNION (not UNION ALL) is cycle-safe.
-- 0094 renamed courses.org_unit_id -> faculty_id (courses live on a faculty).
-- Composed by load_course_permissions under a top-level WITH RECURSIVE.
org_unit_tree AS (
    SELECT c.faculty_id AS unit_id
    FROM courses c
    WHERE c.id = :course_id

    UNION

    SELECT ou.parent_unit_id AS unit_id
    FROM org_units ou
    JOIN org_unit_tree t ON ou.id = t.unit_id
    WHERE ou.parent_unit_id IS NOT NULL
)
