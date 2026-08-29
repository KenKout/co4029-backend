-- Organizations with no activity in the last :days days.
--
-- ONE definition, used by both the dashboard's inactive-tenant count and the
-- organizations list that count links to (PRD ADM-045 / ADM-021). They were
-- about to disagree the same way the job counters did: the dashboard had this
-- predicate inlined in ``stats/dashboard.sql`` while the list had no filter at
-- all, so clicking a count of 2 landed on every organization on the platform.
--
-- "Activity" is deliberately broad — an organization is inactive only when
-- ALL FOUR of these are silent, because any one of them means somebody is
-- still using it:
--   * a member signed in
--   * a course was edited
--   * a quiz attempt was started
--   * an interview session was started
--
-- Deliberately NOT counted as activity: a course merely existing, a user
-- merely existing, or an AI call. The first two are inventory rather than use,
-- and AI calls carry no organization edge so including them would silently
-- change which organizations qualify depending on how a call was attributed.
--
-- Returns the id plus the most recent activity timestamp across all four
-- sources, so the list can show HOW long each tenant has been quiet rather
-- than only that it crossed the threshold. NULL means no activity was ever
-- recorded, which reads differently from "quiet since March" and should.
--
-- :now             (timestamptz) -- evaluation reference; tests pin it.
-- :days            (int)         -- inactivity threshold.
-- :organization_id (uuid | NULL) -- optional single-tenant check.
WITH bounds AS (
    SELECT
        CAST(:now AS timestamptz) AS as_of,
        CAST(:now AS timestamptz)
            - make_interval(days => CAST(:days AS int)) AS cutoff
),
candidates AS (
    SELECT o.id, o.name, o.slug
    FROM organizations o
    WHERE o.deleted_at IS NULL
      AND (CAST(:organization_id AS uuid) IS NULL
           OR o.id = CAST(:organization_id AS uuid))
),
activity AS (
    SELECT
        c.id,
        GREATEST(
            (
                SELECT MAX(u.last_login_at)
                FROM organization_memberships om
                JOIN users u ON u.id = om.user_id
                WHERE om.organization_id = c.id AND om.deleted_at IS NULL
            ),
            (
                SELECT MAX(c2.updated_at)
                FROM courses c2
                WHERE c2.organization_id = c.id AND c2.deleted_at IS NULL
            ),
            (
                SELECT MAX(qa.started_at)
                FROM quiz_attempts qa
                JOIN quizzes q ON q.id = qa.quiz_id
                JOIN courses c3 ON c3.id = q.course_id
                WHERE c3.organization_id = c.id
            ),
            (
                SELECT MAX(s.started_at)
                FROM interview_sessions s
                JOIN interview_configs ic ON ic.id = s.interview_config_id
                JOIN courses c4 ON c4.id = ic.course_id
                WHERE c4.organization_id = c.id
            )
        ) AS last_activity_at
    FROM candidates c
)
SELECT
    c.id,
    c.name,
    c.slug,
    a.last_activity_at,
    -- NULL when nothing was ever recorded: "never active" is not "quiet for
    -- N days", and a caller rendering days-quiet needs to tell them apart.
    CASE
        WHEN a.last_activity_at IS NULL THEN NULL
        ELSE EXTRACT(DAY FROM (b.as_of - a.last_activity_at))::int
    END AS days_quiet
FROM candidates c
JOIN activity a ON a.id = c.id
CROSS JOIN bounds b
-- GREATEST ignores NULLs in Postgres, so an org with no activity at all lands
-- here via the IS NULL arm rather than being dropped by the comparison.
WHERE a.last_activity_at IS NULL OR a.last_activity_at < b.cutoff
ORDER BY a.last_activity_at ASC NULLS FIRST, c.name;
