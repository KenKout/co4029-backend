-- Operator dashboard: the cost, capacity and tenant half of the
-- ``GET /admin/stats/dashboard`` rollup.
--
-- Job metrics deliberately do NOT live here. Failure rate, terminal counts and
-- queue depth come from ``sql/jobs/terminal_metrics.sql`` and
-- ``sql/jobs/queue_state.sql`` -- the single definition shared with the
-- processing / Operations surface (PRD ADM-004). Adding a job aggregate back
-- into this file re-creates the three-way mismatch those files exist to fix.
--
-- Academic metrics deliberately do NOT live here either (PRD ADM-003). The
-- interview pass rate, unreviewed interview configs and quiz authoring gaps
-- used to occupy the system administrator's priority area; they are Manager /
-- Academic Operations signals and moved there.
--
-- :organization_id (uuid | NULL) -- when NULL, global aggregates; when provided,
--   filters every metric that can be traced to an organization.
-- :now (timestamptz)             -- evaluation reference timestamp; tests pin
--   this so all windows are deterministic.
-- :window_days (int)             -- primary window length. The preceding window
--   of the same length comes back alongside it so the UI can show direction and
--   not only level.
--
-- Org-traceability note: ``ai_model_calls`` has no organization_id and no
-- mandatory parent (ck_ai_model_calls_parent_ref allows stage_name-only rows),
-- so every spend / token / latency column below is GLOBAL even for an
-- org-scoped caller. The router reports that as ``cost_scope`` instead of
-- letting the number imply a tenant filter it never had.
--
-- Org-scoped edges used below:
--   users              -> organization_memberships
--   learning_materials -> lessons -> modules -> courses.organization_id
--   organizations      -> id
WITH bounds AS (
    SELECT
        CAST(:now AS timestamptz)                            AS as_of,
        make_interval(days => CAST(:window_days AS int))      AS window_len,
        make_interval(days => CAST(:window_days AS int) * 2)  AS window_len_x2
),
top_driver AS (
    SELECT
        COALESCE(amc.stage_name, amc.role) AS driver,
        SUM(amc.estimated_cost_usd)        AS spend_usd
    FROM ai_model_calls amc, bounds b
    WHERE amc.called_at >= b.as_of - b.window_len
      AND amc.called_at < b.as_of
      AND amc.estimated_cost_usd IS NOT NULL
      AND COALESCE(amc.stage_name, amc.role) IS NOT NULL
    GROUP BY COALESCE(amc.stage_name, amc.role)
    ORDER BY spend_usd DESC NULLS LAST
    LIMIT 1
),
slowest AS (
    SELECT
        amc.model_name,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY amc.latency_ms) AS p95_ms
    FROM ai_model_calls amc, bounds b
    WHERE amc.called_at >= b.as_of - b.window_len
      AND amc.called_at < b.as_of
      AND amc.latency_ms IS NOT NULL
    GROUP BY amc.model_name
    HAVING COUNT(*) >= 5
    ORDER BY p95_ms DESC NULLS LAST
    LIMIT 1
),
month_spend AS (
    SELECT
        COALESCE(SUM(amc.estimated_cost_usd), 0) AS spend_mtd_usd,
        -- day-of-month of :now == whole days elapsed (partial day counts as 1)
        EXTRACT(DAY FROM CAST(:now AS timestamptz))::numeric AS days_elapsed,
        EXTRACT(
            DAY FROM (
                date_trunc('month', CAST(:now AS timestamptz))
                + INTERVAL '1 month' - INTERVAL '1 day'
            )
        )::numeric AS days_in_month
    FROM ai_model_calls amc
    WHERE amc.called_at >= date_trunc('month', CAST(:now AS timestamptz))
      AND amc.called_at < CAST(:now AS timestamptz)
)
SELECT
    b.as_of AS as_of,
    ------------------------------------------------------------ cost & capacity
    (
        SELECT COALESCE(SUM(amc.estimated_cost_usd), 0)
        FROM ai_model_calls amc
        WHERE amc.called_at >= b.as_of - b.window_len
          AND amc.called_at < b.as_of
    ) AS spend_window_usd,
    (
        SELECT COALESCE(SUM(amc.estimated_cost_usd), 0)
        FROM ai_model_calls amc
        WHERE amc.called_at >= b.as_of - b.window_len_x2
          AND amc.called_at < b.as_of - b.window_len
    ) AS spend_prev_window_usd,
    (
        SELECT COALESCE(SUM(amc.total_tokens), 0)
        FROM ai_model_calls amc
        WHERE amc.called_at >= b.as_of - b.window_len
          AND amc.called_at < b.as_of
    ) AS tokens_window,
    (
        SELECT COUNT(*)
        FROM ai_model_calls amc
        WHERE amc.status = 'failed'
          AND amc.called_at >= b.as_of - b.window_len
          AND amc.called_at < b.as_of
    ) AS failed_ai_calls_window,
    (
        -- denominator for the AI failure rate: without it the UI cannot tell
        -- "no failures" from "no calls" (PRD section 5, no 0-of-0 as 0%).
        SELECT COUNT(*)
        FROM ai_model_calls amc
        WHERE amc.called_at >= b.as_of - b.window_len
          AND amc.called_at < b.as_of
    ) AS ai_calls_window,
    (
        SELECT CASE
                   WHEN ms.days_elapsed <= 0 THEN 0
                   ELSE ms.spend_mtd_usd * ms.days_in_month / ms.days_elapsed
               END
        FROM month_spend ms
    ) AS projected_month_end_usd,
    td.driver AS top_cost_driver,
    COALESCE(td.spend_usd, 0) AS top_cost_driver_usd,
    sl.model_name AS slowest_model,
    COALESCE(sl.p95_ms, 0) AS slowest_model_p95_ms,
    ------------------------------------------------------------------- usage
    (
        SELECT COUNT(*)
        FROM users u
        WHERE u.last_login_at >= b.as_of - INTERVAL '1 day'
          AND (CAST(:organization_id AS uuid) IS NULL
               OR EXISTS (
                   SELECT 1
                   FROM organization_memberships om
                   WHERE om.user_id = u.id
                     AND om.organization_id = CAST(:organization_id AS uuid)
                     AND om.deleted_at IS NULL
               ))
    ) AS active_users_today,
    (
        SELECT COUNT(*)
        FROM users u
        WHERE u.last_login_at >= b.as_of - b.window_len
          AND (CAST(:organization_id AS uuid) IS NULL
               OR EXISTS (
                   SELECT 1
                   FROM organization_memberships om
                   WHERE om.user_id = u.id
                     AND om.organization_id = CAST(:organization_id AS uuid)
                     AND om.deleted_at IS NULL
               ))
    ) AS active_users_window,
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
        FROM learning_materials lm
        JOIN lessons l ON l.id = lm.lesson_id AND l.deleted_at IS NULL
        JOIN modules m ON m.id = l.module_id AND m.deleted_at IS NULL
        JOIN courses c ON c.id = m.course_id AND c.deleted_at IS NULL
        WHERE lm.deleted_at IS NULL
          AND lm.created_at >= b.as_of - b.window_len
          AND lm.created_at < b.as_of
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS materials_ingested_window,
    --------------------------------------------------------- tenant anomalies
    -- The inactive-tenant COUNT is deliberately not here. It used to be an
    -- inlined copy of a four-way NOT EXISTS predicate that the organizations
    -- list did not have at all, so the dashboard could say "2 inactive" and
    -- link to a page showing every organization on the platform. Both now read
    -- ``access_control/queries/sql/inactive_organizations.sql`` through the
    -- feature's public API (PRD ADM-004 / ADM-045).
    (
        SELECT COUNT(*)
        FROM organizations o
        WHERE o.deleted_at IS NULL
          AND (CAST(:organization_id AS uuid) IS NULL
               OR o.id = CAST(:organization_id AS uuid))
    ) AS orgs_total
FROM bounds b
LEFT JOIN top_driver td ON TRUE
LEFT JOIN slowest sl ON TRUE;
