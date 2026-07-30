-- Operator dashboard: one row of needs-action / cost / activity / checklist
-- metrics for ``GET /admin/stats/dashboard``.
--
-- :organization_id (uuid | NULL) -- when NULL, global aggregates; when provided,
--   filters every metric that can be traced to an organization.
-- :now (timestamptz)             -- evaluation reference timestamp; tests pin
--   this so all windows are deterministic.
--
-- Org-traceability notes (metrics that IGNORE :organization_id because the
-- schema has no usable org edge):
--   * processing_jobs  -- no organization_id; entity_id is a polymorphic ref
--     (entity_type in material_version/lesson/quiz/interview_config/
--     generation_run) with no FK, so it cannot be joined to courses safely.
--     Affects: job_failure_rate_pct, jobs_failed_7d, jobs_total_7d,
--     queue_depth, materials_stuck_processing.
--   * ai_model_calls   -- no organization_id and no mandatory parent (the
--     ck_ai_model_calls_parent_ref check allows stage_name-only rows).
--     Affects: failed_ai_calls_30d, spend_*, projected_month_end_usd,
--     top_cost_driver*, slowest_model*.
--
-- Org-scoped edges used elsewhere:
--   users              -> organization_memberships
--   quiz_attempts      -> quizzes.course_id -> courses.organization_id
--   interview_sessions -> interview_configs.course_id -> courses
--   learning_materials -> lessons -> modules -> courses
--   quizzes            -> courses
--   interview_configs  -> courses
--   organizations      -> id
WITH top_driver AS (
    SELECT
        COALESCE(amc.stage_name, amc.role) AS driver,
        SUM(amc.estimated_cost_usd)        AS spend_usd
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
      AND amc.called_at < CAST(:now AS timestamptz)
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
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
      AND amc.called_at < CAST(:now AS timestamptz)
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
    -------------------------------------------------------------- needs action
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.status = 'failed'
          AND pj.created_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND pj.created_at < CAST(:now AS timestamptz)
    ) AS jobs_failed_7d,
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.created_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND pj.created_at < CAST(:now AS timestamptz)
    ) AS jobs_total_7d,
    -- Prior 7d window, so the UI can show whether the failure rate is
    -- improving or actively degrading rather than just its current level.
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.status = 'failed'
          AND pj.created_at >= CAST(:now AS timestamptz) - INTERVAL '14 days'
          AND pj.created_at < CAST(:now AS timestamptz) - INTERVAL '7 days'
    ) AS jobs_failed_prev_7d,
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.created_at >= CAST(:now AS timestamptz) - INTERVAL '14 days'
          AND pj.created_at < CAST(:now AS timestamptz) - INTERVAL '7 days'
    ) AS jobs_total_prev_7d,
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.status IN ('pending', 'running')
    ) AS queue_depth,
    (
        SELECT COUNT(*)
        FROM ai_model_calls amc
        WHERE amc.status = 'failed'
          AND amc.called_at >= CAST(:now AS timestamptz) - INTERVAL '30 days'
          AND amc.called_at < CAST(:now AS timestamptz)
    ) AS failed_ai_calls_30d,
    ------------------------------------------------------------- cost snapshot
    (
        SELECT COALESCE(SUM(amc.estimated_cost_usd), 0)
        FROM ai_model_calls amc
        WHERE amc.called_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND amc.called_at < CAST(:now AS timestamptz)
    ) AS spend_7d_usd,
    (
        SELECT COALESCE(SUM(amc.estimated_cost_usd), 0)
        FROM ai_model_calls amc
        WHERE amc.called_at >= CAST(:now AS timestamptz) - INTERVAL '14 days'
          AND amc.called_at < CAST(:now AS timestamptz) - INTERVAL '7 days'
    ) AS spend_prev_7d_usd,
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
    ----------------------------------------------------------------- activity
    (
        SELECT COUNT(*)
        FROM users u
        WHERE u.last_login_at >= CAST(:now AS timestamptz) - INTERVAL '1 day'
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
        WHERE u.last_login_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND (CAST(:organization_id AS uuid) IS NULL
               OR EXISTS (
                   SELECT 1
                   FROM organization_memberships om
                   WHERE om.user_id = u.id
                     AND om.organization_id = CAST(:organization_id AS uuid)
                     AND om.deleted_at IS NULL
               ))
    ) AS active_users_7d,
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
        FROM quiz_attempts qa
        JOIN quizzes q ON q.id = qa.quiz_id AND q.deleted_at IS NULL
        JOIN courses c ON c.id = q.course_id AND c.deleted_at IS NULL
        WHERE qa.status IN ('submitted', 'graded')
          AND qa.submitted_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND qa.submitted_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS quiz_sessions_completed_7d,
    (
        SELECT COUNT(*)
        FROM interview_sessions isx
        JOIN interview_configs ic
          ON ic.id = isx.interview_config_id AND ic.deleted_at IS NULL
        JOIN courses c ON c.id = ic.course_id AND c.deleted_at IS NULL
        WHERE isx.started_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND isx.started_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS interview_sessions_7d,
    (
        -- pass rate over 7d sessions that actually reached a verdict
        SELECT COALESCE(
                   ROUND(
                       100.0 * COUNT(*) FILTER (WHERE isx.pass_verdict)
                       / NULLIF(COUNT(*), 0),
                       2
                   ),
                   0
               )
        FROM interview_sessions isx
        JOIN interview_configs ic
          ON ic.id = isx.interview_config_id AND ic.deleted_at IS NULL
        JOIN courses c ON c.id = ic.course_id AND c.deleted_at IS NULL
        WHERE isx.pass_verdict IS NOT NULL
          AND isx.started_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND isx.started_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS interview_pass_rate_pct,
    -- Sample size behind the pass rate. A low rate over a handful of sessions
    -- from one or two students is a testing artifact, not a platform signal;
    -- the UI needs these to decide whether to raise an alarm or caption it.
    (
        SELECT COUNT(*)
        FROM interview_sessions isx
        JOIN interview_configs ic
          ON ic.id = isx.interview_config_id AND ic.deleted_at IS NULL
        JOIN courses c ON c.id = ic.course_id AND c.deleted_at IS NULL
        WHERE isx.pass_verdict IS NOT NULL
          AND isx.started_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND isx.started_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS interview_evaluated_7d,
    (
        SELECT COUNT(DISTINCT isx.student_id)
        FROM interview_sessions isx
        JOIN interview_configs ic
          ON ic.id = isx.interview_config_id AND ic.deleted_at IS NULL
        JOIN courses c ON c.id = ic.course_id AND c.deleted_at IS NULL
        WHERE isx.started_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND isx.started_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS interview_students_7d,
    (
        SELECT COUNT(*)
        FROM learning_materials lm
        JOIN lessons l ON l.id = lm.lesson_id AND l.deleted_at IS NULL
        JOIN modules m ON m.id = l.module_id AND m.deleted_at IS NULL
        JOIN courses c ON c.id = m.course_id AND c.deleted_at IS NULL
        WHERE lm.deleted_at IS NULL
          AND lm.created_at >= CAST(:now AS timestamptz) - INTERVAL '7 days'
          AND lm.created_at < CAST(:now AS timestamptz)
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS materials_ingested_7d,
    --------------------------------------------------------- needs attention
    (
        -- non-terminal processing jobs older than 1h (stuck pipeline)
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.status IN ('pending', 'running')
          AND pj.created_at < CAST(:now AS timestamptz) - INTERVAL '1 hour'
    ) AS materials_stuck_processing,
    (
        SELECT COUNT(DISTINCT q.id)
        FROM quizzes q
        JOIN courses c ON c.id = q.course_id AND c.deleted_at IS NULL
        JOIN quiz_questions qq
          ON qq.quiz_id = q.id
         AND qq.deleted_at IS NULL
         AND qq.review_status = 'approved'
         AND (qq.expected_response_time_ms IS NULL
              OR qq.expected_response_time_ms <= 0)
        WHERE q.status = 'published'
          AND q.deleted_at IS NULL
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS published_quizzes_missing_texp,
    (
        -- configs with zero human-cleared (approved/edited) questions
        SELECT COUNT(*)
        FROM interview_configs ic
        JOIN courses c ON c.id = ic.course_id AND c.deleted_at IS NULL
        WHERE ic.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1
              FROM interview_questions iq
              WHERE iq.interview_config_id = ic.id
                AND iq.deleted_at IS NULL
                AND iq.review_status IN ('approved', 'edited')
          )
          AND (CAST(:organization_id AS uuid) IS NULL
               OR c.organization_id = CAST(:organization_id AS uuid))
    ) AS interview_configs_no_reviewed_questions,
    (
        -- no member login, no course edit, no attempt and no interview in 30d
        SELECT COUNT(*)
        FROM organizations o
        WHERE o.deleted_at IS NULL
          AND (CAST(:organization_id AS uuid) IS NULL
               OR o.id = CAST(:organization_id AS uuid))
          AND NOT EXISTS (
              SELECT 1
              FROM organization_memberships om
              JOIN users u ON u.id = om.user_id
              WHERE om.organization_id = o.id
                AND om.deleted_at IS NULL
                AND u.last_login_at
                    >= CAST(:now AS timestamptz) - INTERVAL '30 days'
          )
          AND NOT EXISTS (
              SELECT 1
              FROM courses c2
              WHERE c2.organization_id = o.id
                AND c2.deleted_at IS NULL
                AND c2.updated_at
                    >= CAST(:now AS timestamptz) - INTERVAL '30 days'
          )
          AND NOT EXISTS (
              SELECT 1
              FROM quiz_attempts qa2
              JOIN quizzes q2 ON q2.id = qa2.quiz_id
              JOIN courses c3 ON c3.id = q2.course_id
              WHERE c3.organization_id = o.id
                AND qa2.started_at
                    >= CAST(:now AS timestamptz) - INTERVAL '30 days'
          )
          AND NOT EXISTS (
              SELECT 1
              FROM interview_sessions is2
              JOIN interview_configs ic2 ON ic2.id = is2.interview_config_id
              JOIN courses c4 ON c4.id = ic2.course_id
              WHERE c4.organization_id = o.id
                AND is2.started_at
                    >= CAST(:now AS timestamptz) - INTERVAL '30 days'
          )
    ) AS orgs_inactive_30d
FROM (SELECT 1) AS anchor(one)
LEFT JOIN top_driver td ON TRUE
LEFT JOIN slowest sl ON TRUE;
