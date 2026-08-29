-- AI spend attributed to organizations (PRD ADM-040).
--
-- ``ai_model_calls`` has no organization column, which is why every cost
-- number on the dashboard is labelled "global". It does have two optional
-- parents, and each of those CAN be walked to a course and therefore a tenant:
--
--   generation_run_id -> generation_runs.course_id  -> courses.organization_id
--   processing_job_id -> processing_jobs.entity_id  -> (per entity_type) -> course
--
-- A third kind of call has neither: session-runtime calls attribute via
-- ``stage_name`` alone (interview follow-ups, live hints). Those are real spend
-- with no derivable tenant, and they are reported as an explicit
-- ``unattributed`` bucket rather than dropped. Dropping them would make the
-- per-organization totals silently fail to add up to the platform total, which
-- is the specific way a cost breakdown becomes untrustworthy.
--
-- The caller reports coverage (attributed spend / total spend) so an operator
-- can see how much of the bill this view actually explains before acting on it.
--
-- :since (timestamptz), :until (timestamptz), :limit (int)
WITH scoped AS (
    SELECT
        amc.id,
        amc.estimated_cost_usd,
        amc.total_tokens,
        amc.status,
        COALESCE(
            (
                SELECT gr.course_id
                FROM generation_runs gr
                WHERE gr.id = amc.generation_run_id
            ),
            (
                SELECT CASE pj.entity_type
                    WHEN 'material_version' THEN (
                        SELECT m.course_id
                        FROM learning_material_versions lmv
                        JOIN learning_materials lm ON lm.id = lmv.material_id
                        JOIN lessons l ON l.id = lm.lesson_id
                        JOIN modules m ON m.id = l.module_id
                        WHERE lmv.id = pj.entity_id
                    )
                    WHEN 'lesson' THEN (
                        SELECT m.course_id
                        FROM lessons l
                        JOIN modules m ON m.id = l.module_id
                        WHERE l.id = pj.entity_id
                    )
                    WHEN 'quiz' THEN (
                        SELECT q.course_id FROM quizzes q WHERE q.id = pj.entity_id
                    )
                    WHEN 'interview_config' THEN (
                        SELECT ic.course_id
                        FROM interview_configs ic
                        WHERE ic.id = pj.entity_id
                    )
                    WHEN 'generation_run' THEN (
                        SELECT gr2.course_id
                        FROM generation_runs gr2
                        WHERE gr2.id = pj.entity_id
                    )
                END
                FROM processing_jobs pj
                WHERE pj.id = amc.processing_job_id
            )
        ) AS course_id
    FROM ai_model_calls amc
    WHERE amc.called_at >= CAST(:since AS timestamptz)
      AND amc.called_at < CAST(:until AS timestamptz)
),
attributed AS (
    SELECT
        c.organization_id,
        s.estimated_cost_usd,
        s.total_tokens,
        s.status
    FROM scoped s
    LEFT JOIN courses c ON c.id = s.course_id
)
SELECT
    a.organization_id,
    o.name AS organization_name,
    COUNT(*)                                       AS call_count,
    COUNT(*) FILTER (WHERE a.status = 'failed')    AS failed_count,
    COALESCE(SUM(a.estimated_cost_usd), 0)         AS spend_usd,
    COALESCE(SUM(a.total_tokens), 0)               AS tokens
FROM attributed a
LEFT JOIN organizations o ON o.id = a.organization_id
GROUP BY a.organization_id, o.name
-- NULLS LAST keeps the unattributed bucket at the bottom of the table even
-- when it is the largest single row, so it reads as a caveat rather than as
-- the top spender.
ORDER BY a.organization_id IS NULL, spend_usd DESC NULLS LAST
LIMIT :limit;
