-- Backfill generation_runs (type 'interview_evaluation') for historical
-- interview evaluation + gap-report work, and attribute the orphaned
-- ai_model_calls (stage_name IN ('evaluation','gap_report'), generation_run_id
-- IS NULL) back to the run they belong to.
--
-- WHY THIS IS SAFE / CORRECT
--   * One generation_run per gap_reports row (each gap_report == one
--     evaluate_and_generate_report execution). Verified: 36 gap_reports, all
--     with non-NULL session/config/course/module.
--   * ai_model_calls.called_at is stamped at commit time, so every call in a
--     single evaluation shares one second-truncated timestamp. Verified 1:1:
--     36 distinct call-seconds each match exactly one gap_report, 0 orphans.
--   * Idempotent: runs are tagged config_json->>'backfill_gap_report_id'; the
--     INSERT is guarded by NOT EXISTS so a re-run inserts nothing, and the
--     UPDATE only touches rows still NULL.
--
-- Runs in one transaction; verification SELECTs print before COMMIT.

BEGIN;

-- 1) Create one 'interview_evaluation' run per historical gap_report.
INSERT INTO generation_runs (
    id, generation_type, source_scope_kind,
    course_id, module_id, lesson_id, requested_by,
    status, config_json, started_at, finished_at, created_at, updated_at
)
SELECT
    uuid_generate_v4(),
    'interview_evaluation',
    'module',
    ic.course_id,
    ic.module_id,
    NULL,
    s.student_id,
    'completed',
    jsonb_build_object(
        'interview_config_id', ic.id::text,
        'interview_session_id', s.id::text,
        'backfill_gap_report_id', g.id::text,
        'backfilled', true
    ),
    g.created_at,
    g.created_at,
    g.created_at,
    g.updated_at
FROM gap_reports g
JOIN interview_sessions s ON s.id = g.source_interview_session_id
JOIN interview_configs ic ON ic.id = s.interview_config_id
WHERE NOT EXISTS (
    SELECT 1 FROM generation_runs gr
    WHERE gr.generation_type = 'interview_evaluation'
      AND gr.config_json->>'backfill_gap_report_id' = g.id::text
);

-- 2) Attribute orphaned eval/gap calls to their run by second-truncated
--    timestamp match against the backfilled run's started_at.
UPDATE ai_model_calls amc
SET generation_run_id = gr.id
FROM generation_runs gr
WHERE amc.generation_run_id IS NULL
  AND amc.stage_name IN ('evaluation', 'gap_report')
  AND gr.generation_type = 'interview_evaluation'
  AND (gr.config_json->>'backfilled')::boolean IS TRUE
  AND date_trunc('second', amc.called_at) = date_trunc('second', gr.started_at);

-- 3) Verification.
\echo '=== backfilled runs (should be 36) ==='
SELECT COUNT(*) AS backfilled_runs
FROM generation_runs
WHERE generation_type = 'interview_evaluation'
  AND (config_json->>'backfilled')::boolean IS TRUE;

\echo '=== remaining orphan eval/gap calls (should be 0) ==='
SELECT COUNT(*) AS remaining_orphans
FROM ai_model_calls
WHERE generation_run_id IS NULL
  AND stage_name IN ('evaluation', 'gap_report');

\echo '=== calls now linked to backfilled runs (should be 123) ==='
SELECT COUNT(*) AS linked_calls
FROM ai_model_calls amc
JOIN generation_runs gr ON gr.id = amc.generation_run_id
WHERE (gr.config_json->>'backfilled')::boolean IS TRUE;

COMMIT;
