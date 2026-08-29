-- The individual AI calls one job made, newest first (PRD ADM-014).
--
-- The join already existed -- ``ai_model_calls.processing_job_id`` is a real
-- foreign key -- it was simply never surfaced, so an operator looking at a
-- failed job had to copy its id into the AI-cost page to find the call that
-- broke. Migration 0090 indexes the FK; before that this was a sequential scan
-- of every call ever recorded.
--
-- ``request_payload`` and ``response_payload`` are deliberately NOT projected.
-- They hold prompt and completion text -- student answers and course content --
-- and an operator triaging a failure needs the error, the model and the
-- latency, not the material. Surfacing them here would put course content into
-- an admin screen that has no business showing it.
--
-- :job_id (uuid), :limit (int)
SELECT
    amc.id                 AS id,
    amc.stage_name         AS stage_name,
    amc.role               AS role,
    amc.model_name         AS model_name,
    amc.operation          AS operation,
    amc.status             AS status,
    amc.error_message      AS error_message,
    amc.latency_ms         AS latency_ms,
    amc.input_tokens       AS input_tokens,
    amc.output_tokens      AS output_tokens,
    amc.total_tokens       AS total_tokens,
    amc.estimated_cost_usd AS estimated_cost_usd,
    amc.request_id         AS request_id,
    amc.called_at          AS called_at
FROM ai_model_calls amc
WHERE amc.processing_job_id = CAST(:job_id AS uuid)
ORDER BY amc.called_at DESC
LIMIT :limit;
