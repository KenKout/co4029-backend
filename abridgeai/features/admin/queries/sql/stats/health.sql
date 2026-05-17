-- System health snapshot: failed jobs in the last :since window + recent error
-- ai_model_calls. Bounded by :since to prevent unbounded scans.
SELECT
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.status = 'failed' AND pj.updated_at >= CAST(:since AS timestamptz)
    ) AS failed_jobs_count,
    (
        SELECT COUNT(*)
        FROM processing_jobs pj
        WHERE pj.status IN ('pending', 'running') AND pj.updated_at >= CAST(:since AS timestamptz)
    ) AS in_flight_jobs_count,
    (
        SELECT COUNT(*)
        FROM ai_model_calls amc
        WHERE amc.status = 'failed' AND amc.called_at >= CAST(:since AS timestamptz)
    ) AS failed_ai_calls_count;
