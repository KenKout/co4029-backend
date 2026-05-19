-- Failed (or any-status) processing_jobs filtered by status + since window.
-- :status     -- string in ('pending','running','completed','failed','cancelled') or NULL for all.
-- :since      -- timestamptz lower bound on updated_at (required to bound the scan).
-- :limit      -- row cap.
SELECT
    pj.id              AS id,
    pj.entity_type     AS entity_type,
    pj.entity_id       AS entity_id,
    pj.job_type        AS job_type,
    pj.status          AS status,
    pj.progress_percent AS progress_percent,
    pj.error_message   AS error_message,
    pj.retry_count     AS retry_count,
    pj.started_at      AS started_at,
    pj.finished_at     AS finished_at,
    pj.created_at      AS created_at,
    pj.updated_at      AS updated_at
FROM processing_jobs pj
WHERE pj.updated_at >= CAST(:since AS timestamptz)
  AND (CAST(:status AS text) IS NULL OR pj.status = CAST(:status AS text))
ORDER BY pj.updated_at DESC
LIMIT :limit;
