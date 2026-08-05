-- Per-status job counts over the SAME `since` window as the jobs list
-- (updated_at lower bound). Used for the admin processing page's status tab
-- badges: they must keep covering the whole window regardless of which status
-- the table is currently filtered by, and stay exact even when the window
-- holds more jobs than the list endpoint's row cap.
-- :since -- timestamptz lower bound on updated_at (required to bound the scan).
WITH combined AS (
    SELECT status FROM processing_jobs WHERE updated_at >= CAST(:since AS timestamptz)
    UNION ALL
    SELECT status FROM generation_runs WHERE updated_at >= CAST(:since AS timestamptz)
)
SELECT
    SUM(CASE WHEN status = 'pending'   THEN 1 ELSE 0 END) AS pending_count,
    SUM(CASE WHEN status = 'running'   THEN 1 ELSE 0 END) AS running_count,
    SUM(CASE WHEN status = 'failed'    THEN 1 ELSE 0 END) AS failed_count,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count,
    COUNT(*)                                              AS total_count
FROM combined;
