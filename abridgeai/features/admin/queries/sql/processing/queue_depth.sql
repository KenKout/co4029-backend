-- Aggregate queue depth by status (single row).
SELECT
    SUM(CASE WHEN status = 'pending'   THEN 1 ELSE 0 END) AS pending_count,
    SUM(CASE WHEN status = 'running'   THEN 1 ELSE 0 END) AS running_count,
    SUM(CASE WHEN status = 'failed'    THEN 1 ELSE 0 END) AS failed_count,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count,
    COUNT(*)                                              AS total_count
FROM processing_jobs;
