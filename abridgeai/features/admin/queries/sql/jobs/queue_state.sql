-- Canonical queue state -- the as-of half of the job metric contract.
--
-- Contract (PRD section 5): "Queue depth = pending + running at an as-of
-- timestamp", displayed together with the oldest job age. It is a point-in-
-- time reading, NOT a windowed count, so it never carries a `window_days`.
--
-- Population and scope are identical to ``terminal_metrics.sql``: the union of
-- processing_jobs and generation_runs, global (no usable organization edge on
-- processing_jobs). Keeping both files on the same population is what lets the
-- dashboard's queue tile and the Operations jobs list agree.
--
-- Worker count (also named in the contract) is not here: it lives in the ARQ
-- pool, not in Postgres, and arrives with the Operations module.
--
-- :now (timestamptz) -- as-of reference; tests pin it.
WITH combined AS (
    SELECT pj.status, pj.created_at FROM processing_jobs pj
    UNION ALL
    SELECT gr.status, gr.created_at FROM generation_runs gr
),
in_flight AS (
    SELECT c.status, c.created_at
    FROM combined c
    WHERE c.status IN ('pending', 'running')
)
SELECT
    CAST(:now AS timestamptz) AS as_of,
    (SELECT COUNT(*) FROM in_flight)                          AS queue_depth,
    (SELECT COUNT(*) FROM in_flight WHERE status = 'pending') AS pending_count,
    (SELECT COUNT(*) FROM in_flight WHERE status = 'running') AS running_count,
    (
        -- NULL when the queue is empty: "no oldest job" is not "0 seconds".
        SELECT EXTRACT(
                   EPOCH FROM (CAST(:now AS timestamptz) - MIN(f.created_at))
               )::bigint
        FROM in_flight f
    ) AS oldest_age_seconds
;
