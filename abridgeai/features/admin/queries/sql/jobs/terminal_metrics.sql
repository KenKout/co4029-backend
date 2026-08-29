-- Canonical job outcome metrics -- the ONE definition of "how many jobs ran
-- and how many failed" shared by the operator dashboard and the Operations
-- module (PRD ADM-004: Dashboard / Processing / Content must not disagree).
--
-- Contract (PRD section 5):
--   Job failure rate = failed TERMINAL jobs / all TERMINAL jobs, in the same
--   window and the same job scope. A window with no terminal jobs has NO
--   failure rate -- the caller must render "No data", never 0%.
--
-- Population: processing_jobs UNION ALL generation_runs -- identical to
--   ``processing/list_jobs.sql``, so the number on the dashboard equals the
--   row count an operator gets when they click through to the jobs list.
-- Window key: ``updated_at`` -- also identical to list_jobs.sql. For a
--   terminal row updated_at is the moment it reached that terminal state, and
--   matching the list's key is what makes drill-down add up.
-- Terminal statuses: completed, failed, cancelled. pending / running are
--   in-flight and belong to queue depth (``queue_state.sql``), not to a rate.
--
-- Scope: GLOBAL. ``processing_jobs`` carries no organization edge and its
--   ``entity_id`` is a polymorphic reference with no FK, so it cannot be
--   filtered by organization. ``generation_runs`` could be (via course_id),
--   but scoping half the population would produce a number that is neither
--   global nor tenant-accurate. Callers surface this as scope="global".
--
-- :as_of          (timestamptz) -- evaluation reference; tests pin it.
-- :current_start  (timestamptz) -- current window start; the stats service
--   derives the same bounds for the dashboard rollup and API reliability, so
--   every tile describes one identical span (PRD ADM-004). The preceding
--   window is the same length immediately before it.
-- :previous_start (timestamptz) -- previous window start.
-- :current_end    (timestamptz) -- exclusive end of the current window.
--   (The single-query contract is shared with the processing surface, which
--   always builds it from ``window_days`` ending at ``now``.)
WITH bounds AS (
    SELECT
        CAST(:as_of AS timestamptz)          AS as_of,
        CAST(:current_start AS timestamptz)  AS current_start,
        CAST(:current_end AS timestamptz)    AS current_end,
        CAST(:previous_start AS timestamptz) AS previous_start
),
combined AS (
    SELECT pj.status, pj.updated_at FROM processing_jobs pj
    UNION ALL
    SELECT gr.status, gr.updated_at FROM generation_runs gr
),
terminal AS (
    SELECT c.status, c.updated_at
    FROM combined c
    WHERE c.status IN ('completed', 'failed', 'cancelled')
)
SELECT
    b.as_of,
    b.current_start,
    b.previous_start,
    (
        SELECT COUNT(*) FROM terminal t, bounds bb
        WHERE t.updated_at >= bb.current_start AND t.updated_at < bb.current_end
    ) AS terminal_total,
    (
        SELECT COUNT(*) FROM terminal t, bounds bb
        WHERE t.status = 'failed'
          AND t.updated_at >= bb.current_start AND t.updated_at < bb.current_end
    ) AS terminal_failed,
    (
        SELECT COUNT(*) FROM terminal t, bounds bb
        WHERE t.updated_at >= bb.previous_start AND t.updated_at < bb.current_start
    ) AS prev_terminal_total,
    (
        SELECT COUNT(*) FROM terminal t, bounds bb
        WHERE t.status = 'failed'
          AND t.updated_at >= bb.previous_start AND t.updated_at < bb.current_start
    ) AS prev_terminal_failed
FROM bounds b;
