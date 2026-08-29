-- API reliability over a window -- error rate and latency percentiles for the
-- dashboard's "Reliability & Throughput" row (PRD section 3).
--
-- Source is ``http_audit_log``, which the audit middleware writes for every
-- non-exempt request (liveness probes are already skipped there, so they do
-- not dilute the rate). ``ix_http_audit_log_created_at`` covers the window
-- bound, keeping this a bounded scan.
--
-- Contract: ``requests_total`` is returned so the caller can tell "no errors"
-- from "no traffic" and render No data rather than a fabricated 0%
-- (PRD section 5). Percentiles are NULL on an empty window for the same
-- reason. 5xx is the server-error rate; 4xx is reported separately because a
-- burst of 401/403 is a security signal, not a reliability one.
--
-- :as_of         (timestamptz) -- as-of reference; tests pin it.
-- :window_start  (timestamptz) -- window start; the service derives it from
--   ``window_days`` (last N days ending now) or a caller date range, and the
--   dashboard's job metrics receive the same bounds (PRD ADM-004).
-- :window_end    (timestamptz) -- exclusive window end.
--
-- Scope: GLOBAL. http_audit_log records the acting user but not their
-- organization, so this metric cannot be tenant-filtered.
WITH bounds AS (
    SELECT
        CAST(:as_of AS timestamptz)         AS as_of,
        CAST(:window_start AS timestamptz)  AS window_start,
        CAST(:window_end AS timestamptz)    AS window_end
),
requests AS (
    SELECT h.status_code, h.latency_ms
    FROM http_audit_log h, bounds b
    WHERE h.created_at >= b.window_start
      AND h.created_at < b.window_end
)
SELECT
    b.as_of,
    b.window_start,
    (SELECT COUNT(*) FROM requests)                              AS requests_total,
    (SELECT COUNT(*) FROM requests WHERE status_code >= 500)     AS requests_5xx,
    (
        SELECT COUNT(*) FROM requests
        WHERE status_code >= 400 AND status_code < 500
    )                                                            AS requests_4xx,
    (
        SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)
        FROM requests
    )                                                            AS p50_latency_ms,
    (
        SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
        FROM requests
    )                                                            AS p95_latency_ms
FROM bounds b;
