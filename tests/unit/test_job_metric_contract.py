"""The job metric contract: one rate, and no rate at all when there is no data.

PRD section 5 turns two habits into requirements. First, "0 of 0" must not be
reported as 0% -- a window with no terminal jobs and a window where everything
succeeded are different facts, and a tile that renders both as a green zero
teaches operators to stop reading it. Second, the failure rate divides by
TERMINAL jobs only; counting pending and running work in the denominator (what
the old dashboard SQL did) silently deflates the rate exactly when the queue is
backing up and the number matters most.

These tests pin the pure arithmetic. The population and window agreement with
the processing surface is pinned by the SQL living in one file per contract
(``sql/jobs/*.sql``) and asserted end-to-end in
``tests/integration/test_admin.py``.
"""

from datetime import UTC, datetime

from abridgeai.features.admin.services.job_metrics import (
    JOB_METRIC_SCOPE,
    JobOutcomeMetrics,
    QueueState,
    _rate_pct,
)
from abridgeai.features.admin.services.stats import _opt_int
from abridgeai.features.admin.services.stats import _rate_pct as stats_rate_pct

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def test_empty_window_has_no_rate() -> None:
    assert _rate_pct(0, 0) is None


def test_clean_window_is_zero_not_none() -> None:
    """Nothing failed out of 40 jobs is a real 0%, and must survive as one."""
    assert _rate_pct(0, 40) == 0.0


def test_rate_is_rounded_to_two_places() -> None:
    assert _rate_pct(1, 3) == 33.33


def test_all_failed_is_one_hundred() -> None:
    assert _rate_pct(7, 7) == 100.0


def test_stats_and_job_metrics_share_one_rate_definition() -> None:
    """Both services round and null identically; a tile fed by either agrees."""
    for failed, total in ((0, 0), (0, 5), (1, 3), (2, 7), (9, 9)):
        assert _rate_pct(failed, total) == stats_rate_pct(failed, total)


def test_negative_denominator_is_treated_as_no_data() -> None:
    """Defensive: a COUNT can never go negative, but None beats a bogus rate."""
    assert _rate_pct(0, -1) is None


def test_job_outcomes_declare_global_scope() -> None:
    """processing_jobs has no organization edge, so the scope is never a lie."""
    metrics = JobOutcomeMetrics(
        as_of=NOW,
        window_days=7,
        window_start=NOW,
        terminal_total=0,
        terminal_failed=0,
        failure_rate_pct=None,
        prev_terminal_total=0,
        prev_terminal_failed=0,
        prev_failure_rate_pct=None,
    )
    assert metrics.scope == JOB_METRIC_SCOPE == "global"


def test_empty_queue_has_no_oldest_job() -> None:
    """An empty queue is not a queue whose oldest job is zero seconds old."""
    queue = QueueState(
        as_of=NOW,
        queue_depth=0,
        pending_count=0,
        running_count=0,
        oldest_age_seconds=None,
    )
    assert queue.oldest_age_seconds is None
    assert queue.scope == "global"


def test_percentiles_preserve_absence() -> None:
    """percentile_cont over an empty window returns NULL, not 0ms."""
    assert _opt_int(None) is None
    assert _opt_int(12.6) == 13
    assert _opt_int(0.0) == 0
