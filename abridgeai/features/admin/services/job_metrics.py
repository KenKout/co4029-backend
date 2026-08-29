"""Job metric contract -- one failure rate, one queue reading (PRD ADM-004).

Every operator surface that reports on jobs goes through here, so the number
on the dashboard, the number on the Operations module and the rows in the jobs
list are the same number computed once.

The type that matters is :class:`JobOutcomeMetrics.failure_rate_pct`: it is
``None``, not ``0.0``, when the window held no terminal jobs. PRD section 5 is
explicit that "0 of 0" must not render as 0% -- a quiet platform and a healthy
platform look identical on a tile that fabricates a zero, and operators learn
to distrust the tile. Callers render ``None`` as "No data".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from abridgeai.features.admin.queries import job_metrics as job_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Job aggregates are always global: ``processing_jobs`` has no organization
# edge (see sql/jobs/terminal_metrics.sql). Surfaced verbatim to the client so
# an org-filtered dashboard can say which of its tiles the filter did not
# reach, rather than implying a tenant-accurate number.
JOB_METRIC_SCOPE = "global"


def _rate_pct(failed: int, total: int) -> float | None:
    """Failure percentage, or ``None`` when there is nothing to divide."""
    if total <= 0:
        return None
    return round(100.0 * failed / total, 2)


@dataclass(frozen=True)
class JobOutcomeMetrics:
    """Terminal job outcomes for a window and the window before it."""

    as_of: datetime
    window_days: int
    window_start: datetime
    terminal_total: int
    terminal_failed: int
    failure_rate_pct: float | None
    prev_terminal_total: int
    prev_terminal_failed: int
    prev_failure_rate_pct: float | None
    scope: str = JOB_METRIC_SCOPE


@dataclass(frozen=True)
class QueueState:
    """Point-in-time queue reading. ``oldest_age_seconds`` is ``None`` when
    the queue is empty -- an empty queue has no oldest job, which is not the
    same statement as "the oldest job is 0 seconds old"."""

    as_of: datetime
    queue_depth: int
    pending_count: int
    running_count: int
    oldest_age_seconds: int | None
    scope: str = JOB_METRIC_SCOPE


async def job_outcomes(
    db: AsyncSession,
    *,
    window_days: int,
    now: datetime | None = None,
    current_start: datetime | None = None,
    current_end: datetime | None = None,
    previous_start: datetime | None = None,
) -> JobOutcomeMetrics:
    """Terminal job counts, current window + the one before it.

    ``window_days`` / ``now`` is the relative shape (last N days ending
    ``now``). Callers that already resolved an explicit window (the stats
    service, whose dashboard must describe one identical span everywhere)
    pass ``current_start`` / ``current_end`` / ``previous_start`` instead.
    """
    if current_start is None or current_end is None or previous_start is None:
        reference = now or datetime.now(tz=UTC)
        current_start = reference - timedelta(days=window_days)
        current_end = reference
        previous_start = reference - timedelta(days=window_days * 2)
    row = await job_queries.terminal_metrics(
        db,
        as_of=now or datetime.now(tz=UTC),
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
    )
    total = int(row["terminal_total"] or 0)
    failed = int(row["terminal_failed"] or 0)
    prev_total = int(row["prev_terminal_total"] or 0)
    prev_failed = int(row["prev_terminal_failed"] or 0)
    return JobOutcomeMetrics(
        as_of=row["as_of"],
        window_days=window_days,
        window_start=row["current_start"],
        terminal_total=total,
        terminal_failed=failed,
        failure_rate_pct=_rate_pct(failed, total),
        prev_terminal_total=prev_total,
        prev_terminal_failed=prev_failed,
        prev_failure_rate_pct=_rate_pct(prev_failed, prev_total),
    )


async def queue_state(
    db: AsyncSession, *, now: datetime | None = None
) -> QueueState:
    row = await job_queries.queue_state(db, now=now or datetime.now(tz=UTC))
    oldest = row["oldest_age_seconds"]
    return QueueState(
        as_of=row["as_of"],
        queue_depth=int(row["queue_depth"] or 0),
        pending_count=int(row["pending_count"] or 0),
        running_count=int(row["running_count"] or 0),
        oldest_age_seconds=None if oldest is None else int(oldest),
    )


__all__ = [
    "JOB_METRIC_SCOPE",
    "JobOutcomeMetrics",
    "QueueState",
    "job_outcomes",
    "queue_state",
]
