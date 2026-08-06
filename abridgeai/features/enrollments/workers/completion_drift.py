"""ARQ cron task: repair drifted ``course_enrollments.status`` rows.

The D2 course-completion writer fires synchronously from every place a
gradeable unit can change state (lesson progress, quiz grading, interview
evaluation). Each of those call sites deliberately swallows its own
exceptions — a student's mark-complete, quiz submit or interview evaluation
must never fail because a bookkeeping side-effect broke.

That safety has a cost: a lost write is silent and permanent. And
``course_enrollments.status`` is not a cosmetic column — career-path stage
unlock reads it as ``satisfied``, so drift either hands out a stage nobody
earned or withholds one that was.

This nightly sweep is the backstop that closes the gap. It recomputes every
``active``/``completed`` enrollment and logs each repair, so a recurring
non-zero ``fixed`` count is a signal that some synchronous call site is
failing and should be investigated rather than quietly patched over.

Scheduled at 02:00, ahead of the 02:30 readiness snapshots, so those snapshots
aggregate already-repaired completion state.
"""

from __future__ import annotations

from typing import Any

from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import get_logger
from abridgeai.features.enrollments.services import completion as completion_service

logger = get_logger(__name__)


async def resync_course_completions_task(ctx: dict[str, Any]) -> int:
    """Recompute every eligible enrollment; returns the number repaired."""
    del ctx
    session_factory = get_sessionmaker()
    async with session_factory() as db:
        scanned, fixed = await completion_service.resync_stale_course_completions(db)
        await db.commit()
    # Logged at warning when anything was repaired: in steady state the
    # synchronous writers should leave nothing to fix, so a non-zero count
    # means a call site is dropping writes.
    log = logger.warning if fixed else logger.info
    log("enrollments.completion_drift_sweep", scanned=scanned, fixed=fixed)
    return fixed


__all__ = ["resync_course_completions_task"]
