"""ARQ cron task: sweep expired timed interview sessions (Phase 4).

Runs periodically (registered in ``WorkerSettings.cron_jobs``). Finalises only
sessions whose configured ``time_limit_minutes`` deadline has elapsed, marking
them ``timed_out`` (with async evaluation enqueued) when there is a transcript,
else ``abandoned``. Untimed sessions remain resumable.

``ctx['redis']`` is the ARQ pool, passed to the lifecycle service so it can
enqueue the evaluation job for timed-out sessions.
"""

from __future__ import annotations

from typing import Any

from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import bind_request_context, clear_request_context, get_logger
from abridgeai.features.interviews.services import lifecycle as lifecycle_service

_logger = get_logger(__name__)


async def sweep_interview_sessions_task(ctx: dict[str, Any]) -> int:
    """Finalise expired timed interview sessions; returns count finalised."""
    bind_request_context(task="sweep_interview_sessions")
    arq_pool = ctx.get("redis")
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            count = await lifecycle_service.sweep_expired_interview_sessions(
                db, arq_pool=arq_pool
            )
            recovered = await lifecycle_service.recover_stalled_evaluations(
                db,
                arq_pool=arq_pool,
            )
        if count:
            _logger.info("swept_expired_interview_sessions", finalised=count)
        if recovered:
            _logger.info("recovered_stalled_interview_evaluations", enqueued=recovered)
        return count
    finally:
        clear_request_context()


__all__ = ["sweep_interview_sessions_task"]
