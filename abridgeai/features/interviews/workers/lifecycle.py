"""ARQ cron task: sweep stale in-progress voice interview sessions (Phase 4).

Runs periodically (registered in ``WorkerSettings.cron_jobs``). Finalises voice
sessions past their deadline — either the configured ``time_limit_minutes`` or,
when none is set, the fixed idle window (``interview_voice_idle_timeout_minutes``)
— marking them ``timed_out`` (with async evaluation enqueued) when there is a
transcript, else ``abandoned``.

``ctx['redis']`` is the ARQ pool, passed to the lifecycle service so it can
enqueue the evaluation job for timed-out sessions.
"""

from __future__ import annotations

from typing import Any

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import bind_request_context, clear_request_context, get_logger
from abridgeai.features.interviews.services import lifecycle as lifecycle_service

_logger = get_logger(__name__)


async def sweep_interview_sessions_task(ctx: dict[str, Any]) -> int:
    """Finalise stale voice sessions; returns count finalised."""
    bind_request_context(task="sweep_interview_sessions")
    arq_pool = ctx.get("redis")
    idle_timeout_minutes = get_settings().interview_voice_idle_timeout_minutes
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            count = await lifecycle_service.sweep_stale_voice_sessions(
                db, arq_pool=arq_pool, idle_timeout_minutes=idle_timeout_minutes
            )
            recovered = await lifecycle_service.recover_stalled_evaluations(
                db,
                arq_pool=arq_pool,
            )
        if count:
            _logger.info("swept_stale_voice_sessions", finalised=count)
        if recovered:
            _logger.info("recovered_stalled_interview_evaluations", enqueued=recovered)
        return count
    finally:
        clear_request_context()


__all__ = ["sweep_interview_sessions_task"]
