"""ARQ cron task: sweep overdue in-progress quiz attempts (Phase 6).

Registered in ``WorkerSettings.cron_jobs``. Every tick it finalizes attempts
past their (grace-extended) deadline — grading them (autosubmit / graceperiod)
or expiring them (autoabandon) — so a timed quiz closes even if the student
never hits submit. Runs unauthenticated inside the trusted worker process; no
network endpoint is added.
"""

from __future__ import annotations

from typing import Any

from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.quizzes.services.sweep import sweep_overdue

_logger = get_logger(__name__)


async def sweep_overdue_attempts_task(ctx: dict[str, Any]) -> dict[str, int]:
    """Finalize/expire overdue in_progress attempts; returns counts."""
    del ctx
    bind_request_context(task="sweep_overdue_attempts")
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            result = await sweep_overdue(db)
            await db.commit()
        if result.get("submitted") or result.get("expired"):
            _logger.info("swept_overdue_quiz_attempts", **result)
        return result
    finally:
        clear_request_context()


__all__ = ["sweep_overdue_attempts_task"]
