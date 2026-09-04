"""ARQ cron task: prune the append-only audit stores.

The HTTP audit middleware writes one row per non-skipped request and, until this
job existed, nothing ever deleted one — ``http_audit_log`` grew for the lifetime
of the deployment. The append-only triggers (migrations 0105 / 0106) are what
make that safe to automate: ordinary code cannot remove an audit row even by
accident, and this job is the single explicit exception, visible in the cron
registry and in the diff.

Scheduled at 04:00 — after the 02:00 completion drift sweep, the 02:30 readiness
snapshots and the 03:00 orphaned-upload cleanup, so a long first prune cannot
delay work that other reporting depends on.
"""

from __future__ import annotations

from typing import Any

from abridgeai.core.audit.retention import prune_audit_logs
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import get_logger

logger = get_logger(__name__)


async def prune_audit_logs_task(ctx: dict[str, Any]) -> int:
    """Prune every retained audit table; returns the total rows removed."""
    del ctx
    session_factory = get_sessionmaker()
    results = await prune_audit_logs(session_factory)
    total = sum(results.values())
    # Info even at zero: a nightly line confirming the sweep ran is what
    # distinguishes "nothing to prune" from "the job silently stopped firing",
    # which is how the unbounded growth went unnoticed in the first place.
    logger.info("audit.retention_sweep", total_deleted=total, **results)
    return total


__all__ = ["prune_audit_logs_task"]
