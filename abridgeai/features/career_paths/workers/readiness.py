"""ARQ cron task: nightly career readiness snapshots (FR-6.8).

Writes one ``career_readiness_snapshots`` row per active career
enrollment so managers get historical, org-scoped aggregates. Mirrors
the session/logging pattern of
``features.interviews.workers.lifecycle``.
"""

from __future__ import annotations

from typing import Any

from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import get_logger
from abridgeai.features.career_paths.services import readiness as readiness_service

logger = get_logger(__name__)


async def snapshot_career_readiness_task(ctx: dict[str, Any]) -> int:
    """Snapshot every active career enrollment; returns rows written."""
    del ctx
    session_factory = get_sessionmaker()
    async with session_factory() as db:
        written = await readiness_service.snapshot_all_active_enrollments(db)
        await db.commit()
    logger.info("career_readiness.snapshots_written", count=written)
    return written


__all__ = ["snapshot_career_readiness_task"]
