"""ARQ reconciliation/retention task for interview recordings."""

from __future__ import annotations

from typing import Any

from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import bind_request_context, clear_request_context, get_logger
from abridgeai.features.interviews.services import recording as recording_service

_logger = get_logger(__name__)


async def reconcile_interview_recordings_task(ctx: dict[str, Any]) -> dict[str, int]:
    """Repair lost LiveKit callbacks and enforce the retention tombstone."""
    del ctx  # provider credentials are resolved by the recording service settings
    bind_request_context(task="reconcile_interview_recordings")
    try:
        async with get_sessionmaker()() as db:
            repaired = await recording_service.reconcile_recordings(db)
            expired = await recording_service.expire_retention_due_recordings(db)
            await db.commit()
        if repaired or expired:
            _logger.info(
                "reconciled_interview_recordings",
                repaired=repaired,
                expired=expired,
            )
        return {"repaired": repaired, "expired": expired}
    finally:
        clear_request_context()


__all__ = ["reconcile_interview_recordings_task"]
