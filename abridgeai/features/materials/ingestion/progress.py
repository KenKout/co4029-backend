"""Live ingest-progress tracking via Redis (lock-free real-time UX).

The materials ingest pipeline runs the entire multi-stage job inside a
single database transaction that only commits at the very end. Because of
Postgres MVCC, any ``progress_percent`` the pipeline writes to
``processing_jobs`` is invisible to other connections (the polling
endpoint) until that final commit — so a DB-backed progress bar would
read 0% for the whole run, then jump to 100%.

To surface *live* progress we write a small JSON blob to Redis at each
stage transition. Redis writes are outside the DB transaction, so the
polling endpoint sees them immediately (no lock contention, no waiting on
the ingest transaction). The key carries a TTL so a crashed/abandoned run
self-cleans; on a clean finish the pipeline clears it explicitly and the
endpoint falls back to the authoritative DB row (which by then reads
``ready`` / 100%).

Design notes
------------
* Best-effort: every helper swallows Redis errors. Progress UX must never
  break or slow down an ingest — if Redis is down we simply fall back to
  the DB-backed status (the pre-existing behaviour).
* Keyed by ``material_version_id`` (the pipeline's unit of work), but the
  read endpoint resolves ``material_id -> current_version_id`` first, so
  the API layer looks it up by version too.
* The stored ``percent`` mirrors the pipeline's existing
  ``job.progress_percent`` waypoints (10/30/60/80/95/100) so the DB and
  Redis never disagree on the number, only on *when* it becomes visible.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from redis.exceptions import RedisError

from abridgeai.core.cache.client import RedisFallbackError, get_cache

logger = logging.getLogger(__name__)

# Live progress is a transient UX signal, not a source of truth. A 1-hour
# TTL comfortably outlives the worker's job_timeout so the key never
# expires mid-run, yet a crashed run's stale key disappears on its own.
_PROGRESS_TTL_SECONDS = 3600


def _progress_key(material_version_id: UUID) -> str:
    return f"material:ingest:progress:{material_version_id}"


async def publish_progress(
    material_version_id: UUID,
    *,
    status: str,
    percent: int,
    stage_label: str | None = None,
    detail: str | None = None,
) -> None:
    """Write a live-progress snapshot to Redis (best-effort, never raises).

    Called by the pipeline at each stage transition. Errors are logged at
    DEBUG and swallowed — a Redis hiccup must never fail or stall an
    ingest, it just means the UI falls back to the DB-backed status.

    ``detail`` is an optional human sub-progress string (e.g. ``"42/85"``)
    for long stages that loop internally — the knowledge-graph build makes
    one LLM call per chunk, so without a running count it would sit frozen
    at a single percent for minutes. Surfaced to the UI so the teacher can
    see it's alive.
    """
    payload = json.dumps(
        {
            "status": status,
            "percent": max(0, min(100, int(percent))),
            "stage_label": stage_label,
            "detail": detail,
        }
    )
    try:
        client = get_cache()
        await client.set(
            _progress_key(material_version_id),
            payload,
            ex=_PROGRESS_TTL_SECONDS,
        )
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.debug(
            "materials.progress.publish_failed",
            extra={"material_version_id": str(material_version_id), "err": repr(exc)},
        )


async def read_progress(material_version_id: UUID) -> dict[str, Any] | None:
    """Read the live-progress snapshot for a version (``None`` if absent).

    Returns ``None`` when there is no live key (run finished/never started
    or Redis unavailable) so the caller can fall back to the DB row.
    """
    try:
        client = get_cache()
        raw = await client.get(_progress_key(material_version_id))
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.debug(
            "materials.progress.read_failed",
            extra={"material_version_id": str(material_version_id), "err": repr(exc)},
        )
        return None
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def clear_progress(material_version_id: UUID) -> None:
    """Delete the live-progress key (best-effort, never raises).

    Called on clean completion so the read endpoint falls back to the
    authoritative DB row (``ready`` / 100%) instead of a stale Redis blob.
    """
    try:
        client = get_cache()
        await client.delete(_progress_key(material_version_id))
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.debug(
            "materials.progress.clear_failed",
            extra={"material_version_id": str(material_version_id), "err": repr(exc)},
        )


__all__ = ["clear_progress", "publish_progress", "read_progress"]
