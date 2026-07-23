"""Resilient enqueue for the materials ingest task.

Bug-fix (material-ingestion "pending forever"): the request-path enqueue
used to be guarded by a bare ``if arq_pool is not None:``. When the
injected pool was ``None`` — e.g. the app lifespan hadn't finished wiring
the ``dependency_overrides`` pool, or a transient Redis blip at startup —
the enqueue was silently skipped. The ``ProcessingJob`` row committed as
``pending`` and nothing ever consumed it, so the document sat in
"pending" forever with no error surfaced anywhere.

This helper removes the silent-skip failure mode:

* When the caller passes a live pool (the normal DI path), it's used.
* When the caller passes ``None``, a process-local ``ArqRedis`` pool is
  lazily created from ``settings.redis_url`` (mirrors the interviews
  ``realtime.orchestration_bridge`` fallback) and reused thereafter.
* If enqueue still fails, the error is logged LOUDLY and re-raised so the
  caller's request fails visibly instead of leaving a silent orphan job.

The task name is the canonical ARQ function name registered on the worker
(``ingest_material_version_task`` — see ``workers/arq_app.py``).
"""

from __future__ import annotations

from uuid import UUID

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from abridgeai.core.config import get_settings
from abridgeai.core.observability import get_logger

_logger = get_logger(__name__)

_INGEST_TASK_NAME = "ingest_material_version_task"

# Process-local fallback pool, created on first use when DI didn't supply one.
_fallback_pool: ArqRedis | None = None


async def _get_fallback_pool() -> ArqRedis:
    """Lazily create + reuse a process-local ArqRedis pool from redis_url."""
    global _fallback_pool
    if _fallback_pool is None:
        _fallback_pool = await create_pool(
            RedisSettings.from_dsn(get_settings().redis_url)
        )
    return _fallback_pool


async def enqueue_material_ingest(
    arq_pool: object | None,
    *,
    actor_id: UUID,
    material_version_id: UUID,
    pipeline_run_id: UUID,
) -> None:
    """Enqueue ``ingest_material_version_task``, never silently skipping.

    Uses the injected ``arq_pool`` when present; otherwise falls back to a
    process-local pool built from ``redis_url``. Any enqueue failure is
    logged and re-raised so the request surfaces the error instead of
    committing an orphaned ``pending`` job row.
    """
    pool = arq_pool if arq_pool is not None else await _get_fallback_pool()
    if arq_pool is None:
        _logger.warning(
            "materials_ingest_enqueue_pool_fallback",
            material_version_id=str(material_version_id),
            pipeline_run_id=str(pipeline_run_id),
            reason="injected arq_pool was None; using process-local fallback pool",
        )
    try:
        await pool.enqueue_job(  # type: ignore[attr-defined]
            _INGEST_TASK_NAME,
            actor_id,
            material_version_id,
            pipeline_run_id,
        )
    except Exception:
        _logger.exception(
            "materials_ingest_enqueue_failed",
            material_version_id=str(material_version_id),
            pipeline_run_id=str(pipeline_run_id),
        )
        raise


__all__ = ["enqueue_material_ingest"]
