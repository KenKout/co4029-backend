"""Cron task — reconcile orphaned ingest ``ProcessingJob`` rows.

The "pending forever, no signal" failure mode
----------------------------------------------
A material ingest is driven by a Postgres ``ProcessingJob`` row *and* a
matching ARQ job in Redis. The two are committed separately: the DB row
is written in the request transaction, the ARQ job is enqueued into
Redis. They can drift apart:

* A worker restart (or crash) mid-flight can drop an in-progress / queued
  ARQ job while the committed ``ProcessingJob(pending|running)`` row
  survives. Nothing is left in Redis to ever consume it.
* A transient Redis blip at enqueue time (pre-``enqueue_material_ingest``
  fix) could commit the DB row without ever landing the ARQ job.

Either way the version sits at ``pending`` with a spinning badge and no
error surfaced anywhere — the exact symptom that stranded Chapter_3 for
40 minutes with no signal.

What this reaper does
---------------------
Every few minutes it finds ``ProcessingJob`` rows for material versions
that are stuck ``pending`` / ``running`` but have **no live ARQ job**
backing them, and recovers each one:

* **Re-enqueue** (default) up to ``_MAX_REQUEUE_ATTEMPTS`` times, via the
  resilient :func:`enqueue_material_ingest` helper, so a dropped job is
  automatically picked back up within one cron interval.
* **Fail** it (``status='failed'`` + a clear ``error_message`` +
  ``version.processing_status='failed'``) once it has exhausted its
  re-enqueue budget, so it stops masquerading as in-flight and the
  teacher sees an actionable error instead of an eternal spinner.

Liveness signal
---------------
A "live" ARQ job leaves an ``arq:job:<id>`` key in Redis for its whole
lifecycle — queued, in-progress, and retry-scheduled — and that key is
deleted once the job completes. So the reaper reads every ``arq:job:*``
payload, extracts the ``material_version_id`` argument, and treats any
version present in that set as *backed by a live job* (leave it alone —
this correctly protects jobs legitimately waiting in a busy queue). A
stuck DB row whose version is **absent** from that set is a true orphan.

Grace period
------------
Only rows older than ``_ORPHAN_GRACE_SECONDS`` are considered, so a job
mid-enqueue (DB row committed, ARQ write in flight) in the same instant
the reaper runs is never falsely reaped.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.ai.models import ProcessingJob
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.materials.ingestion.progress import publish_progress
from abridgeai.features.materials.models import LearningMaterialVersion
from abridgeai.features.materials.workers.enqueue import enqueue_material_ingest

_logger = get_logger(__name__)

# Don't touch a row younger than this — avoids racing a job that's being
# enqueued right now (DB row committed, ARQ write a few ms behind).
_ORPHAN_GRACE_SECONDS = 120

# How many times the reaper will re-enqueue one job before giving up and
# marking it failed. Uses the job's own ``retry_count`` column as the
# durable counter so the budget survives across cron runs and worker
# restarts (unlike ARQ's in-Redis retry count, which is what we lost).
_MAX_REQUEUE_ATTEMPTS = 3

# Only material_version ingest jobs are self-recoverable here (we know how
# to re-enqueue them). Other job types are left for their own owners.
_INGEST_ENTITY_TYPE = "material_version"
_STUCK_STATUSES = ("pending", "running")


async def reconcile_orphaned_ingests_task(ctx: dict[str, Any]) -> None:
    """Re-enqueue or fail ingest jobs stuck with no live ARQ job.

    Registered in :mod:`abridgeai.workers.arq_app`'s ``cron_jobs`` to run
    every few minutes. Fire-and-forget: counts + outcomes are emitted via
    structured logs, nothing is returned. The ARQ pool is read from
    ``ctx['redis']`` (the worker always supplies it) for both liveness
    detection and re-enqueue, falling back to the resilient helper's
    process-local pool when absent.
    """
    bind_request_context(task="reconcile_orphaned_ingests")
    try:
        await _run_reconcile(arq_pool=ctx.get("redis"))
    finally:
        clear_request_context()


async def _live_version_ids(redis: Any) -> set[UUID]:
    """Version IDs that currently have a live ARQ ingest job in Redis.

    Reads every ``arq:job:*`` payload and pulls the second positional arg
    (``material_version_id`` — see ``ingest_material_version_task``'s
    signature ``(ctx, actor_id, material_version_id, pipeline_run_id)``,
    which ARQ stores as ``args`` without ``ctx``). Deserialisation uses
    ARQ's own pickler so we read exactly what the worker will.
    """
    from arq.jobs import deserialize_job_raw  # noqa: PLC0415

    live: set[UUID] = set()
    if redis is None:
        return live
    try:
        keys = await redis.keys("arq:job:*")
        for key in keys:
            raw = await redis.get(key)
            if raw is None:
                continue
            try:
                _fn, args, _kwargs, *_ = deserialize_job_raw(raw)
            except Exception:  # noqa: BLE001 -- a malformed job shouldn't break the sweep
                continue
            # args == (actor_id, material_version_id, pipeline_run_id)
            if len(args) >= 2 and isinstance(args[1], UUID):
                live.add(args[1])
    except Exception:  # noqa: BLE001 -- Redis hiccup: treat as "no info", reap nothing
        _logger.warning("reaper_live_scan_failed", exc_info=True)
        # Return a sentinel-free empty set is dangerous (would reap everything),
        # so re-raise to abort this run instead — a failed scan must not cause
        # mass re-enqueue.
        raise
    return live


async def _run_reconcile(*, arq_pool: Any) -> None:
    settings = get_settings()
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=_ORPHAN_GRACE_SECONDS)

    # Build the live-job set first. If this raises (Redis down), we abort the
    # whole run rather than risk reaping jobs that are actually alive.
    try:
        live_version_ids = await _live_version_ids(arq_pool)
    except Exception:  # noqa: BLE001
        _logger.warning("reaper_aborted_live_scan_unavailable")
        return

    requeued = 0
    failed = 0
    inspected = 0

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        stuck = (
            await db.execute(
                select(ProcessingJob).where(
                    ProcessingJob.entity_type == _INGEST_ENTITY_TYPE,
                    ProcessingJob.status.in_(_STUCK_STATUSES),
                    ProcessingJob.created_at < cutoff,
                )
            )
        ).scalars().all()

        for job in stuck:
            version_id = job.entity_id
            if version_id in live_version_ids:
                # A live ARQ job is backing this row (queued/in-progress/retry).
                # Legitimately waiting — leave it alone.
                continue

            inspected += 1
            version = await db.get(LearningMaterialVersion, version_id)
            if version is None:
                # Version deleted out from under the job row — nothing to
                # recover. Mark the orphan failed so it stops being counted.
                job.status = "failed"
                job.error_message = "orphaned: material version no longer exists"
                job.finished_at = datetime.now(tz=UTC)
                failed += 1
                continue

            if job.retry_count >= _MAX_REQUEUE_ATTEMPTS:
                # Exhausted the re-enqueue budget — fail loudly with an
                # actionable message instead of spinning forever.
                job.status = "failed"
                job.error_message = (
                    f"ingest could not be recovered after {job.retry_count} "
                    "automatic re-enqueue attempts; please reprocess manually"
                )
                job.finished_at = datetime.now(tz=UTC)
                version.processing_status = "failed"
                version.processing_error = job.error_message
                failed += 1
                await publish_progress(
                    version_id,
                    status="failed",
                    percent=job.progress_percent,
                    stage_label="reaper",
                    detail="auto-recovery exhausted",
                )
                _logger.warning(
                    "reaper_job_failed_exhausted",
                    processing_job_id=str(job.id),
                    material_version_id=str(version_id),
                    retry_count=job.retry_count,
                )
                continue

            # Re-enqueue: bump the durable counter, reset to a clean pending
            # state, and push a fresh ARQ job. The pipeline_run_id is a new
            # audit-grouping key for this recovery attempt.
            from uuid import uuid4  # noqa: PLC0415

            job.retry_count += 1
            job.status = "pending"
            job.started_at = None
            job.error_message = None
            version.processing_status = "pending"
            version.processing_error = None
            pipeline_run_id = uuid4()

            # Commit the DB state BEFORE enqueuing so the worker (which may
            # pick the job up almost immediately) never races a not-yet-
            # committed row.
            await db.commit()

            try:
                await enqueue_material_ingest(
                    arq_pool,
                    actor_id=version.uploaded_by,
                    material_version_id=version_id,
                    pipeline_run_id=pipeline_run_id,
                )
            except Exception:  # noqa: BLE001 -- log + continue; next run retries
                _logger.exception(
                    "reaper_requeue_enqueue_failed",
                    processing_job_id=str(job.id),
                    material_version_id=str(version_id),
                )
                continue

            await publish_progress(
                version_id,
                status="pending",
                percent=2,
                stage_label="queued",
                detail="auto-recovered, waiting for worker",
            )
            requeued += 1
            _logger.info(
                "reaper_job_requeued",
                processing_job_id=str(job.id),
                material_version_id=str(version_id),
                attempt=job.retry_count,
                pipeline_run_id=str(pipeline_run_id),
            )

        # Persist the fail-path mutations (re-enqueue path already committed
        # per row above).
        await db.commit()

    del settings  # reserved for future per-env tuning
    _logger.info(
        "reaper_run_completed",
        orphans_inspected=inspected,
        requeued=requeued,
        failed=failed,
        live_jobs=len(live_version_ids),
    )


__all__ = ["reconcile_orphaned_ingests_task"]
