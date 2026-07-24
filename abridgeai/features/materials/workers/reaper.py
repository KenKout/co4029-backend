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

from sqlalchemy import select, text

from abridgeai.ai.models import GenerationRun, ProcessingJob
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.materials.ingestion.progress import publish_progress
from abridgeai.features.materials.models import LearningMaterialVersion
from abridgeai.features.materials.services.completion_notify import (
    notify_material_processing_outcome,
)
from abridgeai.features.materials.workers.enqueue import enqueue_material_ingest
from abridgeai.features.quizzes.services.completion_notify import (
    notify_quiz_generation_outcome,
)

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
        redis = ctx.get("redis")
        await _run_reconcile(arq_pool=redis)
        # Quiz generation runs are driven by the same "DB row + ARQ job"
        # pairing and suffer the identical orphan failure mode (a worker
        # restart mid-run drops the ARQ job while the generation_runs row
        # stays status='running' forever — the "stuck at 25%" symptom). Run
        # the quiz reconciler in the same tick so both recover automatically.
        await _run_reconcile_quiz(arq_pool=redis)
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
    # material_version_ids the reaper gives up on this tick; the initiating
    # teacher is notified AFTER the terminal commit (best-effort, decoupled).
    versions_to_notify: list[UUID] = []

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        stuck = (
            (
                await db.execute(
                    select(ProcessingJob).where(
                        ProcessingJob.entity_type == _INGEST_ENTITY_TYPE,
                        ProcessingJob.status.in_(_STUCK_STATUSES),
                        ProcessingJob.created_at < cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )

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
                if version_id is not None:
                    versions_to_notify.append(version_id)
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

    # Notify each teacher whose ingest the reaper gave up on (best-effort; a
    # fresh session so it's decoupled from the reaper's terminal commit above).
    if versions_to_notify:
        async with sessionmaker() as notify_db:
            for version_id in versions_to_notify:
                # Recipient is the version's uploader.
                version = await notify_db.get(LearningMaterialVersion, version_id)
                if version is None:
                    continue
                await notify_material_processing_outcome(
                    notify_db,
                    recipient_user_id=version.uploaded_by,
                    material_version_id=version_id,
                    succeeded=False,
                    error_message=(
                        "Automatic recovery gave up; please reprocess the material."
                    ),
                    arq_pool=arq_pool,
                )
            await notify_db.commit()

    del settings  # reserved for future per-env tuning
    _logger.info(
        "reaper_run_completed",
        orphans_inspected=inspected,
        requeued=requeued,
        failed=failed,
        live_jobs=len(live_version_ids),
    )


# --- Quiz generation-run reconciliation -------------------------------------

# generation_runs has no retry_count column, so the durable re-enqueue budget
# is tracked inside config_json under this key.
_QUIZ_REAP_KEY = "reap_count"
_QUIZ_MAX_REQUEUE_ATTEMPTS = 2
# A quiz run legitimately runs for minutes (several sequential LLM calls). Only
# consider a run orphaned once it's older than this AND absent from the live
# ARQ set — long enough that a healthy in-flight run is never reaped.
_QUIZ_ORPHAN_GRACE_SECONDS = 300
_QUIZ_STUCK_STATUSES = ("pending", "running")


def _coerce_uuid(value: object) -> UUID | None:
    """Best-effort coerce a config_json value into a UUID (or None)."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


async def _live_quiz_run_ids(redis: Any) -> set[UUID]:
    """Run IDs that currently have a live ARQ ``run_quiz_generation_task``.

    ``run_quiz_generation_task``'s stored args are ``(actor_id,
    generation_run_id)`` (ctx is not stored), so the run id is ``args[1]``.
    Same liveness contract as ingests: an ``arq:job:*`` key exists for a
    job's whole lifecycle and is deleted on completion. On any Redis error
    we re-raise so the caller aborts rather than reaping live runs.
    """
    from arq.jobs import deserialize_job_raw  # noqa: PLC0415

    live: set[UUID] = set()
    if redis is None:
        return live
    keys = await redis.keys("arq:job:*")
    for key in keys:
        raw = await redis.get(key)
        if raw is None:
            continue
        try:
            fn, args, _kwargs, *_ = deserialize_job_raw(raw)
        except Exception:  # noqa: BLE001 -- a malformed job shouldn't break the sweep
            continue
        if fn != "run_quiz_generation_task":
            continue
        # args == (actor_id, generation_run_id). ARQ may serialize the run id
        # as a UUID or as its string form depending on how it was enqueued;
        # coerce both so a live run is NEVER misread as an orphan (a false
        # "orphan" would get a duplicate re-enqueue, doubling load on the LLM
        # endpoint and racing the still-running original).
        if len(args) >= 2:
            try:
                live.add(args[1] if isinstance(args[1], UUID) else UUID(str(args[1])))
            except (TypeError, ValueError):
                continue
    return live


async def _run_reconcile_quiz(*, arq_pool: Any) -> None:
    """Re-enqueue or fail quiz ``generation_runs`` stuck with no live ARQ job.

    Mirrors the ingest reaper. A quiz run only writes its terminal state
    (questions + ``status='completed'``) in a single commit at the very end,
    so an orphaned run has persisted nothing and is safe to re-enqueue: the
    re-run starts cleanly from retrieval.
    """
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=_QUIZ_ORPHAN_GRACE_SECONDS)

    try:
        live_run_ids = await _live_quiz_run_ids(arq_pool)
    except Exception:  # noqa: BLE001 -- Redis down: abort rather than reap live runs
        _logger.warning("reaper_quiz_aborted_live_scan_unavailable", exc_info=True)
        return

    requeued = 0
    failed = 0
    inspected = 0
    # (recipient_user_id, course_id, quiz_id) for each run the reaper gives up
    # on this tick. Notifications are dispatched AFTER the terminal commit so a
    # notify failure can't roll back the fail-state write.
    to_notify: list[tuple[UUID | None, UUID | None, UUID | None]] = []

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        # A still-zombie run row may be lock-held by an orphaned transaction
        # that hasn't been cleaned up yet. Use a short lock_timeout so the
        # reaper never blocks on it — a locked row is skipped this tick and
        # retried next tick (by which point the idle-in-transaction backstop
        # will have released it).
        await db.execute(text("SET LOCAL lock_timeout = '3s'"))
        stuck = (
            (
                await db.execute(
                    select(GenerationRun).where(
                        GenerationRun.generation_type == "quiz",
                        GenerationRun.status.in_(_QUIZ_STUCK_STATUSES),
                        GenerationRun.updated_at < cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )

        for run in stuck:
            if run.id in live_run_ids:
                continue  # backed by a live ARQ job — leave it alone

            # Staleness is measured from the last activity timestamp, not
            # created_at: a re-enqueued run's row can be hours old while the
            # current attempt only just started. started_at (set when the
            # dispatcher marks the run 'running') is the truthful "work began"
            # signal; fall back to updated_at / created_at when it's unset
            # (still 'pending'). A running row that started within the grace
            # window is left alone even if absent from the live set (covers the
            # tiny gap between enqueue and the ARQ job key appearing).
            started = run.started_at or run.updated_at or run.created_at
            if started is not None and started >= cutoff:
                continue

            inspected += 1
            config = dict(run.config_json or {})
            reap_count = int(config.get(_QUIZ_REAP_KEY) or 0)

            if reap_count >= _QUIZ_MAX_REQUEUE_ATTEMPTS:
                run.status = "failed"
                run.finished_at = datetime.now(tz=UTC)
                run.config_json = config | {
                    "failure": {
                        "message": (
                            f"quiz generation could not be recovered after "
                            f"{reap_count} automatic re-enqueue attempts; "
                            "please retry generation"
                        )
                    }
                }
                failed += 1
                # quiz_id (from config_json) + course_id (on the run) feed the
                # notification deep-link. Queue the notify for after the commit.
                to_notify.append(
                    (run.requested_by, run.course_id, _coerce_uuid(config.get("quiz_id")))
                )
                _logger.warning(
                    "reaper_quiz_run_failed_exhausted",
                    generation_run_id=str(run.id),
                    reap_count=reap_count,
                )
                continue

            # Re-enqueue: bump the durable counter, reset to a clean pending
            # state, commit BEFORE enqueuing so the worker never races a
            # not-yet-committed row.
            run.status = "pending"
            run.started_at = None
            run.config_json = config | {_QUIZ_REAP_KEY: reap_count + 1}
            actor_id = run.requested_by
            await db.commit()

            if arq_pool is None:
                continue
            try:
                await arq_pool.enqueue_job(  # type: ignore[attr-defined]
                    "run_quiz_generation_task", actor_id, run.id
                )
            except Exception:  # noqa: BLE001 -- log + continue; next run retries
                _logger.exception(
                    "reaper_quiz_requeue_enqueue_failed",
                    generation_run_id=str(run.id),
                )
                continue

            requeued += 1
            _logger.info(
                "reaper_quiz_run_requeued",
                generation_run_id=str(run.id),
                attempt=reap_count + 1,
            )

        await db.commit()

    # Notify each teacher whose run the reaper gave up on (best-effort; a fresh
    # session so it's fully decoupled from the reaper's terminal commit above).
    if to_notify:
        async with sessionmaker() as notify_db:
            for recipient_id, course_id, quiz_id in to_notify:
                await notify_quiz_generation_outcome(
                    notify_db,
                    recipient_user_id=recipient_id,
                    quiz_id=quiz_id,
                    course_id=course_id,
                    quiz_title=None,
                    succeeded=False,
                    error_message="Automatic recovery gave up; please retry generation.",
                    arq_pool=arq_pool,
                )
            await notify_db.commit()

    _logger.info(
        "reaper_quiz_run_completed",
        orphans_inspected=inspected,
        requeued=requeued,
        failed=failed,
        live_runs=len(live_run_ids),
    )


__all__ = ["reconcile_orphaned_ingests_task"]
