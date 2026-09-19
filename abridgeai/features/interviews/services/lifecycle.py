"""Interview session lifecycle hardening for active interview attempts.

A periodic ARQ sweep finalises only sessions whose config has an explicit
``time_limit_minutes`` and whose assessment deadline has elapsed. Untimed
sessions remain ``in_progress`` and resumable; a hidden idle timeout must not
end a learner's assessment.

For an expired timed session:

* >=1 student turn recorded → ``timed_out`` + enqueue the async judge (so the
  student still gets something back from a partial interview).
* no student turns → ``abandoned`` (nothing to evaluate).

Disconnect itself is NOT terminal: the agent keeps the session ``in_progress``
so the student can re-mint a token and rejoin within the attempt. Only an
explicit configured time-limit closes a stale session.

ORM-free: reads go through ``queries.sessions``; writes mutate the returned
ORM objects + commit. Keeps the "services do not import sqlalchemy" contract
satisfied without a new ignore entry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.security import utcnow
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services.evaluation_state import (
    MAX_EVALUATION_RECOVERY_ATTEMPTS,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_EVALUATE_INTERVIEW_SESSION_TASK = "evaluate_interview_session_task"


def evaluation_job_id(session_id: UUID, *, attempt: int = 0) -> str:
    """Deterministic ARQ job ID for one session's evaluation.

    ``attempt=0`` (the default) keeps the original session-scoped ID used by the
    natural submit and sweep paths, where deduplication is exactly what we want:
    two enqueues for the same finish must not grade twice.

    Recovery passes its attempt number so each re-drive gets a distinct ID. ARQ
    refuses a duplicate ID while the previous result is still in Redis
    (``keep_result_seconds = 3600``), so a session-scoped ID would make every
    recovery within the hour a silent no-op.
    """
    if attempt > 0:
        return f"interview-evaluation:{session_id}:recover-{attempt}"
    return f"interview-evaluation:{session_id}"


def _evaluation_job_id(session_id: UUID, *, attempt: int = 0) -> str:
    """Backward-compatible alias for the deterministic job-id helper."""
    return evaluation_job_id(session_id, attempt=attempt)


async def sweep_expired_interview_sessions(
    db: AsyncSession,
    arq_pool: object | None = None,
) -> int:
    """Finalise sessions past an explicitly configured assessment deadline.

    Untimed sessions are excluded by the query and must remain resumable. A
    timed session cannot expire before assessment starts, so onboarding rows
    with no ``assessment_started_at`` are also left untouched.
    """
    now = utcnow()
    candidates = await sessions_queries.list_in_progress_sessions_with_time_limit(db)
    finalised = 0

    for session, time_limit_minutes in candidates:
        if session.assessment_started_at is None:
            continue
        deadline = session.assessment_started_at + timedelta(minutes=time_limit_minutes)
        if now < deadline:
            continue

        terminal_status = await sessions_queries.finalize_expired_in_progress_session(
            db,
            session.id,
            ended_at=now,
        )
        if terminal_status is None:
            continue
        finalised += 1

        # Terminal path #3 (recording): the sweep closed this session, so its
        # recording must stop too. Best-effort and COMMITTED SEPARATELY — the
        # finalisation above already committed inside the query helper, and a
        # swallowed stop failure must never roll it back (see the two-writer
        # commit-ordering lesson: a best-effort side effect never shares the
        # transaction it reacts to).
        try:
            from abridgeai.features.interviews.services import (  # noqa: PLC0415
                recording as recording_service,
            )

            await recording_service.stop_recording_for_session(db, session_id=session.id)
            await db.commit()
        except Exception:  # noqa: BLE001 -- advisory side effect, never fatal
            await db.rollback()
            logger.warning("recording stop failed during sweep (session=%s)", session.id)

        if terminal_status == "timed_out" and arq_pool is not None:
            await arq_pool.enqueue_job(  # type: ignore[attr-defined]
                _EVALUATE_INTERVIEW_SESSION_TASK,
                session.student_id,
                session.id,
                _job_id=evaluation_job_id(session.id),
            )
        logger.info(
            "swept expired interview session %s → %s",
            session.id,
            terminal_status,
        )

    return finalised


async def record_missing_dispatch(
    db: AsyncSession,
    *,
    session_id: UUID,
    job_id: str,
) -> None:
    """Durable ``phase='missing'`` marker for a dispatch that never reached Redis.

    Called from the terminal paths when ``enqueue_job`` raises (Redis down,
    transport error) AFTER the terminal commit: the session stays terminal,
    the caller returns an honest completion response instead of a 500, and
    ``recover_stalled_evaluations`` redrives the row immediately — a missing
    job needs no grace window because there is nothing to wait for. Owns its
    commit: the terminal state it annotates is already durable, and a failure
    here must never mask the completion response (the sweep remains the
    safety net either way).

    Mirrors the recovery record shape so ``_ACTIVE_PHASES_SQL`` does not count
    ``missing`` as active (it is terminal) while the redrive CAS still finds it.
    """
    from sqlalchemy import text  # noqa: PLC0415

    now = utcnow().isoformat()
    sql = text(
        "UPDATE interview_sessions "
        "   SET internal_summary_json = jsonb_set("
        "         COALESCE(internal_summary_json, '{}'::jsonb), "
        "         '{evaluation_recovery}', "
        "         COALESCE(internal_summary_json -> 'evaluation_recovery', '{}'::jsonb) "
        "           || jsonb_build_object("
        "                'current', jsonb_build_object("
        "                  'job_id', CAST(:job_id AS text), "
        "                  'phase', 'missing', "
        "                  'dispatched_at', CAST(:now AS text))), "
        "         true) "
        " WHERE id = :session_id "
        "RETURNING id"
    )
    try:
        await db.execute(
            sql, {"session_id": session_id, "job_id": job_id, "now": now}
        )
        await db.commit()
    except Exception:  # noqa: BLE001 -- bookkeeping only; the sweep is the safety net
        await db.rollback()
        logger.exception(
            "recording missing dispatch failed (session=%s)", session_id
        )


async def enforce_assessment_deadline(
    db: AsyncSession,
    session_id: UUID,
    assessment_started_at: datetime,
    *,
    arq_pool: object | None,
) -> None:
    """Refuse work on a timed session whose deadline has passed (audit P1).

    Between the deadline and the next sweep the session still reads
    ``in_progress``; this terminalizes it with the SAME conditional helper the
    sweep uses (a concurrent natural submit wins; the closing reason and the
    evaluation dispatch stay identical) and raises. Called before any turn
    write in ``take_session_step``.
    """
    from abridgeai.core.exceptions import AppError  # noqa: PLC0415

    time_limit_minutes = await sessions_queries.get_session_time_limit_minutes(
        db, session_id
    )
    if time_limit_minutes is None:
        return
    if utcnow() < assessment_started_at + timedelta(minutes=time_limit_minutes):
        return
    terminal_status = await sessions_queries.finalize_expired_in_progress_session(
        db,
        session_id,
        ended_at=utcnow(),
    )
    await db.commit()
    if terminal_status == "timed_out" and arq_pool is not None:
        session = await sessions_queries.get_session(db, session_id)
        if session is not None:
            await dispatch_evaluation_or_record_missing(
                db,
                session_id=session_id,
                student_id=session.student_id,
                arq_pool=arq_pool,
            )
    raise AppError("The interview time limit has elapsed")


async def dispatch_evaluation_or_record_missing(
    db: AsyncSession,
    *,
    session_id: UUID,
    student_id: UUID,
    arq_pool: object,
) -> bool:
    """Enqueue evaluation; on failure stamp a durable ``missing`` record.

    The terminal commit lands BEFORE this call, so an enqueue exception must
    not propagate: the caller's retry would only see the terminal row and
    never re-enqueue. Instead the failure stamps ``phase='missing'`` (own
    commit — the annotated terminal state is already durable) and the sweep
    redrives it immediately. True when a job actually reached Redis.
    """
    job_id = evaluation_job_id(session_id)
    try:
        job = await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _EVALUATE_INTERVIEW_SESSION_TASK,
            student_id,
            session_id,
            _job_id=job_id,
        )
    except Exception:  # noqa: BLE001 -- dispatch failure must not 500 a committed terminal state
        logger.exception(
            "evaluation dispatch failed after terminal commit (session=%s, job_id=%s)",
            session_id,
            job_id,
        )
        await record_missing_dispatch(db, session_id=session_id, job_id=job_id)
        return False
    return job is not None


async def redrive_missing_dispatch_on_refinish(
    db: AsyncSession,
    *,
    session_id: UUID,
    student_id: UUID,
    arq_pool: object,
) -> None:
    """Re-attempt a dispatch the first finish lost, on a terminal re-finish.

    Fires ONLY when the durable record says ``phase='missing'``: a winner whose
    dispatch succeeded leaves no recovery record, so an ordinary re-finish
    race loser does not enqueue a second time (the finish-race contract). On a
    successful re-dispatch the record CASes ``missing`` → ``queued`` scoped to
    this job id, so the sweep does not redrive it again.
    """
    from sqlalchemy import text  # noqa: PLC0415

    from abridgeai.features.interviews.queries import (  # noqa: PLC0415
        sessions as sessions_queries,
    )

    phase = (
        await db.execute(
            text(
                "SELECT internal_summary_json #>> "
                "'{evaluation_recovery,current,phase}' "
                "FROM interview_sessions WHERE id = :id"
            ),
            {"id": session_id},
        )
    ).scalar_one_or_none()
    if phase != "missing":
        return
    if await dispatch_evaluation_or_record_missing(
        db, session_id=session_id, student_id=student_id, arq_pool=arq_pool
    ):
        job_id = evaluation_job_id(session_id)
        await sessions_queries.transition_evaluation_recovery_phase(
            db,
            session_id,
            job_id=job_id,
            from_phase="missing",
            to_phase="queued",
            extra={"redrived_at": utcnow().isoformat()},
        )
        await db.commit()


async def recover_stalled_evaluations(
    db: AsyncSession,
    arq_pool: object | None = None,
    *,
    grace_minutes: int = 15,
    max_recovery_attempts: int = MAX_EVALUATION_RECOVERY_ATTEMPTS,
) -> int:
    """Re-enqueue terminal sessions left without an evaluation verdict.

    Repairs rows stranded by worker crashes, and rows where ARQ exhausted its
    own retry budget and stamped ``status='failed'``. That status is an
    infrastructure outcome rather than a judgement about the student — the
    answers are still there and still gradeable — so leaving those rows alone
    permanently discarded work a student had actually done.

    Two bounds keep this from becoming an infinite retry loop:

    * ``max_recovery_attempts`` — counted in
      ``internal_summary_json['evaluation_recovery']['attempts']`` and enforced
      SQL-side by the query, so a session that cannot be processed is abandoned
      after a few sweeps instead of being re-queued every five minutes forever.
    * the attempt counter is stamped BEFORE the job is enqueued, so a task that
      dies hard (OOM, worker kill) still consumes its budget. Under-counting
      would be safe for the student but would reintroduce the loop.

    The stamp-before-enqueue order only holds when a job actually reached
    Redis. A *dispatch* failure creates no job at all, so charging it against
    the budget would strand the session: three sweeps during a Redis outage
    exhaust the ceiling, the SQL-side filter drops the row, and the answers are
    never graded even after Redis recovers. So the counter is rolled back when
    the enqueue raises OR returns ``None`` (ARQ refused the ID — nothing was
    queued). Everything after a successful handoff still costs an attempt.

    Both the charge and the refund are SQL-side, sub-key writes (see
    ``queries.sessions.stamp_evaluation_recovery_attempt`` /
    ``refund_evaluation_recovery_attempt``). They have to be: an evaluator can
    publish a verdict while this sweep is between its candidate query and its
    enqueue, and writing back a whole ``internal_summary_json`` dict from the
    stale ORM snapshot deleted the rubric totals / ``evaluated_at`` that verdict
    had just committed — and resurrected the ``evaluation_failure`` note it had
    cleared. Charging is also conditional on ``pass_verdict IS NULL``, so a
    session graded in that window is skipped instead of re-queued.

    Each candidate's enqueue is isolated: a transport error on one row used to
    propagate out of the loop and skip every candidate behind it in that sweep.

    The job ID is per-attempt rather than per-session: ARQ refuses a duplicate
    job ID while the previous result is still in Redis (``keep_result_seconds``
    is 3600), so reusing the session-scoped ID would make every recovery inside
    that hour a silent no-op — the enqueue returns ``None`` and nothing happens.
    """
    if arq_pool is None:
        return 0
    candidates = await sessions_queries.list_pending_evaluation_sessions(
        db,
        ended_before=utcnow() - timedelta(minutes=max(1, grace_minutes)),
        max_recovery_attempts=max_recovery_attempts,
    )
    enqueued = 0
    for session in candidates:
        # Reconcile the durable phase record against ARQ BEFORE deciding to
        # charge: an active record whose job ARQ no longer holds must be
        # marked terminal (or left active, when Redis cannot say), never
        # charged over.
        try:
            if await _reconcile_current_phase(db, session, arq_pool=arq_pool):
                continue
        except Exception:  # noqa: BLE001 -- one bad row must not skip the sweep
            logger.exception(
                "reconciling recovery phase failed (session=%s)",
                getattr(session, "id", "?"),
            )
            continue
        if await _redrive_one_evaluation(db, session, arq_pool=arq_pool):
            enqueued += 1
    return enqueued


async def _arq_job_is_live(arq_pool: object, job_id: str) -> bool | None:
    """Whether ARQ still holds ``job_id``: True live, False gone, None unknown.

    A live ARQ job keeps an ``arq:job:<id>`` key in Redis for its whole
    lifecycle (queued, in-progress, retry-scheduled) and the key is deleted
    once the job finishes. A result key existing means the job RAN — terminal
    either way. Any Redis failure returns None: "unknown" must never be read
    as "dead", because that is exactly how a queued job was declared exhausted
    while it waited for a worker.
    """
    # arq's ArqRedis subclasses redis.asyncio.Redis, so the pool itself is the
    # client. Typed as object at this boundary; the casts below are honest.
    redis: Any = getattr(arq_pool, "_redis", None) or arq_pool
    try:
        job_key = await redis.exists(f"arq:job:{job_id}")
        if job_key:
            return True
        result_key = await redis.exists(f"arq:result:{job_id}")
        if result_key:
            return False
        return False
    except Exception:  # noqa: BLE001 -- Redis down: the answer is UNKNOWN
        logger.warning("arq job-state lookup failed; treating state as unknown")
        return None


async def _reconcile_current_phase(
    db: AsyncSession,
    session: object,
    *,
    arq_pool: object,
) -> bool:
    """Bring the durable phase record in line with what ARQ knows.

    An ACTIVE record whose job ARQ no longer holds is marked ``missing``
    (terminal) so the sweep may charge a fresh attempt; a record whose result
    key exists is marked ``succeeded``/``failed`` is left to the verdict —
    either way it stops blocking. A live job, or an unknown Redis state,
    changes nothing.

    Returns True when the record now counts as active.
    """

    summary = getattr(session, "internal_summary_json", None) or {}
    recovery = summary.get("evaluation_recovery") if isinstance(summary, dict) else None
    current = recovery.get("current") if isinstance(recovery, dict) else None
    if not isinstance(current, dict):
        return False
    phase = current.get("phase")
    if phase not in ("dispatching", "queued", "running", "retrying"):
        return False

    job_id = current.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        # A record that never named its job cannot be verified; ARQ may still
        # hold it. Leave it active — the conservative direction.
        return True

    state = await _arq_job_is_live(arq_pool, job_id)
    if state is None:
        # Redis unavailable: unknown is not dead. Skip without charging.
        return True

    from abridgeai.features.interviews.queries import sessions as _q  # noqa: PLC0415

    if state:
        return True  # still queued/running/retrying per ARQ
    # ARQ holds neither the job nor a result it could be resumed from: the
    # record's job is gone. Mark it missing (terminal) — one narrow, CASed
    # write scoped to the exact job id and phase we observed.
    await _q.transition_evaluation_recovery_phase(
        db,
        session.id,  # type: ignore[attr-defined]
        job_id=job_id,
        from_phase=str(phase),
        to_phase="missing",
        extra={"settled_at": utcnow().isoformat()},
    )
    return False


async def _redrive_one_evaluation(
    db: AsyncSession,
    session: object,
    *,
    arq_pool: object,
) -> bool:
    """Charge an attempt and enqueue one session's evaluation. True if queued.

    Isolated per candidate so a transport error cannot skip the rest of the sweep,
    and refunds the attempt whenever no job actually reached Redis.
    """
    session_id = session.id  # type: ignore[attr-defined]
    # Audit P1 (hard-stop drain timeout): a candidate that qualified via a
    # STALE verdict (a receipt applied after ``evaluated_at``) must have that
    # verdict retired before enqueueing — the claim refuses any session with a
    # verdict, so the re-driven job would no-op and the late answer would stay
    # ungraded forever. The invalidation re-checks the same predicate
    # atomically; a verdict retired by nobody else stays retired only when the
    # staleness still holds.
    if getattr(session, "pass_verdict", None) is not None:
        await sessions_queries.invalidate_stale_verdict(db, session_id)
    # A live claim / active durable job re-checked at stamp time (the candidate
    # list may be stale) refuses the charge — including a claim that appeared
    # after the query ran.
    job_id = evaluation_job_id(session_id, attempt=1)
    # The attempt number is only known after the stamp (it returns the NEW
    # count), but the stamp builds the deterministic job id itself, so pass the
    # caller-chosen id shape: we must know the id BEFORE dispatch to CAS the
    # phase. Stamp with an explicit job id derived from the returned count is
    # two writes; instead stamp reserves with its own default id, so read it
    # back from the returned attempt count.
    # Snapshot the recovery sub-object ONLY — never the whole summary. See the
    # refund helper: restoring a whole snapshot destroyed concurrent results.
    previous_recovery = (getattr(session, "internal_summary_json", None) or {}).get(
        "evaluation_recovery"
    )
    previous_recovery = dict(previous_recovery) if isinstance(previous_recovery, dict) else None

    attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
        db, session_id, now=utcnow()
    )
    if attempt is not None:
        logger.info(
            "interview.evaluation.dispatch_reserved",
            extra={"session_id": str(session_id), "attempt": attempt},
        )
    if attempt is None:
        # A verdict landed, a live claim appeared, or the previous recovery's
        # job is still active — none of it is this sweep's attempt to spend.
        logger.info(
            "interview.evaluation.recovery_skipped_live_claim",
            extra={"session_id": str(session_id)},
        )
        return False

    job_id = evaluation_job_id(session_id, attempt=attempt)
    job: object | None = None
    dispatch_error: Exception | None = None
    try:
        job = await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _EVALUATE_INTERVIEW_SESSION_TASK,
            session.student_id,  # type: ignore[attr-defined]
            session_id,
            _job_id=job_id,
        )
    except Exception as exc:  # noqa: BLE001 -- transport failure, refunded below
        dispatch_error = exc

    if job is not None:
        # ARQ accepted the job: reserve → queued, scoped to THIS dispatch.
        await sessions_queries.transition_evaluation_recovery_phase(
            db,
            session_id,
            job_id=job_id,
            from_phase="dispatching",
            to_phase="queued",
            extra={"queued_at": utcnow().isoformat()},
        )
        logger.info(
            "interview.evaluation.dispatch_queued",
            extra={"session_id": str(session_id), "attempt": attempt, "job_id": job_id[:12]},
        )
        return True

    # Nothing was queued. Refund the attempt (and drop the dispatching record)
    # so a dispatch outage cannot consume the student's grading opportunities.
    # Metadata only — never the transcript, never anything the evaluator owns.
    refunded = await sessions_queries.refund_evaluation_recovery_attempt(
        db,
        session_id,
        attempt=attempt,
        previous_recovery=previous_recovery,
    )
    logger.warning(
        "interview.evaluation.dispatch_refunded",
        extra={
            "session_id": str(session_id),
            "attempt": attempt,
            "refunded": refunded,
            "job_id": job_id,
            "error": str(dispatch_error) if dispatch_error is not None else "enqueue_refused",
        },
    )
    return False


__all__ = [
    "dispatch_evaluation_or_record_missing",
    "enforce_assessment_deadline",
    "evaluation_job_id",
    "record_missing_dispatch",
    "redrive_missing_dispatch_on_refinish",
    "recover_stalled_evaluations",
    "sweep_expired_interview_sessions",
]
