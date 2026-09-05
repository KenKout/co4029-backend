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
from datetime import timedelta
from typing import TYPE_CHECKING
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


def _evaluation_job_id(session_id: UUID, *, attempt: int = 0) -> str:
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

        if terminal_status == "timed_out" and arq_pool is not None:
            await arq_pool.enqueue_job(  # type: ignore[attr-defined]
                _EVALUATE_INTERVIEW_SESSION_TASK,
                session.student_id,
                session.id,
                _job_id=_evaluation_job_id(session.id),
            )
        logger.info(
            "swept expired interview session %s → %s",
            session.id,
            terminal_status,
        )

    return finalised


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
        previous_summary = dict(session.internal_summary_json or {})
        summary = dict(previous_summary)
        recovery = dict(summary.get("evaluation_recovery") or {})
        attempt = int(recovery.get("attempts") or 0) + 1
        recovery["attempts"] = attempt
        recovery["last_attempt_at"] = utcnow().isoformat()
        summary["evaluation_recovery"] = recovery
        session.internal_summary_json = summary
        await db.commit()

        job: object | None = None
        dispatch_error: Exception | None = None
        try:
            job = await arq_pool.enqueue_job(  # type: ignore[attr-defined]
                _EVALUATE_INTERVIEW_SESSION_TASK,
                session.student_id,
                session.id,
                _job_id=_evaluation_job_id(session.id, attempt=attempt),
            )
        except Exception as exc:  # noqa: BLE001 -- transport failure, refunded below
            dispatch_error = exc

        if job is not None:
            enqueued += 1
            continue

        # Nothing was queued. Refund the attempt so a dispatch outage cannot
        # consume the student's grading opportunities. Metadata only — never
        # the transcript.
        session.internal_summary_json = previous_summary
        await db.commit()
        logger.warning(
            "interview evaluation recovery dispatch failed; attempt refunded",
            extra={
                "session_id": str(session.id),
                "attempt": attempt,
                "job_id": _evaluation_job_id(session.id, attempt=attempt),
                "error": str(dispatch_error) if dispatch_error is not None else "enqueue_refused",
            },
        )
    return enqueued


__all__ = [
    "recover_stalled_evaluations",
    "sweep_expired_interview_sessions",
]
