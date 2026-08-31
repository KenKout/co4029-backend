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

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_EVALUATE_INTERVIEW_SESSION_TASK = "evaluate_interview_session_task"


def _evaluation_job_id(session_id: UUID) -> str:
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
) -> int:
    """Re-enqueue terminal sessions left without an evaluation verdict.

    This repairs rows stranded by worker crashes or by the former retry bug
    where ARQ exhausted its budget before the application could stamp
    ``status='failed'``. A deterministic job ID prevents duplicate work when
    consecutive sweeps overlap an evaluation already in progress.
    """
    if arq_pool is None:
        return 0
    candidates = await sessions_queries.list_pending_evaluation_sessions(
        db,
        ended_before=utcnow() - timedelta(minutes=max(1, grace_minutes)),
    )
    enqueued = 0
    for session in candidates:
        job = await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _EVALUATE_INTERVIEW_SESSION_TASK,
            session.student_id,
            session.id,
            _job_id=_evaluation_job_id(session.id),
        )
        if job is not None:
            enqueued += 1
    return enqueued


__all__ = [
    "recover_stalled_evaluations",
    "sweep_expired_interview_sessions",
]
