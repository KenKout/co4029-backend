"""Interview session lifecycle hardening for active interview attempts.

Voice sessions can be left ``in_progress`` if the student closes the tab or
loses connection. A periodic sweep (ARQ cron) finalises sessions that have
gone stale, using two anchors:

* config has ``time_limit_minutes`` → deadline is ``started_at + limit``.
* config has NO ``time_limit_minutes`` → fall back to a fixed idle window
  (``idle_timeout_minutes``, default 30) measured from the last message (or
  ``started_at`` when silent). This safety net guarantees an untimed session
  can never stay ``in_progress`` forever.

Once a session is past whichever deadline applies:

* >=1 student turn recorded → ``timed_out`` + enqueue the async judge (so the
  student still gets something back from a partial interview): the normal
  evaluation for a graded run, or the criterion-feedback task for a practice
  one, which writes no verdict.
* no student turns → ``abandoned`` (nothing to evaluate).

Disconnect itself is NOT terminal: the agent keeps the session ``in_progress``
so the student can re-mint a token and rejoin within the attempt. Only the
time-limit (via this sweep) closes a stale voice session.

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
_PRACTICE_FEEDBACK_TASK = "generate_practice_feedback_task"


def _evaluation_job_id(session_id: UUID) -> str:
    return f"interview-evaluation:{session_id}"


def _practice_feedback_job_id(session_id: UUID) -> str:
    return f"interview-practice-feedback:{session_id}"


async def sweep_stale_voice_sessions(
    db: AsyncSession,
    arq_pool: object | None = None,
    idle_timeout_minutes: int = 30,
) -> int:
    """Finalise in-progress sessions past their assessed or idle deadline.

    Returns the number of sessions finalised. The deadline is
    ``started_at + time_limit_minutes`` when the config sets a limit; otherwise
    it falls back to ``last_activity + idle_timeout_minutes`` (last message, or
    ``started_at`` when the session has been silent) so an untimed session can
    never stay ``in_progress`` forever.
    """
    now = utcnow()
    candidates = await sessions_queries.list_in_progress_voice_sessions_with_limit(db)
    finalised = 0

    for session, time_limit_minutes in candidates:
        if time_limit_minutes is not None and session.assessment_started_at is not None:
            deadline = session.assessment_started_at + timedelta(minutes=time_limit_minutes)
        else:
            last_activity = await sessions_queries.get_last_activity_at(db, session.id)
            anchor = last_activity or session.started_at
            deadline = anchor + timedelta(minutes=idle_timeout_minutes)
        if now < deadline:
            continue

        user_turns = await sessions_queries.count_user_messages(db, session.id)
        session.ended_at = now
        if user_turns >= 1:
            session.status = "timed_out"
        else:
            session.status = "abandoned"
        await db.commit()
        finalised += 1

        # A swept practice run is still a run the student answered questions in,
        # so it gets the same criterion feedback the normal finish path would
        # have given it — just never the grading task, which is the only writer
        # of ``pass_verdict``.
        if user_turns >= 1 and arq_pool is not None:
            is_practice = session.session_mode == "practice"
            await arq_pool.enqueue_job(  # type: ignore[attr-defined]
                _PRACTICE_FEEDBACK_TASK if is_practice else _EVALUATE_INTERVIEW_SESSION_TASK,
                session.student_id,
                session.id,
                _job_id=(
                    _practice_feedback_job_id(session.id)
                    if is_practice
                    else _evaluation_job_id(session.id)
                ),
            )
        logger.info(
            "swept stale voice session %s → %s (user_turns=%d)",
            session.id,
            session.status,
            user_turns,
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


async def mark_abandoned(db: AsyncSession, session_id: UUID) -> None:
    """Mark a single in-progress session ``abandoned`` (idempotent)."""
    session = await sessions_queries.get_session(db, session_id)
    if session is None or session.status != "in_progress":
        return
    session.status = "abandoned"
    session.ended_at = utcnow()
    await db.commit()


__all__ = [
    "mark_abandoned",
    "recover_stalled_evaluations",
    "sweep_stale_voice_sessions",
]
