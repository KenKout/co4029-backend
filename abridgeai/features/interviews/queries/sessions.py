"""Interview session lifecycle queries.

Plan §6.3. Session-side queries cover both runtime read paths
(active session lookup, message history, evaluation list) and
post-completion artefacts (gap reports, session summaries).

Sessions are HISTORICAL RECORD — they have no soft-delete column
(plan §6.1 explicit). Queries DO NOT filter ``deleted_at`` on the
session aggregate; the T0.7 listener is a no-op for these tables.

One-active-session-per-config invariant
---------------------------------------
The DB-level UNIQUE constraint
``uq_interview_sessions_number(interview_config_id, student_id,
attempt_number)`` enforces that no two sessions share the same
attempt number for a (config, student) pair. The application
ensures concurrent active sessions don't leak by:

1. Allocating ``attempt_number = MAX(attempt_number) + 1`` via
   :func:`get_session_attempt_number` (this query) before insert.
2. Querying :func:`get_active_session` to detect any in-progress /
   pending row before allocating a new attempt — service layer
   returns the existing session instead of starting a new one.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import (
    GapReport,
    InterviewOutcomeEvaluation,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)


async def get_session(db: AsyncSession, session_id: UUID) -> InterviewSession | None:
    """Single session row by id, or ``None``."""
    stmt = select(InterviewSession).where(InterviewSession.id == session_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_session_with_responses(
    db: AsyncSession, session_id: UUID
) -> (
    tuple[
        InterviewSession,
        list[InterviewSessionQuestion],
        list[InterviewSessionMessage],
    ]
    | None
):
    """Session plus the asked-questions audit and the message log.

    Returns ``(session, asked_questions, messages)`` ordered by
    ``sequence_no`` (questions) and ``created_at`` (messages).
    Returns ``None`` if the session does not exist.
    """
    session = await get_session(db, session_id)
    if session is None:
        return None
    questions_stmt = (
        select(InterviewSessionQuestion)
        .where(InterviewSessionQuestion.session_id == session_id)
        .order_by(InterviewSessionQuestion.sequence_no)
    )
    questions = list((await db.execute(questions_stmt)).scalars().all())
    messages = await list_session_messages(db, session_id)
    return session, questions, messages


async def get_user_interview_sessions(
    db: AsyncSession,
    user_id: UUID,
    *,
    status: str | None = None,
    limit: int = 20,
) -> list[InterviewSession]:
    """Recent sessions for a user, newest first.

    ``status`` filters to one of the five session states
    (``in_progress`` / ``completed`` / ``timed_out`` / ``abandoned``
    / ``failed``); ``None`` returns every state. ``limit`` caps the
    result count for the user's history page.
    """
    stmt = select(InterviewSession).where(InterviewSession.student_id == user_id)
    if status is not None:
        stmt = stmt.where(InterviewSession.status == status)
    stmt = stmt.order_by(InterviewSession.started_at.desc()).limit(limit)
    return list((await db.execute(stmt)).scalars().all())


async def get_active_session(
    db: AsyncSession,
    user_id: UUID,
    config_id: UUID,
) -> InterviewSession | None:
    """The single in-progress session for ``(user_id, config_id)``, if any.

    The session model does not have a ``pending`` state — the five
    legal statuses are ``in_progress`` / ``completed`` / ``timed_out``
    / ``abandoned`` / ``failed`` (baseline DDL line 835). Only
    ``in_progress`` is "live" — the others are terminal.

    Returns the most-recently-started ``in_progress`` row, or
    ``None``. Service layer uses this to enforce
    "no concurrent active sessions" by returning the existing live
    session instead of allocating a new attempt number.
    """
    stmt = (
        select(InterviewSession)
        .where(
            InterviewSession.interview_config_id == config_id,
            InterviewSession.student_id == user_id,
            InterviewSession.status == "in_progress",
        )
        .order_by(InterviewSession.started_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_session_attempt_number(
    db: AsyncSession,
    user_id: UUID,
    config_id: UUID,
) -> int:
    """``MAX(attempt_number) + 1`` for the next session insert.

    Returns 1 when the user has no prior sessions for this config.
    Used by ``services/lifecycle.py::start_session`` to allocate the
    next attempt number that satisfies
    ``uq_interview_sessions_number``.
    """
    stmt = select(func.coalesce(func.max(InterviewSession.attempt_number), 0)).where(
        InterviewSession.interview_config_id == config_id,
        InterviewSession.student_id == user_id,
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def get_outcome_evaluations(
    db: AsyncSession, session_id: UUID
) -> list[InterviewOutcomeEvaluation]:
    """Per-criterion evaluations for a session.

    The ``hidden_reasoning`` field is loaded (it lives in the ORM
    model) but the public schema layer (T6.2) MUST strip it before
    serialisation. Routers using these rows directly are responsible
    for the redaction; the public schema enforces it.
    """
    stmt = select(InterviewOutcomeEvaluation).where(
        InterviewOutcomeEvaluation.session_id == session_id
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_gap_report_for_session(db: AsyncSession, session_id: UUID) -> GapReport | None:
    """Latest gap report anchored to this session, or ``None``.

    A session may have zero or more gap reports (the
    ``get_or_create`` flow in legacy ``service.py:166-184``
    deduplicates by source ref). When multiple exist, the
    most-recently-created row wins.
    """
    stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_session_messages(
    db: AsyncSession,
    session_id: UUID,
    session_question_id: UUID | None = None,
) -> list[InterviewSessionMessage]:
    """Messages for a session, ordered by ``created_at`` ascending.

    Pass ``session_question_id`` to filter to messages tied to a
    specific asked-question (the per-question reveal pattern in
    plan §6.2). ``None`` returns the entire transcript — used by
    the teacher review surface and the gap-report builder.
    """
    stmt = select(InterviewSessionMessage).where(InterviewSessionMessage.session_id == session_id)
    if session_question_id is not None:
        stmt = stmt.where(InterviewSessionMessage.session_question_id == session_question_id)
    stmt = stmt.order_by(InterviewSessionMessage.created_at)
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "get_active_session",
    "get_gap_report_for_session",
    "get_outcome_evaluations",
    "get_session",
    "get_session_attempt_number",
    "get_session_with_responses",
    "get_user_interview_sessions",
    "list_session_messages",
]
