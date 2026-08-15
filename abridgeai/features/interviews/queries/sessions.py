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

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import (
    GapReport,
    InterviewConfig,
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
    interview_config_id: UUID | None = None,
    limit: int = 20,
) -> list[InterviewSession]:
    """Recent sessions for a user, newest first.

    ``status`` filters to one of the five session states
    (``in_progress`` / ``completed`` / ``timed_out`` / ``abandoned``
    / ``failed``); ``None`` returns every state. ``interview_config_id``
    scopes the list to one config's sessions (the live-interview page needs
    exactly those and nothing else); ``None`` returns every config.
    ``limit`` caps the result count for the user's history page.
    """
    stmt = select(InterviewSession).where(InterviewSession.student_id == user_id)
    if status is not None:
        stmt = stmt.where(InterviewSession.status == status)
    if interview_config_id is not None:
        stmt = stmt.where(
            InterviewSession.interview_config_id == interview_config_id
        )
    stmt = stmt.order_by(InterviewSession.started_at.desc()).limit(limit)
    return list((await db.execute(stmt)).scalars().all())


async def list_sessions_for_config(db: AsyncSession, config_id: UUID) -> list[InterviewSession]:
    """All sessions (any student) for one interview config, newest first.

    Powers the teacher's per-config attempts list (thesis p77 review surface).
    Authorisation is enforced by the router's config-scoped access guard.
    """
    stmt = (
        select(InterviewSession)
        .where(InterviewSession.interview_config_id == config_id)
        .order_by(InterviewSession.started_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_sessions_for_course(db: AsyncSession, course_id: UUID) -> list[Any]:
    """Every interview session (any student, any config) in a course, newest first.

    Powers the teacher's course-wide "Assessments" tab. Returns
    SQLAlchemy ``Row`` objects with ``.InterviewSession`` and ``.title``
    (the interview config title) so the router avoids a second round-trip.
    """
    stmt = (
        select(
            InterviewSession,
            InterviewConfig.title,
            InterviewConfig.security_incident_summary_enabled,
        )
        .join(InterviewConfig, InterviewConfig.id == InterviewSession.interview_config_id)
        .where(InterviewConfig.course_id == course_id)
        .order_by(InterviewSession.started_at.desc())
    )
    return list((await db.execute(stmt)).all())


async def list_sessions_for_student_in_course(
    db: AsyncSession, course_id: UUID, student_id: UUID
) -> list[Any]:
    """Every interview session by one student across a course's configs, newest first.

    Powers the teacher's per-student profile page. Same row shape as
    :func:`list_sessions_for_course`.
    """
    stmt = (
        select(
            InterviewSession,
            InterviewConfig.title,
            InterviewConfig.security_incident_summary_enabled,
        )
        .join(InterviewConfig, InterviewConfig.id == InterviewSession.interview_config_id)
        .where(InterviewConfig.course_id == course_id, InterviewSession.student_id == student_id)
        .order_by(InterviewSession.started_at.desc())
    )
    return list((await db.execute(stmt)).all())


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


async def count_terminal_sessions(
    db: AsyncSession,
    user_id: UUID,
    config_id: UUID,
    terminal_statuses: tuple[str, ...],
) -> int:
    """Count the student's finished (non-live) attempts for a config.

    Powers the FR-5.3 ``max_attempts`` gate at ``start_session``. Only
    terminal statuses count — an ``in_progress`` row is handled by the
    idempotent active-session short-circuit, not the attempt ceiling.
    """
    stmt = select(func.count(InterviewSession.id)).where(
        InterviewSession.interview_config_id == config_id,
        InterviewSession.student_id == user_id,
        InterviewSession.status.in_(terminal_statuses),
    )
    return int((await db.execute(stmt)).scalar_one())


async def get_last_terminal_ended_at(
    db: AsyncSession,
    user_id: UUID,
    config_id: UUID,
    terminal_statuses: tuple[str, ...],
) -> datetime | None:
    """Most recent ``ended_at`` across the student's finished attempts.

    Anchors the FR-5.3 retake cooldown. Falls back to ``started_at`` when
    a terminal row somehow has a NULL ``ended_at`` (defensive: timed_out /
    abandoned rows should always stamp ``ended_at``, but a crash mid-write
    could leave it null and we still want the cooldown to engage).
    """
    stmt = select(
        func.max(func.coalesce(InterviewSession.ended_at, InterviewSession.started_at))
    ).where(
        InterviewSession.interview_config_id == config_id,
        InterviewSession.student_id == user_id,
        InterviewSession.status.in_(terminal_statuses),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


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


async def list_in_progress_voice_sessions_with_limit(
    db: AsyncSession,
) -> list[tuple[InterviewSession, int | None]]:
    """In-progress sessions paired with their config time-limit.

    Used by the stale-session sweep (Phase 4). Returns ``(session,
    time_limit_minutes)`` so the caller decides staleness in Python (avoids
    per-row SQL interval math). ``time_limit_minutes`` may be ``None`` (no
    limit configured → session is never swept on time).
    """
    stmt = (
        select(InterviewSession, InterviewConfig.time_limit_minutes)
        .join(InterviewConfig, InterviewConfig.id == InterviewSession.interview_config_id)
        .where(
            InterviewSession.status == "in_progress",
        )
    )
    rows = (await db.execute(stmt)).all()
    return [(row[0], row[1]) for row in rows]


async def list_pending_evaluation_sessions(
    db: AsyncSession,
    *,
    ended_before: datetime,
) -> list[InterviewSession]:
    """Terminal sessions whose async evaluation appears to be stranded.

    The grace cutoff keeps an actively running evaluation job out of the
    recovery scan. ``failed`` and ``abandoned`` rows are intentionally
    excluded because they are already terminal from the grader's perspective.
    """
    stmt = select(InterviewSession).where(
        InterviewSession.status.in_(("completed", "timed_out")),
        InterviewSession.pass_verdict.is_(None),
        InterviewSession.ended_at.is_not(None),
        InterviewSession.ended_at <= ended_before,
    )
    return list((await db.execute(stmt)).scalars().all())


async def count_user_messages(db: AsyncSession, session_id: UUID) -> int:
    """Number of graded student answer turns recorded for a session.

    The sweep uses this to decide whether a stale session has enough
    transcript to evaluate (>=1) or should simply be marked ``abandoned``.
    """
    stmt = select(func.count(InterviewSessionMessage.id)).where(
        InterviewSessionMessage.session_id == session_id,
        InterviewSessionMessage.role == "user",
        InterviewSessionMessage.session_question_id.is_not(None),
    )
    return int((await db.execute(stmt)).scalar_one())


async def get_last_activity_at(db: AsyncSession, session_id: UUID) -> datetime | None:
    """Timestamp of the most recent message in a session, or ``None`` if the
    session has no messages yet.

    The stale-session sweep uses this as the "idle since" anchor for voice
    sessions whose config has no ``time_limit_minutes``: a session is finalised
    once it has been silent (no new message) for the fixed idle window. When
    there are no messages, callers fall back to ``started_at``.
    """
    stmt = select(func.max(InterviewSessionMessage.created_at)).where(
        InterviewSessionMessage.session_id == session_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_integrity_events_for_session(db: AsyncSession, session_id: UUID) -> list[Any]:
    """Return FR-5.8 proctoring events for one interview session, oldest first.

    Projection source for the teacher gap-report integrity timeline. Scoped to
    ``assessment_kind='interview'``. Empty list when none were recorded (the
    common case for an honest take). Mirrors the quiz-side
    ``list_integrity_events_for_attempt``.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        AssessmentIntegrityEvent,
    )

    stmt = (
        select(AssessmentIntegrityEvent)
        .where(
            AssessmentIntegrityEvent.assessment_kind == "interview",
            AssessmentIntegrityEvent.interview_session_id == session_id,
        )
        .order_by(AssessmentIntegrityEvent.created_at.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "count_terminal_sessions",
    "count_user_messages",
    "get_last_activity_at",
    "get_last_terminal_ended_at",
    "get_active_session",
    "get_gap_report_for_session",
    "get_outcome_evaluations",
    "get_session",
    "get_session_attempt_number",
    "get_session_with_responses",
    "get_user_interview_sessions",
    "list_in_progress_voice_sessions_with_limit",
    "list_integrity_events_for_session",
    "list_pending_evaluation_sessions",
    "list_session_messages",
    "list_sessions_for_config",
]
