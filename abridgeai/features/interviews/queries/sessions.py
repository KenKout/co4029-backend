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
    limit: int | None = None,
) -> list[InterviewSession]:
    """Recent sessions for a user, newest first.

    ``status`` filters to one of the five session states
    (``in_progress`` / ``completed`` / ``timed_out`` / ``abandoned``
    / ``failed``); ``None`` returns every state. ``interview_config_id``
    scopes the list to one config's sessions (the live-interview page needs
    exactly those and nothing else); ``None`` returns every config.
    ``limit`` is an explicit cap (``None`` = no cap): the student history
    page must show ALL attempts, not a silently truncated window.
    """
    stmt = select(InterviewSession).where(InterviewSession.student_id == user_id)
    if status is not None:
        stmt = stmt.where(InterviewSession.status == status)
    if interview_config_id is not None:
        stmt = stmt.where(
            InterviewSession.interview_config_id == interview_config_id
        )
    stmt = stmt.order_by(InterviewSession.started_at.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
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


async def list_in_progress_sessions_with_time_limit(
    db: AsyncSession,
) -> list[tuple[InterviewSession, int]]:
    """In-progress sessions paired with their configured time-limit.

    Used by the deadline sweep. Untimed interviews never enter this result:
    only an explicit ``time_limit_minutes`` may terminalize an active session.
    Returns ``(session, time_limit_minutes)`` so the caller decides expiry in
    Python without per-row SQL interval math.
    """
    stmt = (
        select(InterviewSession, InterviewConfig.time_limit_minutes)
        .join(InterviewConfig, InterviewConfig.id == InterviewSession.interview_config_id)
        .where(
            InterviewSession.status == "in_progress",
            InterviewConfig.time_limit_minutes.is_not(None),
        )
    )
    rows = (await db.execute(stmt)).all()
    return [(row[0], row[1]) for row in rows]


async def list_pending_evaluation_sessions(
    db: AsyncSession,
    *,
    ended_before: datetime,
    max_recovery_attempts: int = 3,
) -> list[InterviewSession]:
    """Terminal sessions whose async evaluation appears to be stranded.

    The grace cutoff keeps an actively running evaluation job out of the
    recovery scan.

    ``status='failed'`` IS included. That status only means "ARQ exhausted its
    retry budget", which is an infrastructure outcome, not a judgement about the
    student: the session still holds real answers and ``_ungradeable_reason``
    still permits grading it. Excluding it stranded 14 sessions that a student
    had actually sat, with no re-drive path anywhere in the codebase. It is
    bounded by ``max_recovery_attempts`` so a genuinely unprocessable session
    cannot be retried forever.

    ``abandoned`` stays excluded on the merits: that status is only reached when
    the session has no gradeable answer at all.
    """
    recovery_attempts = InterviewSession.internal_summary_json[
        "evaluation_recovery"
    ]["attempts"].as_integer()
    stmt = select(InterviewSession).where(
        InterviewSession.status.in_(("completed", "timed_out", "failed")),
        InterviewSession.pass_verdict.is_(None),
        InterviewSession.ended_at.is_not(None),
        InterviewSession.ended_at <= ended_before,
        # NULL (never recovered) or under the ceiling. func.coalesce keeps the
        # comparison SQL-side so a missing key is not silently false.
        func.coalesce(recovery_attempts, 0) < max_recovery_attempts,
    )
    return list((await db.execute(stmt)).scalars().all())


_RECOVERY_ATTEMPTS_PATH = "'{evaluation_recovery,attempts}'"
# A legacy / hand-edited row could hold a non-numeric there. Mirror the Python
# reader's tolerance (``services.evaluation_state.recovery_attempts``) instead of
# letting one malformed row raise out of the sweep.
_RECOVERY_ATTEMPTS_INT = (
    f"CASE WHEN jsonb_typeof(internal_summary_json #> {_RECOVERY_ATTEMPTS_PATH}) = 'number' "
    f"     THEN (internal_summary_json #>> {_RECOVERY_ATTEMPTS_PATH})::int "
    "     ELSE 0 END"
)


async def stamp_evaluation_recovery_attempt(
    db: AsyncSession,
    session_id: UUID,
    *,
    now: datetime,
) -> int | None:
    """Charge one recovery attempt to this session. Returns the NEW attempt count.

    ``None`` means nothing was charged because the session is no longer a
    recovery candidate — a verdict was published between the candidate query and
    this call, so the caller must not enqueue anything.

    Why this is SQL and not a Python read-modify-write on the ORM row: the sweep
    loads its candidates in one transaction and then works through them, so its
    ``internal_summary_json`` snapshot goes stale the moment the evaluator it is
    trying to repair publishes. Assigning a whole dict back would blind-overwrite
    the column and wipe the ``total_score`` / ``rubric_aggregated`` /
    ``evaluated_at`` the evaluator just wrote (and resurrect the stale
    ``evaluation_failure`` note it had cleared). ``jsonb_set`` touches ONLY the
    ``evaluation_recovery`` key, so bookkeeping and results never collide.
    """
    from sqlalchemy import text  # noqa: PLC0415

    # The only interpolation is _RECOVERY_ATTEMPTS_INT, a module-level SQL
    # constant. Every value (session id, timestamp) is a bound parameter.
    sql = text(
        "UPDATE interview_sessions "  # noqa: S608 -- only _RECOVERY_ATTEMPTS_INT (a SQL constant) is interpolated
        "   SET internal_summary_json = jsonb_set("
        "         COALESCE(internal_summary_json, '{}'::jsonb), "
        "         '{evaluation_recovery}', "
        "         COALESCE(internal_summary_json -> 'evaluation_recovery', '{}'::jsonb) "
        "           || jsonb_build_object("
        f"                'attempts', {_RECOVERY_ATTEMPTS_INT} + 1, "
        # CAST is required: inside jsonb_build_object Postgres has no context to
        # infer a bare parameter's type and rejects the statement outright.
        "                'last_attempt_at', CAST(:now AS text)), "
        "         true) "
        " WHERE id = :session_id "
        "   AND pass_verdict IS NULL "
        f"RETURNING {_RECOVERY_ATTEMPTS_INT} AS attempts"
    )
    attempts = (
        await db.execute(sql, {"session_id": session_id, "now": now.isoformat()})
    ).scalar_one_or_none()
    await db.commit()
    return int(attempts) if attempts is not None else None


async def refund_evaluation_recovery_attempt(
    db: AsyncSession,
    session_id: UUID,
    *,
    attempt: int,
    previous_recovery: dict[str, Any] | None,
) -> bool:
    """Give back the attempt charged by ``stamp_evaluation_recovery_attempt``.

    Used when the dispatch produced no job at all (Redis down, or ARQ refused the
    ID), where charging the budget would strand the session unevaluated.

    Restores ONLY the ``evaluation_recovery`` sub-object, and only while the count
    we wrote is still the current one. Both bounds matter:

    * the sub-object scope means a verdict published while the enqueue was in
      flight survives the refund — writing back a whole pre-enqueue snapshot of
      ``internal_summary_json`` deleted the results the evaluator had just
      committed;
    * the count guard means a later sweep that has already charged its own
      attempt is not silently un-charged. We then leave our attempt spent, which
      is the safe direction (bounded retries, not an infinite loop).
    """
    import json  # noqa: PLC0415

    from sqlalchemy import text  # noqa: PLC0415

    # Same shape as stamp_evaluation_recovery_attempt: the interpolation is a
    # module-level SQL constant, not input.
    sql = text(
        "UPDATE interview_sessions "  # noqa: S608 -- SQL constant interpolation only, see stamp helper
        "   SET internal_summary_json = CASE "
        "         WHEN CAST(:previous_recovery AS jsonb) IS NULL "
        "           THEN internal_summary_json - 'evaluation_recovery' "
        "         ELSE jsonb_set(internal_summary_json, '{evaluation_recovery}', "
        "                        CAST(:previous_recovery AS jsonb), true) "
        "       END "
        " WHERE id = :session_id "
        f"   AND {_RECOVERY_ATTEMPTS_INT} = :attempt "
        "RETURNING id"
    )
    refunded = (
        await db.execute(
            sql,
            {
                "session_id": session_id,
                "attempt": attempt,
                "previous_recovery": (
                    json.dumps(previous_recovery) if previous_recovery is not None else None
                ),
            },
        )
    ).scalar_one_or_none() is not None
    await db.commit()
    return refunded


async def claim_session_evaluation(
    db: AsyncSession,
    session_id: UUID,
    *,
    token: UUID,
    now: datetime,
    lease_expires_at: datetime,
) -> bool:
    """Atomically take exclusive ownership of one session's evaluation.

    Returns ``True`` only when THIS caller won the claim. One conditional UPDATE
    does the whole decision, so two workers racing on the same session cannot
    both succeed: the loser's WHERE clause no longer matches after the winner's
    row lock is released.

    Refuses when:

    * a verdict is already published (``pass_verdict IS NOT NULL``) — including
      ``FALSE``, which is a real judgement, not an absence of one;
    * an unexpired claim is held by someone else.

    An EXPIRED claim is reclaimable: ARQ kills a task at ``job_timeout``, which
    is shorter than the lease, so a lapsed lease means the owner is dead rather
    than slow.
    """
    from sqlalchemy import or_, update  # noqa: PLC0415

    result = await db.execute(
        update(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.pass_verdict.is_(None),
            or_(
                InterviewSession.evaluation_claim_token.is_(None),
                InterviewSession.evaluation_claim_expires_at.is_(None),
                InterviewSession.evaluation_claim_expires_at <= now,
            ),
        )
        .values(
            evaluation_claim_token=token,
            evaluation_claim_expires_at=lease_expires_at,
        )
        .returning(InterviewSession.id)
    )
    claimed = result.scalar_one_or_none() is not None
    await db.commit()
    return claimed


async def holds_session_evaluation_claim(
    db: AsyncSession,
    session_id: UUID,
    *,
    token: UUID,
    now: datetime,
) -> bool:
    """Whether ``token`` still owns this session's evaluation right now.

    Checked before any publish or failure write. A stale owner (lease lapsed
    while it ran, so another job may already have taken over) must not write
    either outcome — that is exactly how a superseded job used to relabel a
    graded interview as failed.

    Reads with ``FOR UPDATE`` so the check and the caller's subsequent write are
    serialized against a concurrent reclaim attempt.
    """
    stmt = (
        select(InterviewSession.evaluation_claim_token)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.evaluation_claim_token == token,
            InterviewSession.evaluation_claim_expires_at.is_not(None),
            InterviewSession.evaluation_claim_expires_at > now,
        )
        .with_for_update()
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None


async def release_session_evaluation_claim(
    db: AsyncSession,
    session_id: UUID,
    *,
    token: UUID,
) -> None:
    """Drop our own claim so a later recovery can re-drive the session.

    Scoped to the token: a job whose lease already lapsed must not clear the
    claim of whoever legitimately took over. Releasing on the failure path is
    what lets the next sweep retry immediately instead of waiting out the lease.
    """
    from sqlalchemy import update  # noqa: PLC0415

    await db.execute(
        update(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.evaluation_claim_token == token,
        )
        .values(evaluation_claim_token=None, evaluation_claim_expires_at=None)
    )
    await db.commit()


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


async def terminalize_in_progress_session(
    db: AsyncSession,
    session_id: UUID,
    *,
    status: str,
    ended_at: datetime,
) -> bool:
    """Atomically move a live session to ``status``; True only if WE moved it.

    The status predicate lives in the UPDATE's own WHERE clause, so a second
    caller that read ``in_progress`` before the first committed still matches zero
    rows. Doing the check in Python instead (read, compare, write) let two
    finishers through: the student's own ``completed`` finish could be relabelled
    ``timed_out`` by the agent's hard-stop timer, and ``ended_at`` — which anchors
    the recovery sweep's grace window — moved with it.

    Mirrors ``finalize_expired_in_progress_session``, which the deadline sweep
    already used; this is the same guarantee for the submit path.
    """
    from sqlalchemy import update  # noqa: PLC0415

    result = await db.execute(
        update(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.status == "in_progress",
        )
        .values(status=status, ended_at=ended_at)
        .returning(InterviewSession.id)
    )
    return result.scalar_one_or_none() is not None


async def finalize_expired_in_progress_session(
    db: AsyncSession,
    session_id: UUID,
    *,
    ended_at: datetime,
) -> str | None:
    """Atomically terminalize an expired live session without overwriting a finish.

    Returns the terminal status only when this caller changed an ``in_progress``
    row. The status is derived inside the conditional update from the latest
    committed gradeable transcript, so a concurrent natural submit wins and an
    answer committed immediately before the sweep is not misclassified as
    ``abandoned``.
    """
    from sqlalchemy import case, exists, update  # noqa: PLC0415

    has_user_turn = exists().where(
        InterviewSessionMessage.session_id == InterviewSession.id,
        InterviewSessionMessage.role == "user",
        InterviewSessionMessage.session_question_id.is_not(None),
    )
    result = await db.execute(
        update(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.status == "in_progress",
        )
        .values(
            status=case((has_user_turn, "timed_out"), else_="abandoned"),
            ended_at=ended_at,
        )
        .returning(InterviewSession.status)
    )
    await db.commit()
    return result.scalar_one_or_none()


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
    "finalize_expired_in_progress_session",
    "get_active_session",
    "get_last_terminal_ended_at",
    "get_gap_report_for_session",
    "get_outcome_evaluations",
    "get_session",
    "get_session_attempt_number",
    "get_session_with_responses",
    "get_user_interview_sessions",
    "list_in_progress_sessions_with_time_limit",
    "list_integrity_events_for_session",
    "list_pending_evaluation_sessions",
    "list_session_messages",
    "list_sessions_for_config",
    "refund_evaluation_recovery_attempt",
    "stamp_evaluation_recovery_attempt",
]
