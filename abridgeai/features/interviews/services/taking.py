"""Interview taking service (T6.11).

Student-side session lifecycle: ``start_session``, ``take_session_step``
(record answer + optional follow-up + advance to next question), and
``submit_session`` (mark completed + enqueue async evaluation).

Security invariants:

* ``start_session`` enforces "1 active session per (config, student)"
  by short-circuiting on ``queries.sessions.get_active_session`` —
  duplicate POSTs return the existing live session.
* ``take_session_step`` and ``submit_session`` raise
  :class:`ForbiddenError` when ``session.student_id != actor.user_id``
  (router maps to HTTP 403). Cross-user session leak = security bug.

Submit dispatches the post-session evaluation via ARQ task
``evaluate_interview_session_task`` — never run synchronously in the
HTTP path (latency + retry budget concerns).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.interviews.ai.stages.followup import maybe_generate_followup
from abridgeai.features.interviews.models import (
    InterviewQuestion,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_EVALUATE_INTERVIEW_SESSION_TASK = "evaluate_interview_session_task"


async def _require_session(db: AsyncSession, session_id: UUID) -> InterviewSession:
    session = await sessions_queries.get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")
    return session


def _assert_owns_session(session: InterviewSession, actor: CurrentUser) -> None:
    if session.student_id != actor.user_id:
        raise ForbiddenError("Session does not belong to the calling user")


async def _first_published_question(db: AsyncSession, config_id: UUID) -> InterviewQuestion | None:
    questions = await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )
    return questions[0] if questions else None


async def _next_published_question_after(
    db: AsyncSession,
    config_id: UUID,
    asked_question_ids: set[UUID],
) -> InterviewQuestion | None:
    questions = await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )
    for question in questions:
        if question.id not in asked_question_ids:
            return question
    return None


async def _next_session_sequence(db: AsyncSession, session_id: UUID) -> int:
    from sqlalchemy import func, select  # noqa: PLC0415

    row = await db.execute(
        select(func.coalesce(func.max(InterviewSessionQuestion.sequence_no), 0)).where(
            InterviewSessionQuestion.session_id == session_id
        )
    )
    return int(row.scalar_one()) + 1


async def start_session(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewSessionStartRequest at router edge.
    actor: CurrentUser,
) -> InterviewSession:
    """Create (or return the existing live) :class:`InterviewSession`.

    Idempotent: a second ``start`` call while a previous session is
    still ``in_progress`` returns the same row instead of allocating a
    new attempt number — enforces plan §6.3 "1 active per config".

    Self-heal: if the existing live session has zero attached questions
    (e.g. it was created before any question was approved), attach the
    first approved question now so the learner doesn't get stuck on an
    empty session forever.

    Input-mode upgrade: if the caller specifies an ``input_mode`` that
    differs from the existing session's recorded mode AND the parent
    config allows it, update the session in-place. Without this, a
    student whose first ``start`` request landed as ``text`` could never
    escalate to ``voice``/``hybrid`` — the realtime-token endpoint then
    rejects them with HTTP 409 "session is not a voice interview".
    """
    data = payload.model_dump(exclude_unset=True) if payload is not None else {}
    requested_input_mode = data.get("input_mode") or "text"

    # Thesis §4.3: the pass/fail verdict is judged per learning outcome. Starting
    # an interview whose config has no outcomes guarantees an automatic fail with
    # nothing to evaluate — block it so the student never enters a no-win session.
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
    if not outcomes:
        raise AppError(
            "interview_no_outcomes: this interview has no learning outcomes configured"
        )

    existing = await sessions_queries.get_active_session(db, actor.user_id, config_id)
    if existing is not None:
        await _ensure_first_question_attached(db, existing.id, config_id)
        await _maybe_upgrade_input_mode(db, existing, config_id, requested_input_mode)
        return existing

    attempt_number = await sessions_queries.get_session_attempt_number(db, actor.user_id, config_id)
    session = InterviewSession(
        interview_config_id=config_id,
        student_id=actor.user_id,
        attempt_number=attempt_number,
        status="in_progress",
        input_mode=requested_input_mode,
    )
    db.add(session)
    await flush_or_conflict(db)

    first_question = await _first_published_question(db, config_id)
    if first_question is not None:
        db.add(
            InterviewSessionQuestion(
                session_id=session.id,
                interview_question_id=first_question.id,
                sequence_no=1,
            )
        )
        await flush_or_conflict(db)

    await db.refresh(session)
    return session


_CONFIG_MODE_ALLOWS: dict[str, set[str]] = {
    "text": {"text"},
    "voice": {"voice"},
    "hybrid": {"text", "voice", "hybrid"},
}


async def _maybe_upgrade_input_mode(
    db: AsyncSession,
    session: InterviewSession,
    config_id: UUID,
    requested: str,
) -> None:
    """Promote ``session.input_mode`` to ``requested`` when permitted.

    No-op when the modes already match or when the parent config does
    not allow the requested mode. Silent on mismatch by design — the
    realtime-token endpoint is the right place to reject voice on a
    text-only config; here we just sync the session to what the caller
    asked for.
    """
    if session.input_mode == requested:
        return
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None:
        return
    allowed = _CONFIG_MODE_ALLOWS.get(config.supported_modes, set())
    if requested not in allowed:
        return
    session.input_mode = requested
    await flush_or_conflict(db)


async def _ensure_first_question_attached(
    db: AsyncSession, session_id: UUID, config_id: UUID
) -> None:
    """Attach question #1 to ``session_id`` if it has none yet.

    Closes the gap where ``start_session`` short-circuited on a stale
    ``in_progress`` row created before any question reached
    ``review_status='approved'``. Without this, approving questions
    after the fact never reaches the learner's existing session.
    """
    from sqlalchemy import func, select  # noqa: PLC0415

    count = (
        await db.execute(
            select(func.count(InterviewSessionQuestion.id)).where(
                InterviewSessionQuestion.session_id == session_id
            )
        )
    ).scalar_one()
    if int(count) > 0:
        return
    first_question = await _first_published_question(db, config_id)
    if first_question is None:
        return
    db.add(
        InterviewSessionQuestion(
            session_id=session_id,
            interview_question_id=first_question.id,
            sequence_no=1,
        )
    )
    await flush_or_conflict(db)


async def take_session_step(
    db: AsyncSession,
    session_id: UUID,
    answer_text: str,
    actor: CurrentUser,
    *,
    audio_object_id: UUID | None = None,
) -> dict[str, Any]:
    """Record the student's answer + advance the session.

    Returns ``{next_question, is_finished, followup_text}``. When the
    follow-up stage decides the answer is shallow, the session does
    NOT advance (single follow-up per question, enforced by
    :func:`maybe_generate_followup`). Otherwise the next approved
    question is appended to ``interview_session_questions`` and
    returned; ``is_finished=True`` signals the question budget is
    exhausted (router prompts the client to ``/finish``).
    """
    session = await _require_session(db, session_id)
    _assert_owns_session(session, actor)
    if session.status != "in_progress":
        raise AppError(f"Cannot record answer on session with status={session.status}")

    current_session_question = await _current_session_question(db, session_id)
    if current_session_question is None:
        raise AppError("Session has no current question to answer")

    db.add(
        InterviewSessionMessage(
            session_id=session_id,
            session_question_id=current_session_question.id,
            role="user",
            content_text=answer_text,
            audio_object_id=audio_object_id,
            metadata_json={"kind": "answer"},
        )
    )
    await flush_or_conflict(db)

    current_question: InterviewQuestion | None = None
    if current_session_question.interview_question_id is not None:
        current_question = await db.get(
            InterviewQuestion, current_session_question.interview_question_id
        )

    followup_text: str | None = None
    if current_question is not None:
        followup_text = await maybe_generate_followup(
            db,
            session=session,
            current_question=current_question,
            student_answer=answer_text,
        )

    if followup_text:
        db.add(
            InterviewSessionMessage(
                session_id=session_id,
                session_question_id=current_session_question.id,
                role="ai",
                content_text=followup_text,
                metadata_json={"kind": "followup"},
            )
        )
        await flush_or_conflict(db)
        return {
            "next_question": None,
            "is_finished": False,
            "followup_text": followup_text,
        }

    asked_ids = await _asked_question_ids(db, session_id)
    next_question = await _next_published_question_after(db, session.interview_config_id, asked_ids)
    if next_question is None:
        return {
            "next_question": None,
            "is_finished": True,
            "followup_text": None,
        }

    sequence_no = await _next_session_sequence(db, session_id)
    db.add(
        InterviewSessionQuestion(
            session_id=session_id,
            interview_question_id=next_question.id,
            sequence_no=sequence_no,
        )
    )
    await flush_or_conflict(db)
    return {
        "next_question": next_question,
        "is_finished": False,
        "followup_text": None,
    }


async def submit_session(
    db: AsyncSession,
    session_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> InterviewSession:
    """Mark session ``completed`` + enqueue evaluation.

    Commits inline so the ARQ worker sees the terminal status before
    the evaluation job dequeues. Tolerates ``arq_pool=None`` (test
    mode); production routers inject the real pool via Depends.
    """
    session = await _require_session(db, session_id)
    _assert_owns_session(session, actor)
    if session.status != "in_progress":
        return session

    session.status = "completed"
    session.ended_at = utcnow()
    await db.commit()
    await db.refresh(session)

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _EVALUATE_INTERVIEW_SESSION_TASK, actor.user_id, session.id
        )
    return session


async def get_session_for_user(
    db: AsyncSession, session_id: UUID, user_id: UUID
) -> InterviewSession | None:
    session = await sessions_queries.get_session(db, session_id)
    if session is None or session.student_id != user_id:
        return None
    return session


async def get_user_sessions(
    db: AsyncSession,
    user_id: UUID,
    *,
    status: str | None = None,
) -> list[InterviewSession]:
    return await sessions_queries.get_user_interview_sessions(db, user_id, status=status)


async def _current_session_question(
    db: AsyncSession, session_id: UUID
) -> InterviewSessionQuestion | None:
    from sqlalchemy import select  # noqa: PLC0415

    stmt = (
        select(InterviewSessionQuestion)
        .where(InterviewSessionQuestion.session_id == session_id)
        .order_by(InterviewSessionQuestion.sequence_no.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _asked_question_ids(db: AsyncSession, session_id: UUID) -> set[UUID]:
    from sqlalchemy import select  # noqa: PLC0415

    stmt = select(InterviewSessionQuestion.interview_question_id).where(
        InterviewSessionQuestion.session_id == session_id
    )
    rows = (await db.execute(stmt)).scalars().all()
    return {row for row in rows if row is not None}


__all__ = [
    "get_session_for_user",
    "get_user_sessions",
    "start_session",
    "submit_session",
    "take_session_step",
]
