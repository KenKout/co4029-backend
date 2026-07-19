"""Bridge between the LiveKit voice agent and the existing interview brain.

The voice agent does I/O only (STT/TTS). *What to say next* is decided by the
already-built, teacher-reviewed text orchestration in
:mod:`features.interviews.services.taking` — so voice mode reaches exact
parity with text mode and persists byte-for-byte identical state (the user
answer + any AI follow-up are written by ``take_session_step``; questions are
recorded as ``InterviewSessionQuestion`` rows exactly as in text mode). We do
NOT persist extra AI-question rows here: that would diverge from text mode and
could confuse the post-session evaluation stage.

Trusted context: the service layer takes an ``actor: CurrentUser`` but
ownership is enforced at the HTTP router. This worker is a trusted server
process; the (session_id, student_id) pair was already authorized when the
join token was minted (Phase 2), so we construct a ``CurrentUser`` directly.
This module is never exposed over HTTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.security import CurrentUser
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.services.taking import submit_session, take_session_step

logger = logging.getLogger(__name__)

# One ARQ pool per worker process (lazily created). Reused so we don't open a
# Redis connection per finished session.
_arq_pool: ArqRedis | None = None


@dataclass(frozen=True)
class TurnResult:
    """What the agent should speak after a student turn, and whether to end.

    ``suppress_default_closing`` is True when the adaptive interviewer already
    produced its own closing utterance (in ``speak_text``) — the runtime then
    skips the generic canned closing remark so the student doesn't hear two
    closings. False on the legacy path (no adaptive closing was generated), so
    the runtime speaks its canned remark exactly as before.
    """

    speak_text: str | None
    is_finished: bool
    suppress_default_closing: bool = False


def _build_actor(student_id: UUID) -> CurrentUser:
    # session_id is unused by the taking service (only user_id matters for
    # ownership + the evaluation enqueue); a synthetic value is fine here.
    return CurrentUser(user_id=student_id, session_id=uuid4(), permissions=frozenset())


async def _get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _arq_pool


async def get_current_question_text(session_id: UUID) -> str | None:
    """Prompt text of the session's current (highest-sequence) question.

    Used to speak the first question on join — and, on a mid-session rejoin,
    to re-speak whatever question is currently pending.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )

    async with get_sessionmaker()() as db:
        sq = (
            await db.execute(
                select(InterviewSessionQuestion)
                .where(InterviewSessionQuestion.session_id == session_id)
                .order_by(InterviewSessionQuestion.sequence_no.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if sq is None or sq.interview_question_id is None:
            return None
        question = await db.get(InterviewQuestion, sq.interview_question_id)
        return question.prompt_text if question is not None else None


async def handle_student_turn(
    session_id: UUID,
    student_id: UUID,
    transcript: str,
    *,
    language: str = "en",
    turn_id: str | None = None,
) -> TurnResult:
    """Feed a transcribed answer to the text brain; return the next utterance.

    Mirrors the ``/respond`` → ``/finish`` HTTP flow but in-process: persists
    the answer (+ any follow-up) via ``take_session_step``, and on a finished
    session calls ``submit_session`` to enqueue the async evaluation.

    ``language`` (\"vi\"/\"en\") shapes the adaptive interviewer's spoken utterance,
    reaching parity with the REST path (which reads Accept-Language). It flows
    from the LiveKit dispatch metadata set when the join token was minted.

    ``turn_id`` correlates the decision-level observability events emitted here
    with the I/O-level events emitted by the runtime for the same turn.
    """
    actor = _build_actor(student_id)
    async with get_sessionmaker()() as db:
        result = await take_session_step(db, session_id, transcript, actor, language=language)

        # ``ai_turn_text`` is the adaptive combined utterance (ack + transition +
        # question/probe). It is present ONLY on the adaptive path; on the legacy
        # path it is None and we fall back to the legacy fields. Read everything
        # we need (incl. next_question.prompt_text) while still attached to the
        # session, THEN commit.
        ai_turn_text = result.get("ai_turn_text")
        followup_text = result.get("followup_text")
        next_question = result.get("next_question")
        next_question_text = next_question.prompt_text if next_question is not None else None
        # Question metadata (Phase 7) — surface WHAT KIND of question the adaptive
        # brain selected on an advance, so sign-off can see the difficulty/type
        # mix without reading transcripts. Only populated on an advance (a
        # freshly-selected question rides in ``next_question``); probe/clarify/
        # repeat/closing turns re-use the current question and leave these None.
        # Read while still attached to the session, before the commit below.
        selected_question_type = (
            getattr(next_question, "question_type", None) if next_question is not None else None
        )
        selected_question_difficulty = (
            getattr(next_question, "difficulty", None) if next_question is not None else None
        )
        # should_finish is the adaptive end signal; is_finished the legacy one.
        finished = bool(result.get("should_finish") or result.get("is_finished"))

        # ── Observability: decision-level event (no transcript content) ──────
        # ``action`` is present only on the adaptive path; its absence is the
        # signal that this turn ran the legacy sequential path.
        action = result.get("action")
        utterance_status = result.get("_utterance_status")
        obs.emit(
            obs.EV_DECISION,
            session_id=session_id,
            turn_id=turn_id,
            adaptive=action is not None,
            action=action,
            reason_code=result.get("reason_code"),
            selected_question_id=result.get("current_question_id"),
            selected_question_type=selected_question_type,
            selected_question_difficulty=selected_question_difficulty,
            target_outcome_id=result.get("target_outcome_id"),
            state_version=result.get("state_version"),
            utterance_status=utterance_status,
            answer_chars=len(transcript),
            finished=finished,
        )
        if utterance_status == "fallback":
            # The adaptive brain ran but phrasing degraded to the deterministic
            # template — a key quality signal (see the language-render bug).
            obs.emit(
                obs.EV_FALLBACK,
                session_id=session_id,
                turn_id=turn_id,
                action=action,
            )

        await db.commit()

        if finished:
            pool = await _get_arq_pool()
            await submit_session(db, session_id, actor, arq_pool=pool)
            obs.emit(obs.EV_SESSION_SUBMITTED, session_id=session_id, turn_id=turn_id)
            obs.emit(obs.EV_EVALUATION_ENQUEUED, session_id=session_id, turn_id=turn_id)
            # If the adaptive path generated a closing utterance, speak THAT and
            # suppress the runtime's canned remark (avoid a double closing).
            closing = ai_turn_text or followup_text
            suppress = closing is not None
            obs.emit(
                obs.EV_CLOSING_EMITTED,
                session_id=session_id,
                turn_id=turn_id,
                adaptive_closing=suppress,
                closing_chars=len(closing) if closing else 0,
            )
            if suppress:
                obs.emit(
                    obs.EV_DEFAULT_CLOSING_SUPPRESSED,
                    session_id=session_id,
                    turn_id=turn_id,
                )
            return TurnResult(
                speak_text=closing,
                is_finished=True,
                suppress_default_closing=suppress,
            )

        # Not finished. Prefer the adaptive combined utterance when present: it
        # already contains ack + transition + the selected question, so we must
        # NOT also append next_question (that would double-speak the question).
        if ai_turn_text:
            return TurnResult(speak_text=ai_turn_text, is_finished=False)

        # ── Legacy path (adaptive off / failed) — unchanged behaviour ────────
        if followup_text:
            return TurnResult(speak_text=followup_text, is_finished=False)

        if next_question_text is not None:
            return TurnResult(speak_text=next_question_text, is_finished=False)

        # Defensive: no follow-up, not finished, no next question.
        logger.warning("interview turn produced no next utterance (session=%s)", session_id)
        return TurnResult(speak_text=None, is_finished=True)


async def record_integrity_event(
    session_id: UUID,
    student_id: UUID,
    event_type: str,
    *,
    severity: str = "info",
) -> None:
    """Persist an agent-observed integrity signal (e.g. participant disconnect).

    Best-effort: only writes while the session is still ``in_progress`` (we do
    NOT terminate on disconnect — the student may rejoin; the time-limit sweep
    is the backstop). Failures are swallowed so they never crash the agent.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        AssessmentIntegrityEvent,
        InterviewSession,
    )

    try:
        async with get_sessionmaker()() as db:
            session = await db.get(InterviewSession, session_id)
            if session is None or session.status != "in_progress":
                return
            db.add(
                AssessmentIntegrityEvent(
                    assessment_kind="interview",
                    interview_session_id=session_id,
                    student_id=student_id,
                    event_type=event_type,
                    severity=severity,
                    metadata_json={"source": "agent"},
                )
            )
            await db.commit()
    except Exception:
        logger.exception("failed to record integrity event (session=%s)", session_id)


__all__ = [
    "TurnResult",
    "get_current_question_text",
    "handle_student_turn",
    "record_integrity_event",
]
