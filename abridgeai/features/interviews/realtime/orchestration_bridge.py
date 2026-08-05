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
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.security import CurrentUser
from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    identity_from_config,
)
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.schemas.session import InterviewSubmitAnswerResponse
from abridgeai.features.interviews.services.ceremony import (
    ensure_ceremony_message,
    onboarding_ceremony_kind,
    room_intro_text,
)
from abridgeai.features.interviews.services.taking import (
    get_session_for_user,
    session_time_remaining_seconds,
    submit_session,
    take_session_step,
)

logger = logging.getLogger(__name__)

# One ARQ pool per worker process (lazily created). Reused so we don't open a
# Redis connection per finished session.
_arq_pool: ArqRedis | None = None


@dataclass(frozen=True)
class TurnResult:
    """What the agent should speak after a student turn, and whether to end.

    ``suppress_default_closing`` remains for compatibility with existing voice
    runtime consumers. Ceremony-aware terminal turns always carry their one
    canonical closing in ``speak_text`` and set this flag.
    """

    speak_text: str | None
    is_finished: bool
    suppress_default_closing: bool = False

    # ── Structured turn state, for clients that drive the interview over
    # LiveKit text streams instead of REST `/respond` (hybrid text mode).
    #
    # `turn_state` is the FULL `InterviewSubmitAnswerResponse` payload, already
    # JSON-serialized (mode="json", so UUIDs and datetimes are strings). It is
    # built by `InterviewSubmitAnswerResponse.from_step_result` — the same
    # classmethod the REST route uses — so a typed client over `lk.chat` receives
    # byte-identical state to a REST client, including `next_question` as a
    # properly projected `InterviewQuestionPublic`.
    #
    # Deliberately NOT a hand-picked subset: the earlier version listed six
    # fields, which silently dropped nine (next_question, transition_*,
    # pending_confirmation, assistance_kind, …) and published a null timer,
    # because the brain never returns `time_remaining_seconds` at all.
    #
    # Empty dict on the voice path, which ignores it entirely.
    turn_state: dict[str, Any] = field(default_factory=dict)

    # The brain's OWN per-session version (from `take_session_step`), NOT a
    # counter invented here: it is what the client reconciles persisted history
    # against. Kept as a top-level field because it is control-plane metadata
    # rather than part of the REST response body.
    state_version: int | None = None


def _build_actor(student_id: UUID) -> CurrentUser:
    # session_id is unused by the taking service (only user_id matters for
    # ownership + the evaluation enqueue); a synthetic value is fine here.
    return CurrentUser(user_id=student_id, session_id=uuid4(), permissions=frozenset())


async def _get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _arq_pool


async def get_current_question_text(session_id: UUID, *, language: str = "en") -> str | None:
    """Prompt text of the session's current (highest-sequence) question.

    Used to speak the first question on join — and, on a mid-session rejoin,
    to re-speak whatever question is currently pending.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSession,
        InterviewSessionQuestion,
    )
    from abridgeai.features.interviews.orchestrator.security import (  # noqa: PLC0415
        SecurityAction,
        SecurityAssessment,
        SecurityCategory,
    )
    from abridgeai.features.interviews.services import security as security_service  # noqa: PLC0415

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
        session = await db.get(InterviewSession, session_id)
        if question is None or session is None:
            return None
        if getattr(session, "onboarding_stage", "completed") != "completed":
            return None
        assessment = SecurityAssessment(
            category=SecurityCategory.BENIGN,
            detected=False,
            confidence=1.0,
            should_block=False,
            should_record_academic_evidence=False,
            response_key=None,
            normalized_fingerprint=None,
            source="livekit_output_boundary",
        )
        fallback = (
            "Tôi có thể nhắc lại hoặc giải thích câu hỏi hiện tại."
            if language.lower().startswith("vi")
            else "I can repeat or clarify the current question."
        )
        guarded = await security_service.guard_student_output(
            db,
            session_id=session_id,
            config_id=session.interview_config_id,
            turn_key=f"livekit-current:{sq.id}",
            proposed_text=question.prompt_text,
            fallback_text=fallback,
            allowed_question_ids=[question.id],
            assessment=assessment,
            action=SecurityAction.ALLOW,
            attempt_count=0,
        )
        await db.commit()
        return guarded.text


async def get_opening_text(session_id: UUID, *, language: str = "en") -> str | None:
    """Return the persisted greeting for a voice session."""
    from abridgeai.features.interviews.models import InterviewSession  # noqa: PLC0415

    async with get_sessionmaker()() as db:
        session = await db.get(InterviewSession, session_id)
        if session is None:
            return None
        if getattr(session, "onboarding_stage", "completed") == "completed":
            return None
        message = await ensure_ceremony_message(
            db,
            session=session,
            kind=onboarding_ceremony_kind(session.onboarding_stage),
            language=getattr(session, "interview_language", language),
        )
        await db.commit()
        return message.content_text


async def get_room_intro_text(session_id: UUID, *, language: str = "en") -> str | None:
    """Short line spoken as the voice channel opens, or ``None``.

    Unlike :func:`get_opening_text` this deliberately does NOT short-circuit on
    ``onboarding_stage == "completed"``. That short-circuit, combined with
    ``/realtime-token`` refusing to mint a token until onboarding IS complete,
    is why the opening branch in ``InterviewAgent.on_enter`` never ran in
    production and a voice candidate's first experience was a bank question
    with nobody introducing it.

    ``None`` whenever the config has not opted into a named interviewer, so
    existing voice sessions are unchanged.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )

    async with get_sessionmaker()() as db:
        session = await db.get(InterviewSession, session_id)
        if session is None:
            return None
        config = await db.get(InterviewConfig, session.interview_config_id)
        return room_intro_text(
            identity=identity_from_config(getattr(config, "persona_profile_json", None)),
            language=getattr(session, "interview_language", language),
        )


async def get_voice_persona(session_id: UUID) -> tuple[str | None, int]:
    """``(persona label, verbosity dial)`` for the session's config.

    The realtime path never had access to persona: it read only ``tts_voice``,
    so tone reached voice indirectly (through the phrasing layer) and not at all
    in the audio itself on the Vietnamese branch. Returns the neutral default
    when anything is missing, matching ``persona.profile_from``.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )
    from abridgeai.features.interviews.orchestrator.persona import (  # noqa: PLC0415
        profile_from_config,
    )

    async with get_sessionmaker()() as db:
        session = await db.get(InterviewSession, session_id)
        if session is None:
            return None, 2
        config = await db.get(InterviewConfig, session.interview_config_id)
        persona = getattr(config, "persona", None)
        profile = profile_from_config(persona, getattr(config, "persona_profile_json", None))
        return persona, profile.verbosity


async def get_tts_voice(session_id: UUID) -> str | None:
    """Return the config's chosen Deepgram Aura voice for a voice session.

    ``None`` when the session/config is missing or no voice is set, in which
    case the runtime falls back to the deployment default.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )

    async with get_sessionmaker()() as db:
        session = await db.get(InterviewSession, session_id)
        if session is None:
            return None
        config = await db.get(InterviewConfig, session.interview_config_id)
        return getattr(config, "tts_voice", None) if config is not None else None


async def handle_student_turn(
    session_id: UUID,
    student_id: UUID,
    transcript: str,
    *,
    language: str = "en",
    turn_id: str | None = None,
    turn_action: str = "answer",
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
        result = await take_session_step(
            db,
            session_id,
            transcript,
            actor,
            language=language,
            turn_key=turn_id,
            turn_action=turn_action,
        )

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

        # Structured state for text clients, built ONCE here and spread into
        # every return below so no branch can drift from another. The voice path
        # ignores these fields entirely.
        #
        # Built through `InterviewSubmitAnswerResponse.from_step_result` — the
        # SAME projection REST `/respond` returns — so a typed client over
        # `lk.chat` gets full parity, `next_question` included, without the ORM
        # row ever escaping. `mode="json"` renders UUIDs/datetimes as strings so
        # the payload is directly JSON-serializable onto the control topic.
        #
        # The timer is computed here rather than read from `result`: the brain
        # does not return `time_remaining_seconds`, so reading it from the result
        # dict yields None (the bug this replaces). Best-effort, exactly like the
        # REST route — a lookup failure must never fail a committed turn.
        remaining_seconds: int | None = None
        try:
            live_session = await get_session_for_user(db, session_id, student_id)
            if live_session is not None:
                remaining_seconds = await session_time_remaining_seconds(db, live_session)
        except Exception:  # noqa: BLE001 — advisory timer; never fail the turn
            logger.warning(
                "handle_student_turn: time-remaining lookup failed (session=%s)",
                session_id,
            )

        turn_payload = InterviewSubmitAnswerResponse.from_step_result(
            result,
            time_remaining_seconds=remaining_seconds,
        ).model_dump(mode="json")
        # Spread into every TurnResult below so no branch can drift. Named
        # distinctly from the payload it carries: `turn_state` is the REST-parity
        # body, `state_version` is control-plane metadata beside it.
        control_fields: dict[str, Any] = {
            "turn_state": turn_payload,
            "state_version": result.get("state_version"),
        }

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
            session = await submit_session(
                db,
                session_id,
                actor,
                arq_pool=pool,
                reason="natural",
                language=language,
            )
            closing_message = await ensure_ceremony_message(
                db,
                session=session,
                kind="closing",
                language=language,
                reason="natural",
            )
            obs.emit(obs.EV_SESSION_SUBMITTED, session_id=session_id, turn_id=turn_id)
            obs.emit(obs.EV_EVALUATION_ENQUEUED, session_id=session_id, turn_id=turn_id)
            # Ceremony is the canonical terminal utterance for both adaptive
            # and legacy paths. This prevents the adaptive phrasing layer from
            # asking one more question after the interview is already final.
            closing = closing_message.content_text
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
                **control_fields,
            )

        # Not finished. Prefer the adaptive combined utterance when present: it
        # already contains ack + transition + the selected question, so we must
        # NOT also append next_question (that would double-speak the question).
        if ai_turn_text:
            return TurnResult(speak_text=ai_turn_text, is_finished=False, **control_fields)

        # ── Legacy path (adaptive off / failed) — unchanged behaviour ────────
        if followup_text:
            return TurnResult(speak_text=followup_text, is_finished=False, **control_fields)

        if next_question_text is not None:
            return TurnResult(speak_text=next_question_text, is_finished=False, **control_fields)

        # Defensive: no follow-up, not finished, no next question.
        logger.warning("interview turn produced no next utterance (session=%s)", session_id)
        return TurnResult(speak_text=None, is_finished=True, **control_fields)


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
    "get_opening_text",
    "handle_student_turn",
    "record_integrity_event",
]
