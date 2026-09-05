"""Interviews learner SESSION router — session lifecycle routes.

Split out of :mod:`routers.learner` to keep both router modules under the
feature's god-file cap. Holds the session-taking surface (start, onboarding,
voice realtime + dispatch, narration, integrity events, answer, finish) and
the narration-language resolver.

Shared helpers (``_not_found``, ``_bad_request``, ``_conflict``,
``_voice_unavailable``, ``_ensure_config_course_enrolled``,
``_current_session_question``, ``_session_history``, ``get_arq_pool``) live
in :mod:`routers.learner` and are imported here — learner never imports this
module, so there is no import cycle.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.interviews.routers._deps import require_session_owner_access
from abridgeai.features.interviews.routers.learner import (
    _bad_request,
    _conflict,
    _current_session_question,
    _ensure_config_course_enrolled,
    _not_found,
    _session_history,
    _voice_unavailable,
    get_arq_pool,
)
from abridgeai.features.interviews.schemas import (
    InterviewOnboardingRespondRequest,
    InterviewOnboardingRespondResponse,
    InterviewQuestionPublic,
    InterviewSessionFinishRequest,
    InterviewSessionFinishResponse,
    InterviewSessionPublic,
    InterviewSessionStartRequest,
    InterviewSessionStartResponse,
    InterviewSubmitAnswerRequest,
    InterviewSubmitAnswerResponse,
    RealtimeTokenResponse,
)
from abridgeai.features.interviews.schemas.integrity import IntegrityEventBatchRequest
from abridgeai.features.interviews.services import narration as narration_service
from abridgeai.features.interviews.services import narration_cache
from abridgeai.features.interviews.services import onboarding as onboarding_service
from abridgeai.features.interviews.services import real_time as realtime_service
from abridgeai.features.interviews.services import taking as taking_service
from abridgeai.features.interviews.services.ceremony import (
    ensure_ceremony_message,
    onboarding_ceremony_kind,
)
from abridgeai.features.interviews.services.evaluation_state import derive_evaluation_state

router = APIRouter(tags=["interviews-learner-sessions"])

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_REQUIRE_SESSION_OWNER = require_session_owner_access()


@router.post(
    "/interview-configs/{config_id}/sessions",
    response_model=InterviewSessionStartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_session(
    config_id: UUID,
    payload: InterviewSessionStartRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    accept_language: Annotated[str | None, Header()] = None,
) -> InterviewSessionStartResponse:
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None or config.status != "published":
        raise _not_found("interview_config", config_id)
    await _ensure_config_course_enrolled(db, current_user, config)
    try:
        session = await taking_service.start_session(
            db,
            config_id,
            payload,
            current_user,
            language=_resolve_language(accept_language),
        )
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except taking_service.InterviewCooldownActive as exc:
        # FR-5.3 — retake cooldown still active; surface Retry-After.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "cooldown_active",
                "message": str(exc),
                "retry_after": exc.retry_after.isoformat(),
            },
            headers={
                "Retry-After": str(
                    max(0, int((exc.retry_after - datetime.now(UTC)).total_seconds()))
                )
            },
        ) from exc
    except taking_service.InterviewMaxAttemptsReached as exc:
        # FR-5.3 — attempt ceiling reached.
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    first_question = await _current_session_question(db, session.id)
    opening = await ensure_ceremony_message(
        db,
        session=session,
        kind=onboarding_ceremony_kind(session.onboarding_stage),
        language=session.interview_language,
    )
    await db.commit()
    onboarding_complete = session.onboarding_stage == "completed"
    history = await _session_history(db, session)
    return InterviewSessionStartResponse(
        session_id=session.id,
        opening_text=opening.content_text,
        first_question=(
            InterviewQuestionPublic.model_validate(first_question)
            if onboarding_complete and first_question is not None
            else None
        ),
        time_remaining_seconds=await taking_service.session_time_remaining_seconds(db, session),
        question_count_remaining=None,
        onboarding_stage=session.onboarding_stage,
        interview_language=session.interview_language,
        assessment_started_at=session.assessment_started_at,
        history=history,
    )


@router.post(
    "/interview-sessions/{session_id}/onboarding/respond",
    response_model=InterviewOnboardingRespondResponse,
)
async def respond_to_onboarding(
    session_id: UUID,
    payload: InterviewOnboardingRespondRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewOnboardingRespondResponse:
    """Confirm setup/readiness before revealing the first assessed question."""
    try:
        result = await onboarding_service.respond(
            db,
            session_id=session_id,
            actor=current_user,
            stage=payload.stage,
            response_text=payload.response_text,
            action=payload.action,
            language=payload.language,
            turn_key=payload.turn_key,
        )
        await db.commit()
        await db.refresh(result.session)
    except NotFoundError as exc:
        raise _not_found("interview_session", session_id) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": str(exc)},
        ) from exc
    except AppError as exc:
        raise _conflict(str(exc)) from exc

    first_question = (
        await _current_session_question(db, session_id)
        if result.session.onboarding_stage == "completed"
        else None
    )
    return InterviewOnboardingRespondResponse(
        onboarding_stage=result.session.onboarding_stage,
        interview_language=result.session.interview_language,
        ai_text=result.ai_text,
        is_complete=result.session.onboarding_stage == "completed",
        first_question=(
            InterviewQuestionPublic.model_validate(first_question)
            if first_question is not None
            else None
        ),
        assessment_started_at=result.session.assessment_started_at,
        time_remaining_seconds=await taking_service.session_time_remaining_seconds(
            db, result.session
        ),
    )


@router.post(
    "/interview-sessions/{session_id}/realtime-token",
    response_model=RealtimeTokenResponse,
)
async def realtime_token(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
    accept_language: Annotated[str | None, Header()] = None,
    warm: Annotated[bool, Query()] = False,
) -> RealtimeTokenResponse:
    """Mint a LiveKit join token for a voice session.

    Ownership + existence are enforced by ``_REQUIRE_SESSION_OWNER``. Here we
    additionally gate on the voice feature flag and the session's
    mode/status, persist the room name on first call (idempotent), and return
    a short-lived participant token.

    Two shapes:

    * default — the token carries the agent dispatch, so joining starts the
      interviewer. Still requires completed onboarding, because that IS the
      moment the interview begins.
    * ``?warm=true`` — a **warm-up** token with no dispatch, allowed DURING
      onboarding. It lets the client open the room while the candidate is
      still doing setup, so the ~10-13s worker startup (measured) overlaps
      work the candidate is doing anyway instead of sitting in front of
      question one as dead air. The interviewer is sent in afterwards by
      ``POST .../realtime-agent``.

    A warm token grants exactly the same room access as a normal one — it
    simply does not start the interview. The onboarding gate that used to sit
    on every mint is preserved where it actually matters: on the dispatch.
    """
    settings = get_settings()
    if not settings.interview_voice_enabled:
        raise _voice_unavailable()

    from abridgeai.features.interviews.models import InterviewSession  # noqa: PLC0415

    session = await db.get(InterviewSession, session_id)
    if session is None:  # pragma: no cover - dep already 404s; defensive
        raise _not_found("interview_session", session_id)
    if session.status != "in_progress":
        raise _conflict(f"session is not in progress (status={session.status})")
    # Only the DISPATCHING mint waits for onboarding. A warm token starts
    # nothing, so holding it back would just reinstate the delay it exists to
    # remove.
    if not warm and session.onboarding_stage != "completed":
        raise _conflict("Complete interview onboarding before joining the voice room")

    room_name = session.livekit_room_name or realtime_service.build_room_name(session_id)
    if session.livekit_room_name is None:
        session.livekit_room_name = room_name
        await db.commit()

    try:
        return realtime_service.mint_participant_token(
            session_id=session_id,
            student_id=current_user.user_id,
            room_name=room_name,
            settings=settings,
            language=session.interview_language,
            dispatch_agent=not warm,
        )
    except ValueError as exc:  # credentials missing despite the flag
        raise _voice_unavailable() from exc


@router.post(
    "/interview-sessions/{session_id}/realtime-agent",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def dispatch_realtime_agent(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Send the interviewer into a room the candidate already warmed.

    The second half of the warm-room flow: the client joined during setup with
    ``?warm=true`` (which starts nothing), and calls this once onboarding is
    complete. The onboarding gate lives HERE now — this is the call that
    actually begins the interview, so it carries the same guarantee the
    token-mint gate used to.

    ``language`` is read from the session at THIS point rather than at warm-up,
    which is the other reason the split matters: the language check is part of
    onboarding, so a dispatch embedded in an early token would have shipped a
    stale value.
    """
    settings = get_settings()
    if not settings.interview_voice_enabled:
        raise _voice_unavailable()

    from abridgeai.features.interviews.models import InterviewSession  # noqa: PLC0415

    session = await db.get(InterviewSession, session_id)
    if session is None:  # pragma: no cover - dep already 404s; defensive
        raise _not_found("interview_session", session_id)
    if session.status != "in_progress":
        raise _conflict(f"session is not in progress (status={session.status})")
    if session.onboarding_stage != "completed":
        raise _conflict("Complete interview onboarding before the interviewer joins")
    if session.livekit_room_name is None:
        raise _conflict("no voice room has been opened for this session")

    try:
        await realtime_service.dispatch_interview_agent(
            session_id=session_id,
            student_id=current_user.user_id,
            room_name=session.livekit_room_name,
            settings=settings,
            language=session.interview_language,
        )
    except ValueError as exc:  # credentials missing despite the flag
        raise _voice_unavailable() from exc
    except Exception as exc:  # noqa: BLE001 -- surfaced, never swallowed
        # A room with no interviewer is a dead interview. Report it so the
        # client can fall back to the token-embedded dispatch path rather than
        # leaving the candidate staring at silence.
        logger.exception("realtime agent dispatch failed (session=%s)", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "agent_dispatch_failed", "message": "Could not start the interviewer"},
        ) from exc


class NarrationRequest(BaseModel):
    """Body for ``POST /interview-sessions/{session_id}/narration``.

    The client sends the exact AI utterance (question or follow-up) it is
    rendering so the server can synthesize matching speech. Persona is derived
    server-side from the session's config — the client never chooses the voice.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1200)


@router.post("/interview-sessions/{session_id}/narration")
async def narrate_session_text(
    session_id: UUID,
    payload: NarrationRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
    accept_language: Annotated[str | None, Header(alias="Accept-Language")] = None,
) -> Response:
    """Synthesize an AI utterance to MP3 using the agent-quality gateway TTS.

    Gives text/hybrid sessions the same *voice* as the LiveKit agent without
    mounting a realtime room (which would race the REST loop for control of
    the session). Persona → voice mapping is resolved from the session's
    config. On any TTS failure returns 503 so the browser falls back to its
    local speech synthesizer.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )

    session = await db.get(InterviewSession, session_id)
    if session is None:  # pragma: no cover - dep already 404s; defensive
        raise _not_found("interview_session", session_id)

    # Narration is an output boundary, not an arbitrary text-to-speech proxy.
    # Accept only a question already attached to this session or an AI message
    # that was persisted after the shared output guard.
    import hashlib  # noqa: PLC0415

    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionMessage,
        InterviewSessionQuestion,
    )
    from abridgeai.features.interviews.orchestrator.security import (  # noqa: PLC0415
        SecurityAction,
        SecurityAssessment,
        SecurityCategory,
    )
    from abridgeai.features.interviews.services import security as security_service  # noqa: PLC0415

    asked_ids = list(
        (
            await db.execute(
                select(InterviewSessionQuestion.interview_question_id).where(
                    InterviewSessionQuestion.session_id == session_id,
                    InterviewSessionQuestion.interview_question_id.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    ai_text_exists = await db.scalar(
        select(InterviewSessionMessage.id).where(
            InterviewSessionMessage.session_id == session_id,
            InterviewSessionMessage.role == "ai",
            InterviewSessionMessage.content_text == payload.text.strip(),
        )
    )
    question_text_exists = None
    if asked_ids:
        question_text_exists = await db.scalar(
            select(InterviewQuestion.id).where(
                InterviewQuestion.id.in_(asked_ids),
                InterviewQuestion.prompt_text == payload.text.strip(),
            )
        )
    if ai_text_exists is None and question_text_exists is None:
        raise _bad_request("narration text is not an approved interview utterance")

    narration_assessment = SecurityAssessment(
        category=SecurityCategory.BENIGN,
        detected=False,
        confidence=1.0,
        should_block=False,
        should_record_academic_evidence=False,
        response_key=None,
        normalized_fingerprint=None,
        source="narration_boundary",
    )
    narration_key = "narration:" + hashlib.sha256(payload.text.encode()).hexdigest()[:32]
    # Narration provider is chosen by the SESSION's interview_language, NOT the
    # Accept-Language UI locale. An English interview viewed with a Vietnamese
    # UI must still narrate in English via Deepgram — routing by the header sent
    # English sessions down the VI gateway path (tts-1), which 403s on this
    # deployment (no gateway TTS model), forcing a 503 + browser-voice fallback.
    # interview_language is NOT NULL and constrained to 'en'/'vi'.
    narration_language = realtime_service.normalize_language(
        session.interview_language or accept_language
    )
    guarded_narration = await security_service.guard_student_output(
        db,
        session_id=session_id,
        config_id=session.interview_config_id,
        turn_key=narration_key,
        proposed_text=payload.text.strip(),
        fallback_text=(
            "Tôi có thể nhắc lại hoặc giải thích câu hỏi hiện tại."
            if narration_language == "vi"
            else "I can repeat or clarify the current question."
        ),
        allowed_question_ids=[qid for qid in asked_ids if qid is not None],
        assessment=narration_assessment,
        action=SecurityAction.ALLOW,
        attempt_count=0,
    )
    await db.commit()

    persona: str | None = None
    tts_voice: str | None = None
    config = await db.get(InterviewConfig, session.interview_config_id)
    if config is not None:
        persona = config.persona
        tts_voice = config.tts_voice

    settings = get_settings()
    # Cache lookup sits BEHIND the output guard above: only text already
    # approved as an interview utterance can reach it, so this never widens
    # what the endpoint will speak. The win is the ceremony lines — the
    # transition ("Great—the introduction is complete…") is a fixed string,
    # identical for every session in a language, yet cost a full ~3.0-3.6s
    # Deepgram round trip every time. The browser holds its "preparing"
    # indicator for that whole window (text and voice are released together),
    # which is exactly the delay reported at the head of that line.
    cached = narration_cache.get(
        text=guarded_narration.text,
        voice=tts_voice,
        persona=persona,
        language=narration_language,
    )
    if cached is not None:
        return Response(
            content=cached,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-store"},
        )

    try:
        audio = await narration_service.synthesize_speech(
            guarded_narration.text,
            persona=persona,
            settings=settings,
            language=narration_language,
            voice=tts_voice,
        )
    except narration_service.NarrationUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "narration_unavailable", "message": str(exc)},
        ) from exc

    narration_cache.put(
        text=guarded_narration.text,
        voice=tts_voice,
        persona=persona,
        language=narration_language,
        audio=audio,
    )

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/interview-sessions/{session_id}/integrity-events",
    status_code=status.HTTP_202_ACCEPTED,
)
async def record_integrity_events(
    session_id: UUID,
    payload: IntegrityEventBatchRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, int]:
    """Best-effort ingest of browser integrity signals for a live session.

    Owner/existence enforced by the dep. Events are recorded only while the
    session is ``in_progress`` (late events for finished sessions are silently
    dropped — this never blocks the interview). Post-session / teacher review
    only; never surfaced to the student.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        AssessmentIntegrityEvent,
        InterviewSession,
    )

    session = await db.get(InterviewSession, session_id)
    if session is None:  # pragma: no cover - dep already 404s; defensive
        raise _not_found("interview_session", session_id)
    if session.status != "in_progress":
        return {"accepted": 0}

    for item in payload.events:
        db.add(
            AssessmentIntegrityEvent(
                assessment_kind="interview",
                interview_session_id=session_id,
                student_id=current_user.user_id,
                event_type=item.event_type,
                severity=item.severity,
                metadata_json=dict(item.metadata),
            )
        )
    await db.commit()
    return {"accepted": len(payload.events)}


@router.get("/interview-sessions/{session_id}", response_model=InterviewSessionPublic)
async def get_session(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewSessionPublic:
    session = await taking_service.get_session_for_user(db, session_id, current_user.user_id)
    if session is None:
        raise _not_found("interview_session", session_id)
    retake = await taking_service.compute_retake_status(
        db, config_id=session.interview_config_id, student_id=current_user.user_id
    )
    return InterviewSessionPublic(
        session_id=session.id,
        interview_config_id=session.interview_config_id,
        status=session.status,
        input_mode=session.input_mode,
        attempt_number=session.attempt_number,
        started_at=session.started_at,
        assessment_started_at=session.assessment_started_at,
        onboarding_stage=session.onboarding_stage,
        interview_language=session.interview_language,
        ended_at=session.ended_at,
        resume_deadline_at=session.resume_deadline_at,
        current_question_index=None,
        time_remaining_seconds=await taking_service.session_time_remaining_seconds(db, session),
        pass_verdict=session.pass_verdict,
        evaluation_state=derive_evaluation_state(session),
        remaining_attempts=retake.remaining_attempts,
        retake_available_at=retake.retake_available_at,
        can_retake=retake.can_retake,
    )


def _resolve_language(accept_language: str | None) -> str:
    """Map an Accept-Language header to the interview utterance language.

    Returns 'vi' when Vietnamese is the leading preference, else 'en'. Kept
    deliberately simple — the frontend also drives narration language client
    side; this only shapes the server-generated interviewer utterance.
    """
    if accept_language and accept_language.strip().lower().startswith("vi"):
        return "vi"
    return "en"


@router.post(
    "/interview-sessions/{session_id}/respond",
    response_model=InterviewSubmitAnswerResponse,
)
async def respond_to_session(
    session_id: UUID,
    payload: InterviewSubmitAnswerRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
    accept_language: Annotated[str | None, Header()] = None,
) -> InterviewSubmitAnswerResponse:
    if payload.session_id != session_id:
        raise _bad_request("session_id mismatch")
    if payload.answer_text is None and payload.audio_object_id is None:
        raise _bad_request("answer_text or audio_object_id is required")
    answer_text = payload.answer_text or ""
    try:
        result = await taking_service.take_session_step(
            db,
            session_id,
            answer_text,
            current_user,
            audio_object_id=payload.audio_object_id,
            turn_key=payload.turn_key,
            language=_resolve_language(accept_language),
            turn_action=payload.turn_action or "answer",
        )
    except NotFoundError as exc:
        raise _not_found("interview_session", session_id) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": str(exc)},
        ) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- log internally; return an allowlisted error
        await db.rollback()
        logger.exception("respond_to_session: unhandled error (session=%s)", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": "Unable to process this turn"},
        ) from exc
    try:
        await db.commit()
    except Exception as exc:  # noqa: BLE001 -- log internally; return an allowlisted error
        await db.rollback()
        logger.exception("respond_to_session: commit failed (session=%s)", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": "Unable to save this turn"},
        ) from exc
    # Server-authoritative timer (resilience A-Tier-1 #4): return the current
    # whole-second countdown on every turn so the client reconciles its locally
    # computed deadline against the server clock instead of trusting a value
    # captured once at start. Best-effort — a lookup failure must never fail the
    # turn (the answer is already committed), so fall back to None.
    remaining_seconds: int | None = None
    try:
        fresh_session = await taking_service.get_session_for_user(
            db, session_id, current_user.user_id
        )
        if fresh_session is not None:
            remaining_seconds = await taking_service.session_time_remaining_seconds(
                db, fresh_session
            )
    except Exception:  # noqa: BLE001 -- timer reconciliation is advisory, never fatal
        logger.warning("respond_to_session: time-remaining lookup failed (session=%s)", session_id)
    # Built by the shared projection so the LiveKit control topic (which carries
    # this same state for typed turns over `lk.chat`) cannot drift from REST.
    return InterviewSubmitAnswerResponse.from_step_result(
        result,
        time_remaining_seconds=remaining_seconds,
    )


@router.post(
    "/interview-sessions/{session_id}/finish",
    response_model=InterviewSessionFinishResponse,
)
async def finish_session(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
    payload: InterviewSessionFinishRequest | None = None,
    accept_language: Annotated[str | None, Header()] = None,
) -> InterviewSessionFinishResponse:
    finish_reason = payload.reason if payload is not None else "natural"
    try:
        session = await taking_service.submit_session(
            db,
            session_id,
            current_user,
            arq_pool=arq_pool,
            reason=finish_reason,
            language=_resolve_language(accept_language),
        )
    except NotFoundError as exc:
        raise _not_found("interview_session", session_id) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": str(exc)},
        ) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- log internally; return an allowlisted error
        await db.rollback()
        logger.exception("finish_session: unhandled error (session=%s)", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": "Unable to finish this session"},
        ) from exc
    # Thesis §4.3: the student-facing result is binary pass/fail ONLY — no
    # score, no per-outcome breakdown. The rubric total + per-outcome verdicts
    # live in internal_summary_json / InterviewOutcomeEvaluation for teachers.
    closing = await ensure_ceremony_message(
        db,
        session=session,
        kind="closing",
        language=session.interview_language,
        reason=finish_reason,
    )
    # ``submit_session`` commits the normal path. This additional commit makes
    # the additive ceremony field self-healing for older terminal sessions that
    # predate persisted closing turns.
    await db.commit()
    retake = await taking_service.compute_retake_status(
        db, config_id=session.interview_config_id, student_id=current_user.user_id
    )
    return InterviewSessionFinishResponse(
        session_id=session.id,
        status=session.status,
        closing_text=closing.content_text,
        total_score=None,
        rubric_scores=[],
        pass_verdict=session.pass_verdict,
        ended_at=session.ended_at,
        remaining_attempts=retake.remaining_attempts,
        retake_available_at=retake.retake_available_at,
        can_retake=retake.can_retake,
    )


