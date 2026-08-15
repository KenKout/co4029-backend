"""Interviews learner router (T6.12).

Seven endpoints (no router prefix; the legacy paths self-prefix under
``/interview-configs``, ``/interview-sessions``, and ``/me``). Composes
:mod:`features.interviews.services.taking` for the session lifecycle
and :mod:`features.interviews.queries.published` for the take payload.

Voice future-proofing
---------------------
``POST /interview-sessions/{session_id}/respond`` accepts an optional
``audio_storage_object_id`` on the body. The service layer stores it
on the :class:`InterviewSessionMessage` row but performs NO
transcription — the field is the forward-compat hook for the future
voice-mode build out (STT lands separately).

Security invariant
------------------
Every session-scoped endpoint depends on
:func:`features.interviews.routers._deps.require_session_owner_access`
which closes the perimeter at the HTTP boundary. The service layer
also raises :class:`ForbiddenError` on cross-user access — defence in
depth.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.courses.api.public import can_view_course_content
from abridgeai.features.interviews.routers._deps import require_session_owner_access
from abridgeai.features.interviews.schemas import (
    GapReportRead,
    InterviewConfigPublic,
    InterviewForTakingPublic,
    InterviewOnboardingRespondRequest,
    InterviewOnboardingRespondResponse,
    InterviewProgressRead,
    InterviewQuestionPublic,
    InterviewSessionFinishRequest,
    InterviewSessionFinishResponse,
    InterviewSessionHistoryTurn,
    InterviewSessionPublic,
    InterviewSessionStartRequest,
    InterviewSessionStartResponse,
    InterviewSubmitAnswerRequest,
    InterviewSubmitAnswerResponse,
    RealtimeTokenResponse,
)
from abridgeai.features.interviews.schemas.integrity import IntegrityEventBatchRequest
from abridgeai.features.interviews.services import (
    learner_progress as learner_progress_service,
)
from abridgeai.features.interviews.services import narration as narration_service
from abridgeai.features.interviews.services import narration_cache
from abridgeai.features.interviews.services import onboarding as onboarding_service
from abridgeai.features.interviews.services import real_time as realtime_service
from abridgeai.features.interviews.services import taking as taking_service
from abridgeai.features.interviews.services.ceremony import (
    ensure_ceremony_message,
    onboarding_ceremony_kind,
)

router = APIRouter(tags=["interviews-learner"])

if TYPE_CHECKING:
    from abridgeai.features.interviews.models import (
        InterviewConfig,
        InterviewQuestion,
        InterviewSession,
        InterviewSessionMessage,
        InterviewSessionQuestion,
    )

logger = logging.getLogger(__name__)

_REQUIRE_SESSION_OWNER = require_session_owner_access()


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": message},
    )


def _conflict(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": message},
    )


def _voice_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "voice_unavailable", "message": "Voice interviews are not enabled"},
    )


async def _ensure_config_course_enrolled(
    db: AsyncSession, current_user: CurrentUser, config: InterviewConfig
) -> None:
    """BR gate: an interview config is a course item — enrollment required.

    A published config is only reachable through a course's curriculum,
    and per BR an unenrolled student must not reach ANY course item. This
    404s (the feature's no-existence-leak convention) unless the caller
    has an ``active`` / ``completed`` enrollment in the owning course or
    holds course-management rights (owner/teacher preview).
    """
    from abridgeai.features.access_control.policies import (  # noqa: PLC0415
        can_manage_course,
    )
    from abridgeai.features.enrollments.api import public as enrollments_api  # noqa: PLC0415

    course_id = config.course_id
    if await enrollments_api.has_active_or_completed_enrollment(
        db, student_id=current_user.user_id, course_id=course_id
    ):
        return
    if await can_manage_course(db, current_user.user_id, course_id):
        return
    raise _not_found("interview_config", config.id)


async def get_arq_pool() -> object | None:
    return None


@router.get("/interview-configs/{config_id}", response_model=InterviewForTakingPublic)
async def get_interview_for_taking(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewForTakingPublic:
    from sqlalchemy import func, select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewOutcome,
        InterviewQuestion,
        InterviewSession,
    )

    config = await db.get(InterviewConfig, config_id)
    if config is None or config.status != "published":
        raise _not_found("interview_config", config_id)
    await _ensure_config_course_enrolled(db, current_user, config)
    if config.max_attempts is not None and config.max_attempts > 0:
        used = (
            await db.execute(
                select(func.count(InterviewSession.id)).where(
                    InterviewSession.interview_config_id == config_id,
                    InterviewSession.student_id == current_user.user_id,
                )
            )
        ).scalar_one()
        if used >= config.max_attempts:
            raise _not_found("interview_config", config_id)

    # Just-in-time reveal: this learner request never materializes the full bank.
    #
    # Pinned to the graded partition. No session exists yet, so there is nothing
    # to read a mode off — and the assessment side is the one that preserves this
    # endpoint's behaviour exactly as it was before the split. A practice run
    # gets its first question from ``start_session``, which does know the mode.
    first_question = (
        await db.execute(
            select(InterviewQuestion)
            .where(
                InterviewQuestion.interview_config_id == config_id,
                InterviewQuestion.review_status == "approved",
            )
            .order_by(InterviewQuestion.position)
            .limit(1)
        )
    ).scalar_one_or_none()
    # Count-only signal: how many rubric criteria this interview assesses. We
    # count rows rather than fetch them so no outcome text / weight / threshold
    # is materialized in the learner path (see InterviewForTakingPublic docs).
    outcome_count = (
        await db.execute(
            select(func.count(InterviewOutcome.id)).where(
                InterviewOutcome.interview_config_id == config_id,
                InterviewOutcome.deleted_at.is_(None),
            )
        )
    ).scalar_one()
    return InterviewForTakingPublic(
        config=InterviewConfigPublic.model_validate(config),
        first_question=(
            InterviewQuestionPublic.model_validate(first_question)
            if first_question is not None
            else None
        ),
        outcome_count=int(outcome_count),
    )


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
    if session.input_mode not in ("voice", "hybrid"):
        raise _conflict("session is not a voice interview")
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
    if session.input_mode not in ("voice", "hybrid"):
        raise _conflict("session is not a voice interview")
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


@router.get("/interview-sessions/{session_id}/gap-report", response_model=GapReportRead)
async def get_gap_report(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GapReportRead:
    del current_user
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import GapReport  # noqa: PLC0415

    stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(stmt)).scalar_one_or_none()
    if report is None:
        raise _not_found("gap_report", session_id)
    return await _gap_report_view(db, report)


@router.get("/me/interview-sessions", response_model=list[InterviewSessionPublic])
async def list_my_sessions(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewSessionPublic]:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    sessions = await taking_service.get_user_sessions(db, current_user.user_id)
    # Batch-fetch the parent config title + course for each distinct config so the
    # student history list is self-describing (the session row only carries the id).
    config_ids = {s.interview_config_id for s in sessions}
    config_meta: dict[UUID, tuple[str, UUID]] = {}
    if config_ids:
        rows = (
            await db.execute(
                select(
                    InterviewConfig.id,
                    InterviewConfig.title,
                    InterviewConfig.course_id,
                ).where(InterviewConfig.id.in_(config_ids))
            )
        ).all()
        config_meta = {row.id: (row.title, row.course_id) for row in rows}
    return [
        InterviewSessionPublic(
            session_id=s.id,
            interview_config_id=s.interview_config_id,
            interview_title=config_meta.get(s.interview_config_id, (None, None))[0],
            course_id=config_meta.get(s.interview_config_id, (None, None))[1],
            status=s.status,
            input_mode=s.input_mode,
            attempt_number=s.attempt_number,
            started_at=s.started_at,
            assessment_started_at=s.assessment_started_at,
            onboarding_stage=s.onboarding_stage,
            interview_language=s.interview_language,
            ended_at=s.ended_at,
            resume_deadline_at=s.resume_deadline_at,
            current_question_index=None,
            time_remaining_seconds=await taking_service.session_time_remaining_seconds(db, s),
            pass_verdict=s.pass_verdict,
        )
        for s in sessions
    ]


def _history_elapsed_seconds(
    assessment_started_at: datetime | None,
    created_at: datetime,
) -> int | None:
    if assessment_started_at is None or created_at < assessment_started_at:
        return None
    return max(0, int((created_at - assessment_started_at).total_seconds()))


def _history_message_kind(message: InterviewSessionMessage) -> str:
    metadata = message.metadata_json if isinstance(message.metadata_json, dict) else {}
    stored_kind = metadata.get("kind")
    if stored_kind in {"clarification", "term_explanation"}:
        return "clarification"
    if stored_kind == "hint":
        return "hint"
    # Standardized between-question transition turns (Natural Interview
    # Transitions spec) map to the existing "transition" history kind so the
    # transcript renders them as their own chronological AI turn.
    if stored_kind == "transition":
        return "transition"
    # Rich-closing sub-steps (Slice 13): self-reflection prompt, invite-candidate-
    # questions, and the answer-safe reply are NON-assessed ceremony that the
    # adaptive path persists attached to the last question. Without this they'd
    # fall through to "followup" below and render under "Question N", so classify
    # them as "closing" — the transcript then groups them in their own wrap-up
    # section, never under a numbered assessed question.
    if metadata.get("action") in {
        "prompt_self_reflection",
        "invite_candidate_questions",
        "answer_candidate_question",
    }:
        return "closing"
    if message.role == "user":
        return "answer"
    ceremony_key = metadata.get("ceremony_key")
    if ceremony_key == "briefing":
        return "briefing"
    if ceremony_key == "ready_transition":
        return "transition"
    if ceremony_key == "closing":
        return "closing"
    if message.session_question_id is not None:
        return "followup"
    return "opening"


def _build_session_history(  # noqa: C901 -- explicit transcript merge is easier to audit
    session: InterviewSession,
    question_rows: list[tuple[InterviewSessionQuestion, InterviewQuestion | None]],
    messages: list[InterviewSessionMessage],
) -> list[InterviewSessionHistoryTurn]:
    """Merge persisted ceremony/messages with every revealed question.

    The ``ready_transition`` handoff is deliberately withheld. Its row still exists
    — ``orchestration_bridge`` reads it as the marker that the REST ceremony already
    introduced the interviewer — but nobody ever SAYS it: the agent owns its own
    opening and speaks the question itself. Replaying it here put a line the
    candidate never heard back on screen every time the page reloaded mid-session.
    """
    history: list[InterviewSessionHistoryTurn] = []
    visible_messages = [
        message
        for message in messages
        if message.role in {"ai", "user"}
        and bool((message.content_text or "").strip())
        and (message.metadata_json or {}).get("ceremony_key") != "ready_transition"
    ]

    def append_message(message: InterviewSessionMessage) -> None:
        kind = _history_message_kind(message)
        history.append(
            InterviewSessionHistoryTurn(
                id=f"message:{message.id}",
                role=cast(Any, message.role),
                content_text=message.content_text,
                kind=cast(Any, kind),
                created_at=message.created_at,
                elapsed_seconds=_history_elapsed_seconds(
                    session.assessment_started_at,
                    message.created_at,
                ),
                is_follow_up=kind == "followup",
            )
        )

    for message in visible_messages:
        if message.session_question_id is None:
            append_message(message)

    messages_by_question: dict[UUID, list[InterviewSessionMessage]] = {}
    for message in visible_messages:
        if message.session_question_id is not None:
            messages_by_question.setdefault(message.session_question_id, []).append(message)

    included_question_ids: set[UUID] = set()
    if session.onboarding_stage == "completed":
        for session_question, question in question_rows:
            included_question_ids.add(session_question.id)
            if question is not None and (question.prompt_text or "").strip():
                question_created_at = session_question.asked_at
                if session_question.sequence_no == 1 and session.assessment_started_at is not None:
                    question_created_at = session.assessment_started_at
                history.append(
                    InterviewSessionHistoryTurn(
                        id=f"question:{session_question.id}",
                        role="ai",
                        content_text=question.prompt_text,
                        kind="question",
                        created_at=question_created_at,
                        elapsed_seconds=_history_elapsed_seconds(
                            session.assessment_started_at,
                            question_created_at,
                        ),
                        question_type=question.question_type,
                    )
                )
            # The agent's spoken question (a paraphrase of the prompt) is the
            # first AI turn for this question, recorded before the candidate's
            # answer. It duplicates the canonical question turn appended above,
            # so drop that ONE turn — but only turns the native agent tagged
            # ``kind == "question"`` that precede the first answer. Follow-ups
            # (also ``kind == "question"``) come AFTER an answer and stay;
            # closing/clarification/hint turns carry distinct kinds and stay.
            # Without this, a resumed voice session showed the question twice
            # (bank wording in the history + the paraphrase pinned as the active
            # card), which read as the two panels being swapped.
            answer_seen = False
            for message in messages_by_question.get(session_question.id, []):
                if message.role == "user":
                    answer_seen = True
                if (
                    not answer_seen
                    and message.role == "ai"
                    and (message.metadata_json or {}).get("kind") == "question"
                ):
                    continue
                append_message(message)

    # Defensive recovery for messages whose question was removed after the
    # interview. The prompt may no longer exist, but the candidate's own turns
    # still belong in their history.
    for question_id, question_messages in messages_by_question.items():
        if question_id in included_question_ids:
            continue
        for message in question_messages:
            append_message(message)

    return history


async def _session_history(
    db: AsyncSession,
    session: InterviewSession,
) -> list[InterviewSessionHistoryTurn]:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )
    from abridgeai.features.interviews.queries import sessions as sessions_queries  # noqa: PLC0415

    rows = (
        await db.execute(
            select(InterviewSessionQuestion, InterviewQuestion)
            .outerjoin(
                InterviewQuestion,
                InterviewQuestion.id == InterviewSessionQuestion.interview_question_id,
            )
            .where(InterviewSessionQuestion.session_id == session.id)
            .order_by(InterviewSessionQuestion.sequence_no)
        )
    ).all()
    question_rows = [(row[0], row[1]) for row in rows]
    messages = await sessions_queries.list_session_messages(db, session.id)
    return _build_session_history(session, question_rows, messages)


async def _current_session_question(db: AsyncSession, session_id: UUID) -> object | None:
    """Return the latest revealed question so an active attempt resumes in place."""
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )

    sq_stmt = (
        select(InterviewSessionQuestion)
        .where(InterviewSessionQuestion.session_id == session_id)
        .order_by(InterviewSessionQuestion.sequence_no.desc())
        .limit(1)
    )
    sq = (await db.execute(sq_stmt)).scalar_one_or_none()
    if sq is None or sq.interview_question_id is None:
        return None
    return await db.get(InterviewQuestion, sq.interview_question_id)


async def _resolve_resource_titles(
    db: AsyncSession,
    resource_ids: set[str],
) -> dict[str, str]:
    """Map resource UUID → human title for study-plan display.

    The gap report persists only resource UUIDs in ``report_json``; resolving
    them to titles at projection time repairs existing reports WITHOUT a data
    migration. IDs that no longer resolve (deleted resource, malformed id) are
    simply omitted so the UI never shows a raw UUID.
    """
    if not resource_ids:
        return {}
    from sqlalchemy import bindparam  # noqa: PLC0415
    from sqlalchemy import text as _text  # noqa: PLC0415

    valid: list[UUID] = []
    for rid in resource_ids:
        try:
            valid.append(UUID(str(rid)))
        except (ValueError, AttributeError, TypeError):
            continue
    if not valid:
        return {}
    stmt = _text(
        "SELECT id, title FROM lesson_resources WHERE id IN :ids AND deleted_at IS NULL"
    ).bindparams(bindparam("ids", expanding=True))
    rows = (await db.execute(stmt, {"ids": valid})).mappings().all()
    return {str(row["id"]): row["title"] for row in rows}


def _study_plan_from_report(report_json: object) -> list[dict[str, Any]]:
    """Extract the raw study-plan entries (UUID resources) from report_json."""
    raw_plan = report_json.get("study_plan") if isinstance(report_json, dict) else None
    study_plan: list[dict[str, Any]] = []
    if isinstance(raw_plan, list):
        for entry in raw_plan:
            if not isinstance(entry, dict):
                continue
            study_plan.append(
                {
                    "topic": entry.get("topic", ""),
                    "lesson_id": entry.get("suggested_lesson_id"),
                    "suggested_resources": [
                        str(rid) for rid in entry.get("suggested_resource_ids", []) or []
                    ],
                }
            )
    return study_plan


def _apply_resource_titles(
    study_plan: list[dict[str, Any]],
    titles: dict[str, str],
) -> None:
    """Replace each item's UUID resource list with resolved human titles.

    Mutates in place. Unresolvable ids are dropped rather than shown raw, so a
    study-plan item may end up with an empty resource list (the UI already
    hides the resource line when empty).
    """
    for item in study_plan:
        item["suggested_resources"] = [
            titles[rid] for rid in item["suggested_resources"] if rid in titles
        ]


async def _gap_report_view(db: AsyncSession, report: Any) -> GapReportRead:  # noqa: ANN401  -- ORM row, typed via duck shape
    report_json = report.report_json or {}
    study_plan = _study_plan_from_report(report_json)
    resource_ids = {rid for item in study_plan for rid in item["suggested_resources"]}
    _apply_resource_titles(study_plan, await _resolve_resource_titles(db, resource_ids))
    # FR-5.7: student sees pass/fail + qualitative remediation only. The
    # per-criterion mean rubric scores (report_json["rubric_aggregated"]) are
    # deliberately NOT projected here — they are teacher-only via
    # GapReportAuthoringRead. See interviews/schemas/report.py.
    summary_text = report.student_summary or ""
    return GapReportRead.model_validate(
        {
            "id": report.id,
            "student_id": report.student_id,
            "course_id": report.course_id,
            "module_id": report.module_id,
            "discrepancy_summary": summary_text or None,
            "study_plan": study_plan,
            "generated_at": report.created_at,
        }
    )


@router.get(
    "/courses/{course_id}/interview-progress",
    response_model=list[InterviewProgressRead],
)
async def list_my_interview_progress(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewProgressRead]:
    """Per-interview completion state for the calling student in a course.

    Feeds the course-learn curriculum, which had no completion signal for
    interview items at all — they stayed pending forever and a module holding
    one could never auto-collapse, even after the student passed.

    Completed ⟺ at least one non-practice attempt has ``pass_verdict = TRUE``.
    Deliberately stricter than the quiz rule (which also completes on
    failed-and-exhausted): the tag reads as *passed*. See
    :class:`InterviewProgressRead` for the field semantics.

    Gated by :func:`can_view_course_content` — the same org/enrollment
    perimeter the quiz-progress endpoint uses — so a cross-tenant caller gets
    404 with no existence leak.
    """
    if not await can_view_course_content(db, user_id=current_user.user_id, course_id=course_id):
        raise _not_found("course", course_id)
    rows = await learner_progress_service.list_my_interview_progress(
        db, course_id=course_id, user_id=current_user.user_id
    )
    return [InterviewProgressRead.model_validate(r) for r in rows]


__all__ = ["get_arq_pool", "router"]
