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
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.interviews.routers._deps import require_session_owner_access
from abridgeai.features.interviews.schemas import (
    GapReportRead,
    InterviewConfigPublic,
    InterviewForTakingPublic,
    InterviewQuestionPublic,
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
from abridgeai.features.interviews.services import real_time as realtime_service
from abridgeai.features.interviews.services import taking as taking_service

router = APIRouter(tags=["interviews-learner"])

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
        InterviewQuestion,
        InterviewSession,
    )

    config = await db.get(InterviewConfig, config_id)
    if config is None or config.status != "published":
        raise _not_found("interview_config", config_id)
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
    return InterviewForTakingPublic(
        config=InterviewConfigPublic.model_validate(config),
        first_question=(
            InterviewQuestionPublic.model_validate(first_question)
            if first_question is not None
            else None
        ),
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
) -> InterviewSessionStartResponse:
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None or config.status != "published":
        raise _not_found("interview_config", config_id)
    try:
        session = await taking_service.start_session(db, config_id, payload, current_user)
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
    first_question = await _first_session_question(db, session.id)
    return InterviewSessionStartResponse(
        session_id=session.id,
        first_question=(
            InterviewQuestionPublic.model_validate(first_question)
            if first_question is not None
            else None
        ),
        time_remaining_seconds=None,
        question_count_remaining=None,
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
) -> RealtimeTokenResponse:
    """Mint a LiveKit join token (+ agent dispatch) for a voice session.

    Ownership + existence are enforced by ``_REQUIRE_SESSION_OWNER``. Here we
    additionally gate on the voice feature flag and the session's
    mode/status, persist the room name on first call (idempotent), and return
    a short-lived participant token. Phase 3's agent worker is dispatched by
    the token's room-config when the room is first created.
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
            language=realtime_service.normalize_language(accept_language),
        )
    except ValueError as exc:  # credentials missing despite the flag
        raise _voice_unavailable() from exc


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
    narration_language = realtime_service.normalize_language(accept_language)
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
    config = await db.get(InterviewConfig, session.interview_config_id)
    if config is not None:
        persona = config.persona

    settings = get_settings()
    try:
        audio = await narration_service.synthesize_speech(
            guarded_narration.text, persona=persona, settings=settings
        )
    except narration_service.NarrationUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "narration_unavailable", "message": str(exc)},
        ) from exc

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
    return InterviewSessionPublic(
        session_id=session.id,
        interview_config_id=session.interview_config_id,
        status=session.status,
        input_mode=session.input_mode,
        attempt_number=session.attempt_number,
        started_at=session.started_at,
        ended_at=session.ended_at,
        resume_deadline_at=session.resume_deadline_at,
        current_question_index=None,
        time_remaining_seconds=None,
        pass_verdict=session.pass_verdict,
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
    next_question = result.get("next_question")
    return InterviewSubmitAnswerResponse(
        # ── legacy fields (always present; unchanged for existing clients) ───
        next_question=(
            InterviewQuestionPublic.model_validate(next_question)
            if next_question is not None
            else None
        ),
        is_finished=bool(result.get("is_finished")),
        ai_followup_text=result.get("followup_text"),
        time_remaining_seconds=None,
        # ── adaptive structured fields (None on the legacy/sequential path) ──
        ai_turn_text=result.get("ai_turn_text"),
        language=result.get("language"),
        should_narrate=result.get("should_narrate"),
        should_await_response=result.get("should_await_response"),
        should_finish=result.get("should_finish"),
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
) -> InterviewSessionFinishResponse:
    try:
        session = await taking_service.submit_session(
            db, session_id, current_user, arq_pool=arq_pool
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
    return InterviewSessionFinishResponse(
        session_id=session.id,
        status=session.status,
        total_score=None,
        rubric_scores=[],
        pass_verdict=session.pass_verdict,
        ended_at=session.ended_at,
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
    return _gap_report_view(report)


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
            ended_at=s.ended_at,
            resume_deadline_at=s.resume_deadline_at,
            current_question_index=None,
            time_remaining_seconds=None,
            pass_verdict=s.pass_verdict,
        )
        for s in sessions
    ]


async def _first_session_question(db: AsyncSession, session_id: UUID) -> object | None:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )

    sq_stmt = (
        select(InterviewSessionQuestion)
        .where(InterviewSessionQuestion.session_id == session_id)
        .order_by(InterviewSessionQuestion.sequence_no)
        .limit(1)
    )
    sq = (await db.execute(sq_stmt)).scalar_one_or_none()
    if sq is None or sq.interview_question_id is None:
        return None
    return await db.get(InterviewQuestion, sq.interview_question_id)


def _gap_report_view(report: Any) -> GapReportRead:  # noqa: ANN401  -- ORM row, typed via duck shape
    report_json = report.report_json or {}
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


__all__ = ["get_arq_pool", "router"]
