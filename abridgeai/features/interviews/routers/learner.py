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

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.interviews.routers._deps import require_session_owner_access
from abridgeai.features.interviews.schemas import (
    GapReportRead,
    InterviewConfigPublic,
    InterviewForTakingPublic,
    InterviewOutcomePublic,
    InterviewQuestionPublic,
    InterviewRubricScore,
    InterviewSessionFinishResponse,
    InterviewSessionPublic,
    InterviewSessionStartRequest,
    InterviewSessionStartResponse,
    InterviewSubmitAnswerRequest,
    InterviewSubmitAnswerResponse,
)
from abridgeai.features.interviews.services import taking as taking_service

router = APIRouter(tags=["interviews-learner"])

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

    outcomes = list(
        (
            await db.execute(
                select(InterviewOutcome)
                .where(InterviewOutcome.interview_config_id == config_id)
                .order_by(InterviewOutcome.position)
            )
        )
        .scalars()
        .all()
    )
    questions = list(
        (
            await db.execute(
                select(InterviewQuestion)
                .where(
                    InterviewQuestion.interview_config_id == config_id,
                    InterviewQuestion.review_status == "approved",
                )
                .order_by(InterviewQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    first_question = questions[0] if questions else None
    return InterviewForTakingPublic(
        config=InterviewConfigPublic.model_validate(config),
        outcomes=[InterviewOutcomePublic.model_validate(o) for o in outcomes],
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


@router.post(
    "/interview-sessions/{session_id}/respond",
    response_model=InterviewSubmitAnswerResponse,
)
async def respond_to_session(
    session_id: UUID,
    payload: InterviewSubmitAnswerRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
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
    await db.commit()
    next_question = result.get("next_question")
    return InterviewSubmitAnswerResponse(
        next_question=(
            InterviewQuestionPublic.model_validate(next_question)
            if next_question is not None
            else None
        ),
        is_finished=bool(result.get("is_finished")),
        ai_followup_text=result.get("followup_text"),
        time_remaining_seconds=None,
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
    rubric_scores = _rubric_scores_from_session(session)
    total = (session.internal_summary_json or {}).get("total_score")
    return InterviewSessionFinishResponse(
        session_id=session.id,
        status=session.status,
        total_score=_decimal_or_none(total),
        rubric_scores=rubric_scores,
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
    sessions = await taking_service.get_user_sessions(db, current_user.user_id)
    return [
        InterviewSessionPublic(
            session_id=s.id,
            interview_config_id=s.interview_config_id,
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


def _decimal_or_none(value: object) -> object | None:
    if value is None:
        return None
    from decimal import Decimal  # noqa: PLC0415

    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (ValueError, ArithmeticError):
        return None


def _rubric_scores_from_session(session: object) -> list[InterviewRubricScore]:
    del session
    return []


def _gap_report_view(report: object) -> GapReportRead:
    return GapReportRead.model_validate(report)


__all__ = ["get_arq_pool", "router"]
