"""Interviews authoring router (T6.12).

Fourteen endpoints under prefix ``/teacher`` covering interview-config
CRUD, manual question + outcome CRUD, soft-delete, and the ARQ-enqueue
triggers for full + per-question generation. Composes
:mod:`features.interviews.services.authoring` (routers→services
boundary, T0.4 import-linter contract).

Security perimeter (FIX-SEC-1, Reconciliation §A9 + §E4)
--------------------------------------------------------
Every endpoint enforces a course-scoped permission check via the
factories in :mod:`._deps`:

* ``POST /teacher/courses/{course_id}/interview-configs`` →
  :func:`features.access_control.policies.require_course_permission`
  on ``course.update`` (mirrors T3.7 / T5.14).
* Endpoints with a ``config_id`` path parameter →
  :func:`require_interview_authoring_access` (walks
  ``config_id → courses.id``).
* Endpoints with a ``question_id`` path parameter →
  :func:`require_question_authoring_access` (walks
  ``question_id → interview_configs → courses.id``; cross-checks any
  sibling ``config_id`` to prevent existence leaks).
* The teacher gap-report endpoint goes through
  :func:`require_session_owner_access`'s authoring sibling — an
  inline course-scoped check via the session's parent config.

No bare ``Depends(get_current_user)`` appears on any write endpoint
(verified by the source-grep test
``test_no_bare_get_current_user_on_interview_authoring_endpoints``).

Service-layer exceptions are mapped to HTTP errors locally — services
stay HTTP-agnostic.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_course_permission
from abridgeai.features.interviews.routers._deps import (
    require_interview_authoring_access,
    require_question_authoring_access,
    require_session_authoring_access,
)
from abridgeai.features.interviews.schemas import (
    GapReportAuthoringRead,
    GenerationRunStatusLiteral,
    InterviewConfigAuthoring,
    InterviewConfigCreate,
    InterviewConfigUpdate,
    InterviewForAuthoringPublic,
    InterviewGenerationRequest,
    InterviewGenerationRunPublic,
    InterviewOutcomeAuthoring,
    InterviewOutcomeCreate,
    InterviewQuestionAuthoring,
    InterviewQuestionCreate,
    InterviewSessionSummary,
    InterviewTranscriptRead,
    InterviewTranscriptTurn,
)
from abridgeai.features.interviews.services import authoring as authoring_service

router = APIRouter(prefix="/teacher", tags=["interviews-authoring"])

_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_CONFIG = require_interview_authoring_access()
_REQUIRE_QUESTION = require_question_authoring_access()
_REQUIRE_SESSION_AUTHORING = require_session_authoring_access()


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


async def get_arq_pool() -> object | None:
    return None


@router.post(
    "/courses/{course_id}/interview-configs",
    response_model=InterviewConfigAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_interview_config(
    course_id: UUID,
    payload: InterviewConfigCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewConfigAuthoring:
    if payload.course_id != course_id:
        raise _bad_request("course_id mismatch")
    try:
        config = await authoring_service.create_interview_config(
            db, course_id, payload, current_user
        )
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewConfigAuthoring.model_validate(config)


@router.get("/interview-configs/{config_id}", response_model=InterviewForAuthoringPublic)
async def get_interview_config(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewForAuthoringPublic:
    del current_user
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415
    from abridgeai.features.interviews.queries import (  # noqa: PLC0415
        list_outcomes_for_config,
        list_questions_for_config,
    )

    config = await db.get(InterviewConfig, config_id)
    if config is None:
        raise _not_found("interview_config", config_id)

    outcomes = await list_outcomes_for_config(db, config_id)
    questions = await list_questions_for_config(db, config_id)

    return InterviewForAuthoringPublic(
        config=InterviewConfigAuthoring.model_validate(config),
        outcomes=[InterviewOutcomeAuthoring.model_validate(o) for o in outcomes],
        questions=[InterviewQuestionAuthoring.model_validate(q) for q in questions],
    )


@router.patch("/interview-configs/{config_id}", response_model=InterviewConfigAuthoring)
async def update_interview_config(
    config_id: UUID,
    payload: InterviewConfigUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewConfigAuthoring:
    try:
        config = await authoring_service.update_interview_config(
            db, config_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewConfigAuthoring.model_validate(config)


@router.post("/interview-configs/{config_id}/publish", response_model=InterviewConfigAuthoring)
async def publish_interview_config(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewConfigAuthoring:
    try:
        config = await authoring_service.publish_interview_config(db, config_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewConfigAuthoring.model_validate(config)


@router.post("/interview-configs/{config_id}/archive", response_model=InterviewConfigAuthoring)
async def archive_interview_config(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewConfigAuthoring:
    try:
        config = await authoring_service.archive_interview_config(db, config_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewConfigAuthoring.model_validate(config)


@router.post("/interview-configs/{config_id}/unarchive", response_model=InterviewConfigAuthoring)
async def unarchive_interview_config(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewConfigAuthoring:
    try:
        config = await authoring_service.unarchive_interview_config(db, config_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewConfigAuthoring.model_validate(config)


@router.delete(
    "/interview-configs/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_interview_config(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await authoring_service.delete_interview_config(db, config_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    await db.commit()


@router.post(
    "/interview-configs/{config_id}/generate",
    response_model=InterviewGenerationRunPublic,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_generation(
    config_id: UUID,
    payload: InterviewGenerationRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> InterviewGenerationRunPublic:
    try:
        run = await authoring_service.start_generation_run(
            db, config_id, payload, current_user, arq_pool=arq_pool
        )
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return _generation_run_view(run)


@router.get(
    "/interview-configs/{config_id}/generation-runs/{run_id}",
    response_model=InterviewGenerationRunPublic,
)
async def get_generation_run(
    config_id: UUID,
    run_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewGenerationRunPublic:
    del current_user
    run = await authoring_service.get_generation_run(db, config_id, run_id)
    if run is None:
        raise _not_found("generation_run", run_id)
    return _generation_run_view(run)


@router.post(
    "/interview-configs/{config_id}/questions",
    response_model=InterviewQuestionAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_question(
    config_id: UUID,
    payload: InterviewQuestionCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionAuthoring:
    try:
        question = await authoring_service.add_question(db, config_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewQuestionAuthoring.model_validate(question)


@router.patch(
    "/interview-configs/{config_id}/questions/{question_id}",
    response_model=InterviewQuestionAuthoring,
)
async def update_question(
    config_id: UUID,
    question_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionAuthoring:
    try:
        question = await authoring_service.update_question(
            db, config_id, question_id, _AttrShim(payload), current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question", question_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewQuestionAuthoring.model_validate(question)


@router.delete(
    "/interview-configs/{config_id}/questions/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_question(
    config_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await authoring_service.delete_question(db, config_id, question_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_question", question_id) from exc
    await db.commit()


@router.post(
    "/interview-configs/{config_id}/questions/{question_id}/regenerate",
    response_model=InterviewGenerationRunPublic,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_question(
    config_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> InterviewGenerationRunPublic:
    try:
        run = await authoring_service.regenerate_question(
            db, config_id, question_id, current_user, arq_pool=arq_pool
        )
    except NotFoundError as exc:
        raise _not_found("interview_question", question_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return _generation_run_view(run)


@router.post(
    "/interview-configs/{config_id}/outcomes",
    response_model=InterviewOutcomeAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_outcome(
    config_id: UUID,
    payload: InterviewOutcomeCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewOutcomeAuthoring:
    try:
        outcome = await authoring_service.add_outcome(db, config_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewOutcomeAuthoring.model_validate(outcome)


@router.patch(
    "/interview-configs/{config_id}/outcomes/{outcome_id}",
    response_model=InterviewOutcomeAuthoring,
)
async def update_outcome(
    config_id: UUID,
    outcome_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewOutcomeAuthoring:
    try:
        outcome = await authoring_service.update_outcome(
            db, config_id, outcome_id, _AttrShim(payload), current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_outcome", outcome_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return InterviewOutcomeAuthoring.model_validate(outcome)


@router.delete(
    "/interview-configs/{config_id}/outcomes/{outcome_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_outcome(
    config_id: UUID,
    outcome_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await authoring_service.delete_outcome(db, config_id, outcome_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_outcome", outcome_id) from exc
    await db.commit()


@router.get(
    "/interview-configs/{config_id}/sessions",
    response_model=list[InterviewSessionSummary],
)
async def list_config_sessions(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewSessionSummary]:
    """Teacher's per-config attempts list (thesis p77 review surface)."""
    del current_user
    from sqlalchemy import text as _text  # noqa: PLC0415

    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    sessions = await _sessions_q.list_sessions_for_config(db, config_id)
    student_ids = {s.student_id for s in sessions}
    names: dict[UUID, str] = {}
    if student_ids:
        rows = (
            (
                await db.execute(
                    _text(
                        "SELECT u.id, COALESCE(p.display_name, u.primary_email) AS name "
                        "FROM users u "
                        "LEFT JOIN user_profiles p ON p.user_id = u.id "
                        "WHERE u.id = ANY(:ids)"
                    ),
                    {"ids": list(student_ids)},
                )
            )
            .mappings()
            .all()
        )
        names = {row["id"]: row["name"] for row in rows}
    return [
        InterviewSessionSummary(
            session_id=s.id,
            student_id=s.student_id,
            student_name=names.get(s.student_id),
            attempt_number=s.attempt_number,
            status=s.status,
            input_mode=s.input_mode,
            pass_verdict=s.pass_verdict,
            started_at=s.started_at,
            ended_at=s.ended_at,
        )
        for s in sessions
    ]


@router.get(
    "/interview-sessions/{session_id}/transcript",
    response_model=InterviewTranscriptRead,
)
async def get_session_transcript(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewTranscriptRead:
    """Full ordered Q&A transcript for teacher remediation review (thesis p77)."""
    del current_user
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    messages = await _sessions_q.list_session_messages(db, session_id)
    # Resolve each message's question prompt via session_question_id -> question.
    prompts: dict[UUID, str] = {}
    sq_ids = {m.session_question_id for m in messages if m.session_question_id is not None}
    if sq_ids:
        from sqlalchemy import select as _select  # noqa: PLC0415

        rows = (
            await db.execute(
                _select(InterviewSessionQuestion.id, InterviewQuestion.prompt_text)
                .join(
                    InterviewQuestion,
                    InterviewQuestion.id == InterviewSessionQuestion.interview_question_id,
                )
                .where(InterviewSessionQuestion.id.in_(sq_ids))
            )
        ).all()
        prompts = {row[0]: row[1] for row in rows}

    turns = [
        InterviewTranscriptTurn(
            role=m.role,
            question_prompt=prompts.get(m.session_question_id)
            if m.session_question_id is not None
            else None,
            content_text=m.content_text,
            has_audio=m.audio_object_id is not None,
            created_at=m.created_at,
        )
        for m in messages
    ]
    return InterviewTranscriptRead(session_id=session_id, turns=turns)


@router.get(
    "/interview-sessions/{session_id}/gap-report",
    response_model=GapReportAuthoringRead,
)
async def get_session_gap_report_authoring(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GapReportAuthoringRead:
    del current_user
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        GapReport,
        InterviewOutcomeEvaluation,
    )

    report_stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(report_stmt)).scalar_one_or_none()
    if report is None:
        raise _not_found("gap_report", session_id)

    eval_stmt = select(InterviewOutcomeEvaluation).where(
        InterviewOutcomeEvaluation.session_id == session_id
    )
    evaluations = list((await db.execute(eval_stmt)).scalars().all())
    raw_evaluation_json: dict[str, Any] = {
        "outcome_evaluations": [
            {
                "id": str(e.id),
                "outcome_id": str(e.outcome_id),
                "verdict_met": e.verdict_met,
                "hidden_reasoning": e.hidden_reasoning,
                "evidence_excerpt": e.evidence_excerpt,
            }
            for e in evaluations
        ]
    }

    # FR-5.7: per-criterion mean rubric scores are teacher-only. They live
    # in ``report_json["rubric_aggregated"]`` and are surfaced here (never on
    # the student-facing GapReportRead).
    report_json = report.report_json or {}
    per_criterion = report_json.get("rubric_aggregated") if isinstance(report_json, dict) else None
    base = GapReportAuthoringRead.model_validate(report)
    return base.model_copy(
        update={
            "raw_evaluation_json": raw_evaluation_json,
            "per_criterion_breakdown": (per_criterion if isinstance(per_criterion, dict) else {}),
        }
    )


def _generation_run_view(run: Any) -> InterviewGenerationRunPublic:  # noqa: ANN401  -- ORM row, typed via duck shape
    config_json = run.config_json or {}
    failure = config_json.get("failure") if isinstance(config_json, dict) else None
    failure_message = (
        str(failure.get("message")) if isinstance(failure, dict) and "message" in failure else None
    )
    raw_status = run.status
    run_status: GenerationRunStatusLiteral
    if raw_status in ("pending", "running", "completed", "failed", "cancelled"):
        run_status = raw_status
    else:
        run_status = "pending"
    return InterviewGenerationRunPublic(
        run_id=run.id,
        status=run_status,
        config_json=dict(config_json) if isinstance(config_json, dict) else {},
        started_at=run.started_at or run.created_at,
        finished_at=run.finished_at,
        failure_message=failure_message,
    )


class _AttrShim:
    """Adapt a ``dict`` body into the ``model_dump`` / attr-access shape services expect."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def model_dump(self, exclude_unset: bool = False, mode: str | None = None) -> dict[str, Any]:
        del exclude_unset, mode
        return dict(self._data)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401  -- shim returns whatever the dict holds
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            value = self._data[name]
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [_AttrShim(item) for item in value]
            if isinstance(value, dict):
                return _AttrShim(value)
            return value
        return None


__all__ = [
    "get_arq_pool",
    "router",
]
