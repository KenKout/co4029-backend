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
    require_outcome_authoring_access,
    require_question_authoring_access,
    require_session_authoring_access,
)
from abridgeai.features.interviews.schemas import (
    AdaptiveReadinessRead,
    GapReportAuthoringRead,
    GapReportNotesUpdate,
    GenerationRunStatusLiteral,
    InterviewConfigAuthoring,
    InterviewConfigCreate,
    InterviewConfigUpdate,
    InterviewForAuthoringPublic,
    InterviewGenerationRequest,
    InterviewGenerationRunPublic,
    InterviewIntegrityEvent,
    InterviewIntegrityRead,
    InterviewOutcomeAuthoring,
    InterviewOutcomeCreate,
    InterviewOutcomeUpdate,
    InterviewQuestionAuthoring,
    InterviewQuestionBankItemCreate,
    InterviewQuestionBankItemRead,
    InterviewQuestionBankItemUpdate,
    InterviewQuestionCreate,
    InterviewQuestionDuplicateCheck,
    InterviewQuestionDuplicateCheckRequest,
    InterviewQuestionUpdate,
    InterviewSessionPublic,
    InterviewSessionSummary,
    InterviewSessionTeacherRead,
    InterviewTranscriptRead,
    InterviewTranscriptTurn,
    SecuritySessionSummary,
)
from abridgeai.features.interviews.services import authoring as authoring_service

router = APIRouter(prefix="/teacher", tags=["interviews-authoring"])

_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_CONFIG = require_interview_authoring_access()
_REQUIRE_OUTCOME = require_outcome_authoring_access()
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


# --------------------------------------------------------------------------- #
# Course-scoped interview question bank (§QBank-1). Course-update permission
# gates all three. Import into a config reuses the existing create-question
# endpoint client-side (copy semantics), so no dedicated import route here.
# --------------------------------------------------------------------------- #
@router.get(
    "/courses/{course_id}/interview-question-bank",
    response_model=list[InterviewQuestionBankItemRead],
)
async def list_interview_question_bank(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewQuestionBankItemRead]:
    """List the course's reusable interview questions, newest first."""
    del current_user
    try:
        items = await authoring_service.list_question_bank(db, course_id)
    except NotFoundError as exc:
        raise _not_found("course", course_id) from exc
    return [InterviewQuestionBankItemRead.model_validate(i) for i in items]


@router.post(
    "/courses/{course_id}/interview-question-bank",
    response_model=InterviewQuestionBankItemRead,
    status_code=status.HTTP_201_CREATED,
)
async def add_interview_question_bank_item(
    course_id: UUID,
    payload: InterviewQuestionBankItemCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionBankItemRead:
    """Add a reusable question to the course bank (copy semantics)."""
    try:
        item = await authoring_service.add_to_question_bank(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found("course", course_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return InterviewQuestionBankItemRead.model_validate(item)


@router.patch(
    "/courses/{course_id}/interview-question-bank/{item_id}",
    response_model=InterviewQuestionBankItemRead,
)
async def update_interview_question_bank_item(
    course_id: UUID,
    item_id: UUID,
    payload: InterviewQuestionBankItemUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionBankItemRead:
    """Edit a bank item (management page). Only supplied fields change."""
    try:
        item = await authoring_service.update_question_bank_item(
            db, course_id, item_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question_bank_item", item_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return InterviewQuestionBankItemRead.model_validate(item)


@router.delete(
    "/courses/{course_id}/interview-question-bank/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_interview_question_bank_item(
    course_id: UUID,
    item_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a bank item; already-imported questions are untouched."""
    try:
        await authoring_service.delete_question_bank_item(db, course_id, item_id, current_user)
    except NotFoundError as exc:
        raise _not_found("interview_question_bank_item", item_id) from exc
    await db.commit()


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


def _session_teacher_view(
    session: Any,  # noqa: ANN401  -- ORM row
    config_title: str,
    student_name: str | None,
    security_summary: SecuritySessionSummary | None = None,
) -> InterviewSessionTeacherRead:
    return InterviewSessionTeacherRead(
        session_id=session.id,
        interview_config_id=session.interview_config_id,
        interview_config_title=config_title,
        student_id=session.student_id,
        student_name=student_name,
        attempt_number=session.attempt_number,
        status=session.status,
        input_mode=session.input_mode,
        pass_verdict=session.pass_verdict,
        started_at=session.started_at,
        assessment_started_at=session.assessment_started_at,
        onboarding_stage=session.onboarding_stage,
        interview_language=session.interview_language,
        ended_at=session.ended_at,
        security_summary=security_summary,
    )


async def _security_summary_view(
    db: AsyncSession,
    session: Any,  # noqa: ANN401 -- ORM row
    *,
    enabled: bool,
) -> SecuritySessionSummary | None:
    if not enabled:
        return None
    from abridgeai.features.interviews.services import security as _security  # noqa: PLC0415

    metrics = await _security.get_security_session_metrics(db, session.id)
    return SecuritySessionSummary(
        assessment_count=metrics.assessment_count,
        blocked_attempt_count=metrics.blocked_attempt_count,
        repeated_attempt_count=metrics.repeated_attempt_count,
        output_leakage_prevented=metrics.output_leakage_prevented,
        security_fallback_rate=metrics.security_fallback_rate,
        average_classification_latency_ms=metrics.average_classification_latency_ms,
        session_flagged=bool(session.session_security_flagged),
    )


@router.get(
    "/courses/{course_id}/interview-sessions",
    response_model=list[InterviewSessionTeacherRead],
)
async def list_course_interview_sessions(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewSessionTeacherRead]:
    """Every interview session (any student, any config) in this course.

    Powers the teacher's course-wide "Assessments" tab.
    """
    del current_user
    from sqlalchemy import text as _text  # noqa: PLC0415

    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    rows = await _sessions_q.list_sessions_for_course(db, course_id)
    student_ids = {row.InterviewSession.student_id for row in rows}
    names: dict[UUID, str] = {}
    if student_ids:
        name_rows = (
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
        names = {row["id"]: row["name"] for row in name_rows}
    result: list[InterviewSessionTeacherRead] = []
    for row in rows:
        summary = await _security_summary_view(
            db,
            row.InterviewSession,
            enabled=bool(row.security_incident_summary_enabled),
        )
        result.append(
            _session_teacher_view(
                row.InterviewSession,
                row.title,
                names.get(row.InterviewSession.student_id),
                summary,
            )
        )
    return result


@router.get(
    "/courses/{course_id}/students/{student_id}/interview-sessions",
    response_model=list[InterviewSessionTeacherRead],
)
async def list_student_interview_sessions(
    course_id: UUID,
    student_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewSessionTeacherRead]:
    """Every interview session by one student across this course's configs.

    Powers the teacher's per-student profile page.
    """
    del current_user
    from sqlalchemy import text as _text  # noqa: PLC0415

    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    rows = await _sessions_q.list_sessions_for_student_in_course(db, course_id, student_id)
    name_row = (
        (
            await db.execute(
                _text(
                    "SELECT COALESCE(p.display_name, u.primary_email) AS name "
                    "FROM users u "
                    "LEFT JOIN user_profiles p ON p.user_id = u.id "
                    "WHERE u.id = :id"
                ),
                {"id": student_id},
            )
        )
        .mappings()
        .first()
    )
    student_name = name_row["name"] if name_row else None
    result: list[InterviewSessionTeacherRead] = []
    for row in rows:
        summary = await _security_summary_view(
            db,
            row.InterviewSession,
            enabled=bool(row.security_incident_summary_enabled),
        )
        result.append(_session_teacher_view(row.InterviewSession, row.title, student_name, summary))
    return result


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


@router.get(
    "/interview-configs/{config_id}/adaptive-readiness",
    response_model=AdaptiveReadinessRead,
)
async def get_adaptive_readiness(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdaptiveReadinessRead:
    """Advisory adaptive-readiness report for the authoring workspace (Slice 5).

    Read-only; never blocks publishing. Warnings guide the teacher on whether
    the adaptive interviewer has enough structured material (outcome links,
    difficulty labels, coverage) to adapt well.
    """
    del current_user
    try:
        report = await authoring_service.adaptive_readiness(db, config_id)
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    return AdaptiveReadinessRead.model_validate(report)


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


@router.post("/interview-configs/{config_id}/unpublish", response_model=InterviewConfigAuthoring)
async def unpublish_interview_config(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewConfigAuthoring:
    try:
        config = await authoring_service.unpublish_interview_config(db, config_id, current_user)
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


@router.post(
    "/interview-configs/{config_id}/questions/check-duplicate",
    response_model=InterviewQuestionDuplicateCheck,
)
async def check_question_duplicate(
    config_id: UUID,
    payload: InterviewQuestionDuplicateCheckRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionDuplicateCheck:
    """Advisory check: does this question already exist in the bank?

    Read-only and non-blocking — the teacher can save either way. Declared BEFORE
    the ``{question_id}`` routes below so the literal path segment is not captured
    as a UUID path parameter.
    """
    del current_user  # authorisation handled by the dependency
    try:
        result = await authoring_service.check_question_duplicate(
            db,
            config_id,
            prompt_text=payload.prompt_text,
            exclude_question_id=payload.exclude_question_id,
        )
    except NotFoundError as exc:
        raise _not_found("interview_config", config_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return InterviewQuestionDuplicateCheck.model_validate(result)


@router.patch(
    "/interview-configs/{config_id}/questions/{question_id}",
    response_model=InterviewQuestionAuthoring,
)
async def update_question(
    config_id: UUID,
    question_id: UUID,
    payload: InterviewQuestionUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionAuthoring:
    try:
        question = await authoring_service.update_question(
            db, config_id, question_id, payload, current_user
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    payload: InterviewOutcomeUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_OUTCOME)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewOutcomeAuthoring:
    try:
        outcome = await authoring_service.update_outcome(
            db, config_id, outcome_id, payload, current_user
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
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_OUTCOME)],
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

    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    sessions = await _sessions_q.list_sessions_for_config(db, config_id)
    config = await db.get(InterviewConfig, config_id)
    summaries = {
        s.id: await _security_summary_view(
            db,
            s,
            enabled=bool(config and config.security_incident_summary_enabled),
        )
        for s in sessions
    }
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
            security_summary=summaries[s.id],
        )
        for s in sessions
    ]


@router.get(
    "/interview-sessions/{session_id}",
    response_model=InterviewSessionPublic,
)
async def get_session_authoring(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewSessionPublic:
    """Teacher-scoped session detail (course-owner access via
    ``require_session_authoring_access``).

    Mirrors the learner-side ``GET /interview-sessions/{id}`` (student-
    owner-only), which teachers cannot call. The frontend gap-report page
    was hitting that student endpoint and getting a 403 on every load.
    """
    del current_user
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    session = await _sessions_q.get_session(db, session_id)
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
    "/interview-sessions/{session_id}/integrity-events",
    response_model=InterviewIntegrityRead,
)
async def get_session_integrity_events(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewIntegrityRead:
    """FR-5.8 proctoring timeline for teacher post-session integrity review.

    Returns the session's ``assessment_integrity_events`` (focus_lost /
    tab_switch / fullscreen_exit / reconnect / disconnect), oldest first.
    Teacher-only (course-scoped authoring access); never exposed to students.
    """
    del current_user
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    rows = await _sessions_q.list_integrity_events_for_session(db, session_id)
    return InterviewIntegrityRead(
        session_id=session_id,
        events=[InterviewIntegrityEvent.model_validate(ev) for ev in rows],
    )


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

    # ``GapReport`` (the ORM row) has neither ``generated_at`` (it's
    # ``created_at`` via TimestampMixin) nor ``study_plan``/
    # ``per_criterion_breakdown`` (those live inside ``report_json``
    # JSONB) — model_validate(report) directly would 500 on the
    # required ``generated_at`` field and silently default study_plan
    # to []. Build the DTO explicitly instead, mirroring the student
    # projection (_gap_report_view) plus the teacher-only fields.
    from abridgeai.features.interviews.routers.learner import (  # noqa: PLC0415
        _apply_resource_titles,
        _resolve_resource_titles,
        _study_plan_from_report,
    )

    report_json = report.report_json or {}
    study_plan = _study_plan_from_report(report_json)
    # Resolve resource UUIDs → human titles so the teacher study plan shows real
    # resource names instead of a wall of hex (mirrors the student projection).
    resource_ids = {rid for item in study_plan for rid in item["suggested_resources"]}
    _apply_resource_titles(study_plan, await _resolve_resource_titles(db, resource_ids))
    # FR-5.7: per-criterion mean rubric scores are teacher-only. They live
    # in ``report_json["rubric_aggregated"]`` and are surfaced here (never on
    # the student-facing GapReportRead).
    per_criterion = report_json.get("rubric_aggregated") if isinstance(report_json, dict) else None

    # Resolve human-readable context so the teacher view isn't a wall of UUIDs:
    # the student's display name (falling back to email) and the interview
    # config title. Both are read-only projections.
    from sqlalchemy import text as _text  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )

    name_row = (
        (
            await db.execute(
                _text(
                    "SELECT COALESCE(p.display_name, u.primary_email) AS name "
                    "FROM users u "
                    "LEFT JOIN user_profiles p ON p.user_id = u.id "
                    "WHERE u.id = :id"
                ),
                {"id": report.student_id},
            )
        )
        .mappings()
        .first()
    )
    student_name = name_row["name"] if name_row else None

    interview_title: str | None = None
    score_summary: dict[str, Any] = {}
    rubric_weights: dict[str, float] = {}
    persona_adherence: dict[str, Any] = {}
    session_row = await db.get(InterviewSession, session_id)
    if session_row is not None:
        config_row = await db.get(InterviewConfig, session_row.interview_config_id)
        interview_title = config_row.title if config_row is not None else None
        # Quantitative rollup lives in internal_summary_json (teacher-only). Project
        # the numbers that contextualize the per-criterion means: weighted total,
        # outcomes met/total, answered/total/unanswered question counts.
        summary_json = session_row.internal_summary_json or {}
        if isinstance(summary_json, dict):
            score_summary = {
                key: summary_json[key]
                for key in (
                    "total_score",
                    "outcomes_met",
                    "outcomes_total",
                    "questions_total",
                    "questions_answered",
                    "questions_unanswered",
                )
                if key in summary_json
            }
            # Tone-only persona-adherence audit (teacher-only). Absent for
            # sessions evaluated before this shipped or never audited.
            audit = summary_json.get("persona_adherence")
            if isinstance(audit, dict):
                persona_adherence = audit
        # Resolve the per-criterion rubric weights so the teacher sees each
        # criterion's contribution to the weighted total.
        if config_row is not None:
            from abridgeai.features.interviews.ai.stages.evaluation.rubric import (  # noqa: PLC0415
                resolve_rubric_definition,
            )

            # Read the SAME source the grading stage reads
            # (``supplementary_instructions``), so the weights a teacher sees
            # here are the weights their session was actually graded with.
            # This previously probed a non-existent ``config_json`` attribute,
            # which always resolved to None and therefore always displayed the
            # default equal weights regardless of the configured rubric.
            rubric_weights = resolve_rubric_definition(
                config_row.supplementary_instructions
            ).weights

    # Qualitative per-criterion notes (criterion-tagged bullet phrases) already
    # live in report_json; surface them so the teacher sees the "why" per criterion.
    strengths = report_json.get("strengths") if isinstance(report_json, dict) else None
    weaknesses = report_json.get("weaknesses") if isinstance(report_json, dict) else None

    return GapReportAuthoringRead.model_validate(
        {
            "id": report.id,
            "student_id": report.student_id,
            "course_id": report.course_id,
            "module_id": report.module_id,
            "discrepancy_summary": report.student_summary or None,
            "study_plan": study_plan,
            "generated_at": report.created_at,
            "per_criterion_breakdown": (per_criterion if isinstance(per_criterion, dict) else {}),
            "strengths": [str(s) for s in strengths] if isinstance(strengths, list) else [],
            "weaknesses": [str(w) for w in weaknesses] if isinstance(weaknesses, list) else [],
            "score_summary": score_summary,
            "rubric_weights": rubric_weights,
            "persona_adherence": persona_adherence,
            "raw_evaluation_json": raw_evaluation_json,
            "teacher_summary": report.teacher_summary,
            "source_quiz_attempt_id": report.source_quiz_attempt_id,
            "source_interview_session_id": report.source_interview_session_id,
            "student_name": student_name,
            "interview_title": interview_title,
        }
    )


@router.patch(
    "/interview-sessions/{session_id}/gap-report/notes",
    response_model=GapReportAuthoringRead,
)
async def update_session_gap_report_notes(
    session_id: UUID,
    payload: GapReportNotesUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GapReportAuthoringRead:
    """Persist the teacher-authored note (``teacher_summary``) on the report.

    Course-scoped teacher access only (same gate as the GET). Empty/blank input
    clears the note. Returns the full refreshed authoring projection so the
    client re-renders with the saved value.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import GapReport  # noqa: PLC0415

    report_stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(report_stmt)).scalar_one_or_none()
    if report is None:
        raise _not_found("gap_report", session_id)

    cleaned = (payload.teacher_summary or "").strip()
    report.teacher_summary = cleaned or None
    await db.commit()

    return await get_session_gap_report_authoring(
        session_id=session_id, current_user=current_user, db=db
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


__all__ = [
    "get_arq_pool",
    "router",
]
