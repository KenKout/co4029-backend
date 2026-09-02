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
from abridgeai.features.interviews.routers.authoring_sessions import (
    _security_summary_view,
    _session_teacher_view,
)
from abridgeai.features.interviews.schemas import (
    AdaptiveReadinessRead,
    GenerationRunStatusLiteral,
    InterviewConfigAuthoring,
    InterviewConfigCreate,
    InterviewConfigUpdate,
    InterviewForAuthoringPublic,
    InterviewGenerationRequest,
    InterviewGenerationRunPublic,
    InterviewOutcomeAuthoring,
    InterviewOutcomeCreate,
    InterviewOutcomeUpdate,
    InterviewQuestionAuthoring,
    InterviewQuestionBankImportRequest,
    InterviewQuestionBankImportResult,
    InterviewQuestionBankItemCreate,
    InterviewQuestionBankItemRead,
    InterviewQuestionBankItemUpdate,
    InterviewQuestionBankLogicalGroupCreate,
    InterviewQuestionBankSiblingCreate,
    InterviewQuestionCreate,
    InterviewQuestionDuplicateCheck,
    InterviewQuestionDuplicateCheckRequest,
    InterviewQuestionUpdate,
    InterviewSessionTeacherRead,
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


@router.post(
    "/courses/{course_id}/interview-question-bank/logical-groups",
    response_model=list[InterviewQuestionBankItemRead],
    status_code=status.HTTP_201_CREATED,
)
async def create_interview_question_bank_logical_group(
    course_id: UUID,
    payload: InterviewQuestionBankLogicalGroupCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewQuestionBankItemRead]:
    try:
        items = await authoring_service.create_question_bank_logical_group(
            db, course_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found("course", course_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return [InterviewQuestionBankItemRead.model_validate(item) for item in items]


@router.post(
    "/courses/{course_id}/interview-question-bank/{item_id}/siblings",
    response_model=list[InterviewQuestionBankItemRead],
    status_code=status.HTTP_201_CREATED,
)
async def add_interview_question_bank_sibling(
    course_id: UUID,
    item_id: UUID,
    payload: InterviewQuestionBankSiblingCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewQuestionBankItemRead]:
    try:
        items = await authoring_service.add_question_bank_sibling(
            db, course_id, item_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question_bank_item", item_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return [InterviewQuestionBankItemRead.model_validate(item) for item in items]


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
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
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


@router.delete(
    "/courses/{course_id}/interview-question-bank/{item_id}/group",
    response_model=dict[str, int],
)
async def delete_interview_question_bank_group(
    course_id: UUID,
    item_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, int]:
    """Soft-delete every angle of one logical question in the course bank."""
    try:
        deleted = await authoring_service.delete_question_bank_group(
            db, course_id, item_id, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question_bank_item", item_id) from exc
    await db.commit()
    return {"deleted": deleted}


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
    "/interview-configs/{config_id}/questions/import-bank",
    response_model=InterviewQuestionBankImportResult,
    status_code=status.HTTP_201_CREATED,
)
async def import_interview_question_bank_items(
    config_id: UUID,
    payload: InterviewQuestionBankImportRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewQuestionBankImportResult:
    try:
        created = await authoring_service.import_question_bank_items(
            db, config_id, payload.item_ids, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question_bank_item", config_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    group_count = len(
        {question.variant_group_id for question in created if question.variant_group_id}
    )
    return InterviewQuestionBankImportResult(
        created=[InterviewQuestionAuthoring.model_validate(question) for question in created],
        imported_group_count=group_count,
    )


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


@router.post(
    "/interview-configs/{config_id}/questions/{question_id}/approve-variants",
    response_model=dict[str, int],
)
async def approve_question_variants(
    config_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, int]:
    try:
        approved = await authoring_service.approve_question_variants(
            db, config_id, question_id, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question", question_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return {"approved": approved}


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


@router.delete(
    "/interview-configs/{config_id}/questions/{question_id}/variants",
    response_model=dict[str, int],
)
async def delete_question_variants(
    config_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, int]:
    try:
        deleted = await authoring_service.delete_question_variants(
            db, config_id, question_id, current_user
        )
    except NotFoundError as exc:
        raise _not_found("interview_question", question_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return {"deleted": deleted}


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
    """Temporarily unavailable until true per-question regeneration exists."""
    del config_id, question_id, current_user, db, arq_pool
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail={
            "error": "interview_question_regeneration_unavailable",
            "message": "Per-question regeneration is temporarily unavailable. "
            "Use full question generation from a draft interview instead.",
        },
    )


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
