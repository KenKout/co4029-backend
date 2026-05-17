"""Quizzes authoring router (T5.14).

Eleven endpoints under prefix ``/teacher`` covering quiz CRUD, manual
question CRUD, soft-delete, and the ARQ-enqueue triggers for
generation + per-question regeneration. Composes
:mod:`features.quizzes.services.authoring` (routers→services boundary,
T0.4 import-linter contract).

Security perimeter (FIX-SEC-1, Reconciliation §A9 + §E4)
--------------------------------------------------------
Every endpoint enforces a course-scoped permission check via the
factories in :mod:`._deps`:

* ``POST /teacher/courses/{course_id}/quizzes`` →
  :func:`features.access_control.policies.require_course_permission`
  on ``course.update`` (mirrors T3.7).
* Endpoints with a ``quiz_id`` path parameter →
  :func:`require_quiz_authoring_access` (walks
  ``quiz_id → courses.id``).
* Endpoints with a ``question_id`` path parameter →
  :func:`require_question_authoring_access` (walks
  ``question_id → quizzes → courses.id``; cross-checks any sibling
  ``quiz_id`` to prevent existence leaks).

No bare ``Depends(get_current_user)`` appears on any write endpoint
(verified by the source-grep test
``test_no_bare_get_current_user_on_quiz_authoring_endpoints``).

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
from abridgeai.features.quizzes.models import GenerationRun, Quiz, QuizQuestion
from abridgeai.features.quizzes.routers._deps import (
    require_question_authoring_access,
    require_quiz_authoring_access,
)
from abridgeai.features.quizzes.schemas import (
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizGenerationRequest,
    QuizGenerationRunRead,
    QuizQuestionAuthoring,
)
from abridgeai.features.quizzes.services import authoring as authoring_service

router = APIRouter(prefix="/teacher", tags=["quizzes-authoring"])

_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_QUIZ = require_quiz_authoring_access()
_REQUIRE_QUESTION = require_question_authoring_access()


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
    """ARQ Redis pool dependency (overridable in tests).

    Returns ``None`` until the app factory wires a real ``ArqRedis``
    pool via ``app.dependency_overrides``. Mirrors
    :func:`features.materials.routers.authoring.get_arq_pool`; the
    service layer accepts ``None`` and skips the enqueue (useful for
    tests that exercise DB writes without spinning up Redis +
    ``ArqRedis``).
    """
    return None


@router.post(
    "/courses/{course_id}/quizzes",
    response_model=QuizAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_quiz_under_course(
    course_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAuthoring:
    """Create a draft quiz on a module under ``course_id``.

    The legacy route was ``POST /modules/{module_id}/quizzes``; the
    authoring perimeter uses ``course_id`` as the path-anchor (the
    permission walks course-scoped). The body MUST carry ``module_id``
    so the service can resolve the parent module under this course.
    """
    module_id_raw = payload.get("module_id")
    if module_id_raw is None:
        raise _bad_request("module_id is required")
    try:
        module_id = UUID(str(module_id_raw))
    except (TypeError, ValueError) as exc:
        raise _bad_request("module_id must be a UUID") from exc

    create_payload = _AttrShim(payload)
    try:
        quiz = await authoring_service.create_quiz(db, module_id, create_payload, current_user)
    except NotFoundError as exc:
        raise _not_found("module", module_id) from exc
    if quiz.course_id != course_id:
        raise _bad_request("module does not belong to course")
    await db.commit()
    return QuizAuthoring.model_validate(quiz)


@router.get("/quizzes/{quiz_id}", response_model=QuizForAuthoringPublic)
async def get_quiz_authoring(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizForAuthoringPublic:
    """Authoring projection of a quiz + every question (with ``is_correct``)."""
    del current_user
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.quizzes.models import (  # noqa: PLC0415
        QuizQuestion,
        QuizQuestionOption,
    )

    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise _not_found("quiz", quiz_id)

    questions = list(
        (
            await db.execute(
                select(QuizQuestion)
                .where(QuizQuestion.quiz_id == quiz_id)
                .order_by(QuizQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    if questions:
        question_ids = [q.id for q in questions]
        options = list(
            (
                await db.execute(
                    select(QuizQuestionOption)
                    .where(QuizQuestionOption.question_id.in_(question_ids))
                    .order_by(QuizQuestionOption.position)
                )
            )
            .scalars()
            .all()
        )
        options_by_qid: dict[UUID, list[QuizQuestionOption]] = {qid: [] for qid in question_ids}
        for option in options:
            options_by_qid.setdefault(option.question_id, []).append(option)
        for question in questions:
            question.options = options_by_qid.get(question.id, [])  # type: ignore[attr-defined]

    return QuizForAuthoringPublic(
        quiz=QuizAuthoring.model_validate(quiz),
        questions=[QuizQuestionAuthoring.model_validate(q) for q in questions],
    )


@router.patch("/quizzes/{quiz_id}", response_model=QuizAuthoring)
async def update_quiz(
    quiz_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAuthoring:
    try:
        quiz = await authoring_service.update_quiz(db, quiz_id, _AttrShim(payload), current_user)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    await db.commit()
    return QuizAuthoring.model_validate(quiz)


@router.post("/quizzes/{quiz_id}/publish", response_model=QuizAuthoring)
async def publish_quiz(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAuthoring:
    try:
        quiz = await authoring_service.publish_quiz(db, quiz_id, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return QuizAuthoring.model_validate(quiz)


@router.delete("/quizzes/{quiz_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_quiz(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete the quiz + cascade to questions / options / revisions."""
    try:
        await authoring_service.delete_quiz(db, quiz_id, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    await db.commit()


@router.post(
    "/quizzes/{quiz_id}/generate",
    response_model=QuizGenerationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_generation(
    quiz_id: UUID,
    payload: QuizGenerationRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> QuizGenerationRunRead:
    """Persist a :class:`GenerationRun` (status=pending) and enqueue ARQ.

    The service commits inline so the worker can read the row out of
    band; the router does NOT call ``db.commit()`` again.
    """
    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise _not_found("quiz", quiz_id)
    enqueue_payload = _AttrShim(
        {
            **payload.model_dump(),
            "quiz_id": quiz_id,
        }
    )
    try:
        run = await authoring_service.start_generation_run(
            db,
            quiz.module_id,
            enqueue_payload,
            current_user,
            arq_pool=arq_pool,
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return _generation_run_view(run, quiz_id)


@router.get(
    "/quizzes/{quiz_id}/generation-runs/{run_id}",
    response_model=QuizGenerationRunRead,
)
async def get_generation_run(
    quiz_id: UUID,
    run_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizGenerationRunRead:
    """Status-poll endpoint — returns ``pending`` / ``running`` / ``completed`` / ``failed``."""
    del current_user
    run = await db.get(GenerationRun, run_id)
    if run is None:
        raise _not_found("generation_run", run_id)
    config_quiz_raw = (run.config_json or {}).get("quiz_id")
    if config_quiz_raw is None or str(config_quiz_raw) != str(quiz_id):
        raise _not_found("generation_run", run_id)
    return _generation_run_view(run, quiz_id)


@router.post(
    "/quizzes/{quiz_id}/questions",
    response_model=QuizQuestionAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_question(
    quiz_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionAuthoring:
    """Manual (non-AI) question creation — MCQ flavours validated up front."""
    create_payload = _AttrShim(payload)
    try:
        question = await authoring_service.create_question(
            db, quiz_id, create_payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await _attach_question_options(db, question)
    await db.commit()
    return QuizQuestionAuthoring.model_validate(question)


@router.patch(
    "/quizzes/{quiz_id}/questions/{question_id}",
    response_model=QuizQuestionAuthoring,
)
async def update_question(
    quiz_id: UUID,
    question_id: UUID,
    payload: dict[str, Any],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizQuestionAuthoring:
    del quiz_id
    try:
        question = await authoring_service.update_question(
            db, question_id, _AttrShim(payload), current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await _attach_question_options(db, question)
    await db.commit()
    return QuizQuestionAuthoring.model_validate(question)


@router.delete(
    "/quizzes/{quiz_id}/questions/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_question(
    quiz_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a question + repack sibling positions."""
    del quiz_id
    try:
        await authoring_service.delete_question(db, question_id, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    await db.commit()


@router.post(
    "/quizzes/{quiz_id}/questions/{question_id}/regenerate",
    response_model=QuizGenerationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_question(
    quiz_id: UUID,
    question_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUESTION)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> QuizGenerationRunRead:
    """Create a per-question regeneration run + enqueue ARQ.

    The service commits inline so the worker can read the run row.
    """
    try:
        run = await authoring_service.regenerate_question(
            db, question_id, current_user, arq_pool=arq_pool
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", question_id) from exc
    return _generation_run_view(run, quiz_id)


class _AttrShim:
    """Adapt a ``dict`` body into the ``model_dump`` / attr-access shape services expect.

    Kept private to this router. Service helpers were ported (T5.13) to
    consume Pydantic models via ``model_dump`` + ``getattr``; until the
    DTO surface lands in T5.x we accept loose ``dict`` bodies here and
    project them through this shim so the service signatures stay
    untouched.
    """

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


def _generation_run_view(run: GenerationRun, quiz_id: UUID) -> QuizGenerationRunRead:
    config = run.config_json or {}
    failure = config.get("failure") if isinstance(config, dict) else None
    error_message = (
        str(failure.get("message")) if isinstance(failure, dict) and "message" in failure else None
    )
    return QuizGenerationRunRead(
        id=run.id,
        quiz_id=quiz_id,
        status=run.status,
        started_at=run.started_at or run.created_at,
        completed_at=run.finished_at,
        error_message=error_message,
        pipeline_run_id=None,
    )


async def _attach_question_options(db: AsyncSession, question: QuizQuestion) -> None:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.quizzes.models import QuizQuestionOption  # noqa: PLC0415

    options = list(
        (
            await db.execute(
                select(QuizQuestionOption)
                .where(QuizQuestionOption.question_id == question.id)
                .order_by(QuizQuestionOption.position)
            )
        )
        .scalars()
        .all()
    )
    question.options = options  # type: ignore[attr-defined]


__all__ = [
    "get_arq_pool",
    "router",
]
