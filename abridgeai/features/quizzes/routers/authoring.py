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
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_course_permission
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.routers._deps import (
    require_question_authoring_access,
    require_quiz_authoring_access,
)
from abridgeai.features.quizzes.schemas import (
    QuestionBankEntry,
    QuestionBankImportRequest,
    QuestionBankPage,
    QuizAttemptTeacherRead,
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizGenerationProgress,
    QuizGenerationRequest,
    QuizGenerationRunRead,
    QuizOptionDistribution,
    QuizPerStudentRow,
    QuizQuestionAuthoring,
    QuizQuestionBreakdown,
    QuizResultsRead,
    QuizResultsSummary,
    QuizScoreBucket,
)
from abridgeai.features.quizzes.services import (
    authoring as authoring_service,
)
from abridgeai.features.quizzes.services import (
    question_bank as question_bank_service,
)
from abridgeai.features.quizzes.services.authoring import QuizPublishValidationError
from abridgeai.features.quizzes.services.publish_gate import QuizApprovalRequiredError

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


async def _resolve_student_names(db: AsyncSession, student_ids: set[UUID]) -> dict[UUID, str]:
    """Batch-resolve ``{student_id: display_name}`` for a set of ids.

    Mirrors ``interviews.routers.authoring.list_config_sessions`` — a
    single ``users LEFT JOIN user_profiles`` round-trip regardless of how
    many distinct students are in the result set.
    """
    if not student_ids:
        return {}
    from sqlalchemy import text as _text  # noqa: PLC0415

    rows = (
        await db.execute(
            _text(
                "SELECT u.id, COALESCE(p.display_name, u.primary_email) AS name "
                "FROM users u "
                "LEFT JOIN user_profiles p ON p.user_id = u.id "
                "WHERE u.id = ANY(:ids)"
            ),
            {"ids": list(student_ids)},
        )
    ).all()
    return {row[0]: row[1] for row in rows}


def _attempt_teacher_view(
    attempt: Any,  # noqa: ANN401  -- ORM row
    quiz_title: str,
    student_name: str | None,
) -> QuizAttemptTeacherRead:
    return QuizAttemptTeacherRead(
        id=attempt.id,
        quiz_id=attempt.quiz_id,
        quiz_title=quiz_title,
        student_id=attempt.student_id,
        student_name=student_name,
        attempt_number=attempt.attempt_number,
        status=attempt.status,
        started_at=attempt.started_at,
        submitted_at=attempt.submitted_at,
        time_taken_seconds=attempt.time_taken_seconds,
        score_percent=attempt.score_percent,
        passed=attempt.passed,
    )


@router.get(
    "/courses/{course_id}/quiz-attempts",
    response_model=list[QuizAttemptTeacherRead],
)
async def list_course_quiz_attempts(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizAttemptTeacherRead]:
    """Every quiz attempt (any student, any quiz) in this course.

    Powers the teacher's course-wide "Assessments" tab.
    """
    del current_user
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    rows = await _analytics_q.list_attempts_for_course(db, course_id)
    names = await _resolve_student_names(db, {row.QuizAttempt.student_id for row in rows})
    return [
        _attempt_teacher_view(row.QuizAttempt, row.title, names.get(row.QuizAttempt.student_id))
        for row in rows
    ]


@router.get(
    "/courses/{course_id}/students/{student_id}/quiz-attempts",
    response_model=list[QuizAttemptTeacherRead],
)
async def list_student_quiz_attempts(
    course_id: UUID,
    student_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizAttemptTeacherRead]:
    """Every quiz attempt by one student across this course's quizzes.

    Powers the teacher's per-student profile page.
    """
    del current_user
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    rows = await _analytics_q.list_attempts_for_student_in_course(db, course_id, student_id)
    names = await _resolve_student_names(db, {student_id})
    student_name = names.get(student_id)
    return [_attempt_teacher_view(row.QuizAttempt, row.title, student_name) for row in rows]


@router.get(
    "/quizzes/{quiz_id}/results",
    response_model=QuizResultsRead,
)
async def get_quiz_results(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizResultsRead:
    """Assemble the full teacher-facing results analytics payload for a quiz.

    Combines the grading-method-aware summary, a per-student rollup, and the
    per-question breakdown into one response. A quiz with zero completed
    attempts returns a zeroed summary + empty ``per_student`` while
    ``per_question`` still lists every question (zero counts).
    """
    del current_user  # permission already enforced by Depends
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise _not_found("quiz", quiz_id)

    summary_dict = await _analytics_q.quiz_results_summary(db, quiz_id, quiz.grading_method)
    per_question_list = await _analytics_q.quiz_question_breakdown(db, quiz_id)
    rollup = await _analytics_q.quiz_per_student_rollup(db, quiz_id)

    names = await _resolve_student_names(db, {row["student_id"] for row in rollup})

    summary = QuizResultsSummary(
        total_attempts=summary_dict["total_attempts"],
        unique_students=summary_dict["unique_students"],
        mean_score=summary_dict["mean_score"],
        median_score=summary_dict["median_score"],
        p25=summary_dict["p25"],
        p75=summary_dict["p75"],
        pass_rate=summary_dict["pass_rate"],
        mean_time_seconds=summary_dict["mean_time_seconds"],
        histogram=[QuizScoreBucket(**bucket) for bucket in summary_dict["histogram"]],
    )
    per_student = [
        QuizPerStudentRow(
            student_id=row["student_id"],
            student_name=names.get(row["student_id"]),
            best_score_percent=row["best_score_percent"],
            latest_score_percent=row["latest_score_percent"],
            attempts_count=row["attempts_count"],
            passed=row["passed"],
            last_attempt_at=row["last_attempt_at"],
        )
        for row in rollup
    ]
    per_question = [
        QuizQuestionBreakdown(
            question_id=q["question_id"],
            prompt=q["prompt"],
            correct_count=q["correct_count"],
            answered_count=q["answered_count"],
            correctness_rate=q["correctness_rate"],
            option_distribution=[QuizOptionDistribution(**opt) for opt in q["option_distribution"]],
        )
        for q in per_question_list
    ]
    return QuizResultsRead(
        quiz_id=quiz.id,
        quiz_title=quiz.title,
        passing_score_percent=quiz.passing_score_percent,
        grading_method=quiz.grading_method,
        summary=summary,
        per_student=per_student,
        per_question=per_question,
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
    except QuizPublishValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "publish_gate_t_exp_required",
                "message": str(exc),
                "missing_t_exp_question_ids": [str(q) for q in exc.missing_t_exp_question_ids],
            },
        ) from exc
    except QuizApprovalRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "pending_review",
                "message": str(exc),
                "pending_question_ids": [str(q) for q in exc.pending_question_ids],
            },
        ) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return QuizAuthoring.model_validate(quiz)


class BulkSetItem(BaseModel):
    question_id: UUID
    expected_response_time_ms: int = Field(gt=0)


class BulkSetExpectedTimeRequest(BaseModel):
    items: list[BulkSetItem] = Field(min_length=1)


class BulkSetExpectedTimeResponse(BaseModel):
    updated: int


@router.post(
    "/quizzes/{quiz_id}/questions/bulk-set-expected-time",
    response_model=BulkSetExpectedTimeResponse,
)
async def bulk_set_expected_time(
    quiz_id: UUID,
    payload: BulkSetExpectedTimeRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BulkSetExpectedTimeResponse:
    items = [(item.question_id, item.expected_response_time_ms) for item in payload.items]
    try:
        updated = await authoring_service.bulk_set_expected_response_time(
            db, quiz_id, items, current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return BulkSetExpectedTimeResponse(updated=updated)


class BulkApproveRequest(BaseModel):
    question_ids: list[UUID] = Field(min_length=1)


class BulkApproveResponse(BaseModel):
    approved: int


@router.post(
    "/quizzes/{quiz_id}/questions/bulk-approve",
    response_model=BulkApproveResponse,
)
async def bulk_approve_questions(
    quiz_id: UUID,
    payload: BulkApproveRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BulkApproveResponse:
    """Approve many questions at once (bulk sign-off for AI content)."""
    try:
        approved = await authoring_service.bulk_approve_questions(
            db, quiz_id, payload.question_ids, current_user
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return BulkApproveResponse(approved=approved)


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
    # Phase 2 of FR-5 schema port: pass the strictly-typed Pydantic
    # ``QuizGenerationRequest`` straight through to the service layer.
    # The route's ``{quiz_id}`` always wins over any body-side
    # ``quiz_id`` (defence in depth — the router is the trust boundary).
    payload_with_route_quiz = payload.model_copy(update={"quiz_id": quiz_id})
    try:
        run = await authoring_service.start_generation_run(
            db,
            quiz.module_id,
            payload_with_route_quiz,
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
    "/quizzes/{quiz_id}/generation-runs/latest",
    response_model=QuizGenerationRunRead | None,
)
async def get_latest_quiz_generation_run(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizGenerationRunRead | None:
    """Return the most recent ``GenerationRun`` for this quiz, if any.

    Lets the SPA reattach to an in-flight (or terminal) run on mount
    without persisting handles in the browser — survives cross-device
    sessions, tab closes, and lets a second teacher viewing the same
    quiz see the in-flight run too. Returns ``null`` (HTTP 200) when
    the quiz has never been generated.
    """
    del current_user
    run = await authoring_service.get_latest_generation_run(db, quiz_id)
    if run is None:
        return None
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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


@router.get(
    "/courses/{course_id}/question-bank",
    response_model=QuestionBankPage,
)
async def list_question_bank(  # noqa: PLR0913 -- filters mirror service signature
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    module_id: UUID | None = None,
    lesson_id: UUID | None = None,
    question_type: str | None = None,
    bloom_level: str | None = None,
    difficulty: str | None = None,
    review_status: str | None = "approved",
    search: str | None = None,
    exclude_quiz_id: UUID | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> QuestionBankPage:
    """Browse authored questions across the course for cross-quiz reuse.

    Defaults to ``review_status='approved'`` so only vetted questions
    surface; pass ``review_status=`` (empty) to widen. ``exclude_quiz_id``
    is convenient for the modal launched from a target quiz so its own
    questions don't appear in the bank list. ``cursor`` is opaque and
    round-trips through subsequent calls.
    """
    del current_user  # permission already enforced by Depends
    try:
        page = await question_bank_service.list_bank_entries(
            db,
            course_id=course_id,
            module_id=module_id,
            lesson_id=lesson_id,
            question_type=question_type,
            bloom_level=bloom_level,
            difficulty=difficulty,
            review_status=review_status if review_status else None,
            search=search,
            exclude_quiz_id=exclude_quiz_id,
            limit=limit,
            cursor=cursor,
        )
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return QuestionBankPage(
        items=[QuestionBankEntry.model_validate(row) for row in page.items],
        next_cursor=page.next_cursor,
    )


@router.post(
    "/quizzes/{quiz_id}/questions/import",
    response_model=list[QuizQuestionAuthoring],
    status_code=status.HTTP_201_CREATED,
)
async def import_questions_from_bank(
    quiz_id: UUID,
    payload: QuestionBankImportRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizQuestionAuthoring]:
    """Clone bank questions into ``quiz_id``.

    Each clone has a fresh id, ``review_status='pending'``, and an
    ``imported_from_question_id`` back-pointer. Options are cloned in
    place. Source questions must live in the same course as the target.
    """
    try:
        cloned = await question_bank_service.import_questions(
            db,
            target_quiz_id=quiz_id,
            source_question_ids=payload.source_question_ids,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found("quiz_question", quiz_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return [QuizQuestionAuthoring.model_validate(question) for question in cloned]


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

    def model_dump(
        self,
        exclude_unset: bool = False,
        mode: str | None = None,
        exclude: set[str] | None = None,
        include: set[str] | None = None,
    ) -> dict[str, Any]:
        del exclude_unset, mode
        data = dict(self._data)
        if include is not None:
            data = {k: v for k, v in data.items() if k in include}
        if exclude:
            data = {k: v for k, v in data.items() if k not in exclude}
        return data

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
    # Live-progress projection (migration 0035). ``progress_json`` is
    # written incrementally by the pipeline checkpoint helper; validate it
    # leniently so a malformed/partial checkpoint never 500s the poll.
    progress = None
    raw_progress = getattr(run, "progress_json", None)
    if isinstance(raw_progress, dict) and raw_progress:
        try:
            progress = QuizGenerationProgress.model_validate(raw_progress)
        except ValidationError:
            progress = None
    return QuizGenerationRunRead(
        id=run.id,
        quiz_id=quiz_id,
        status=run.status,
        started_at=run.started_at or run.created_at,
        completed_at=run.finished_at,
        error_message=error_message,
        pipeline_run_id=None,
        progress=progress,
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
