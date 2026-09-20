# ruff: noqa
"""Split teacher quiz authoring routes.

The modules share the historical ``authoring.router`` object so the public
router import and dependency overrides remain unchanged.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.pagination import PageResponse, paginate_sequence
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.schemas import (
    CourseAssessmentSummaryRead,
    FeedbackBandIn,
    FeedbackBandRead,
    ManualGradeIn,
    ManualGradeRead,
    NeedsGradingRow,
    QuestionBankEntry,
    QuestionBankImportRequest,
    QuestionBankPage,
    QuizAttemptIntegrityEvent,
    QuizAttemptReviewOption,
    QuizAttemptReviewQuestion,
    QuizAttemptTeacherPage,
    QuizAttemptTeacherRead,
    QuizAttemptTeacherReview,
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizGenerationProgress,
    QuizGenerationRequest,
    QuizGenerationRunRead,
    QuizGradeRow,
    QuizOptionDistribution,
    QuizOverrideIn,
    QuizOverrideRead,
    QuizPerStudentRow,
    QuizQuestionAuthoring,
    QuizQuestionBreakdown,
    QuizResultsRead,
    QuizResultsSummary,
    QuizScoreBucket,
    RegradeRunRead,
    RegradeScopeIn,
)
from abridgeai.features.quizzes.routers.authoring import *  # noqa: F403
from abridgeai.features.quizzes.routers.authoring import (
    _QUIZ_RESULT_PATTERN,
    _REQUIRE_COURSE_UPDATE,
    _REQUIRE_QUESTION,
    _REQUIRE_QUIZ,
    _StudentAvatarTarget,
    _attempt_teacher_view,
    _bad_request,
    _conflict,
    _not_found,
    _resolve_student_contacts,
    _resolve_student_names,
)
from .authoring_questions import _generation_run_view
from abridgeai.features.quizzes.services import authoring as authoring_service
from abridgeai.features.quizzes.services import question_bank as question_bank_service
from abridgeai.features.quizzes.services.authoring import QuizPublishValidationError
from abridgeai.features.quizzes.services.publish_gate import QuizApprovalRequiredError
from abridgeai.infrastructure.s3 import create_stream_url


class _AttrShim:
    """Adapt a dict to the model and attribute interface used by services."""

    _RETIRED_KEYS = frozenset({"browser_security"})

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = {k: v for k, v in data.items() if k not in self._RETIRED_KEYS}

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

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            value = self._data[name]
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [_AttrShim(item) for item in value]
            if isinstance(value, dict):
                return _AttrShim(value)
            return value
        raise AttributeError(name)


@router.get(
    "/quizzes/{quiz_id}/results",
    response_model=QuizResultsRead,
)
async def get_quiz_results(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_breakdowns: bool = True,
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

    module = await courses_api.get_module_by_id(db, quiz.module_id)

    summary_dict = await _analytics_q.quiz_results_summary(db, quiz_id, quiz.grading_method)
    per_question_list = (
        await _analytics_q.quiz_question_breakdown(db, quiz_id) if include_breakdowns else []
    )
    rollup = (
        await _analytics_q.quiz_per_student_rollup(db, quiz_id, quiz.grading_method)
        if include_breakdowns
        else []
    )

    contacts = await _resolve_student_contacts(
        db, {row["student_id"] for row in rollup}, include_avatar=True
    )

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
            student_name=(
                contacts.get(row["student_id"], (None, None, None))[0]
                or contacts.get(row["student_id"], (None, None, None))[1]
            ),
            student_email=contacts.get(row["student_id"], (None, None, None))[1],
            student_avatar_url=contacts.get(row["student_id"], (None, None, None))[2],
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
        module_id=quiz.module_id,
        module_title=module.title if module else None,
        passing_score_percent=quiz.passing_score_percent,
        grading_method=quiz.grading_method,
        summary=summary,
        per_student=per_student,
        per_question=per_question,
    )


@router.get(
    "/quizzes/{quiz_id}/results/students",
    response_model=PageResponse[QuizPerStudentRow],
)
async def get_quiz_results_students(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=200)] = None,
    status_filter: Annotated[
        str | None, Query(alias="status", pattern="^(passed|failed|ungraded)$")
    ] = None,
    score_mode: Annotated[str, Query(pattern="^(best|latest)$")] = "best",
    sort: str | None = None,
    sort_dir: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> PageResponse[QuizPerStudentRow]:
    del current_user
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise _not_found("quiz", quiz_id)
    rows = await _analytics_q.quiz_per_student_rollup(db, quiz_id, quiz.grading_method)
    contacts = await _resolve_student_contacts(
        db, {row["student_id"] for row in rows}, include_avatar=True
    )
    projected = [
        QuizPerStudentRow(
            student_id=row["student_id"],
            student_name=(
                contacts.get(row["student_id"], (None, None, None))[0]
                or contacts.get(row["student_id"], (None, None, None))[1]
            ),
            student_email=contacts.get(row["student_id"], (None, None, None))[1],
            student_avatar_url=contacts.get(row["student_id"], (None, None, None))[2],
            best_score_percent=row["best_score_percent"],
            latest_score_percent=row["latest_score_percent"],
            attempts_count=row["attempts_count"],
            passed=row["passed"],
            last_attempt_at=row["last_attempt_at"],
        )
        for row in rows
    ]
    score = lambda row: row.best_score_percent if score_mode == "best" else row.latest_score_percent
    result = paginate_sequence(
        projected,
        page=page,
        page_size=page_size,
        search=search,
        search_text=lambda row: (
            f"{row.student_name or ''} {row.student_email or ''} {row.student_id}"
        ),
        predicate=lambda row: (
            status_filter is None
            or (status_filter == "passed" and row.passed is True)
            or (status_filter == "failed" and row.passed is False)
            or (status_filter == "ungraded" and row.passed is None)
        ),
        sort=sort,
        sort_dir=sort_dir,
        sortable={
            "student": lambda row: row.student_name or row.student_email or str(row.student_id),
            "score": score,
            "attempts": lambda row: row.attempts_count,
            "last_attempt": lambda row: row.last_attempt_at,
        },
    )
    return PageResponse[QuizPerStudentRow](**result.__dict__)


@router.get(
    "/quizzes/{quiz_id}/results/questions",
    response_model=PageResponse[QuizQuestionBreakdown],
)
async def get_quiz_results_questions(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=200)] = None,
    difficulty: Annotated[str | None, Query(pattern="^(hard|medium|easy|unanswered)$")] = None,
    sort: str | None = None,
    sort_dir: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> PageResponse[QuizQuestionBreakdown]:
    del current_user
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    raw = await _analytics_q.quiz_question_breakdown(db, quiz_id)
    rows = [QuizQuestionBreakdown(**row) for row in raw]

    def matches(row: QuizQuestionBreakdown) -> bool:
        rate = row.correctness_rate
        return (
            difficulty is None
            or (difficulty == "unanswered" and rate is None)
            or (difficulty == "hard" and rate is not None and rate < 0.5)
            or (difficulty == "medium" and rate is not None and 0.5 <= rate < 0.8)
            or (difficulty == "easy" and rate is not None and rate >= 0.8)
        )

    result = paginate_sequence(
        rows,
        page=page,
        page_size=page_size,
        search=search,
        search_text=lambda row: row.prompt,
        predicate=matches,
        sort=sort,
        sort_dir=sort_dir,
        sortable={
            "question": lambda row: row.prompt,
            "answered": lambda row: row.answered_count,
            "correct": lambda row: row.correctness_rate,
        },
    )
    return PageResponse[QuizQuestionBreakdown](**result.__dict__)


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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return QuizAuthoring.model_validate(quiz)


@router.post("/quizzes/{quiz_id}/archive", response_model=QuizAuthoring)
async def archive_quiz(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAuthoring:
    """Withdraw a quiz from new learner access without deleting evidence.

    Existing in-progress attempts remain resumable and submittable through the
    attempt endpoints, which intentionally load their persisted attempt
    snapshot rather than requiring the quiz to remain published.
    """
    try:
        quiz = await authoring_service.archive_quiz(db, quiz_id, current_user)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
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
    """Return the most recent ``GenerationRun`` for this quiz, if any."""
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
    from . import authoring as authoring_router  # noqa: PLC0415

    return authoring_router._generation_run_view(run, quiz_id)
