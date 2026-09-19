"""Teacher quiz authoring endpoints with course-scoped authorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_course_permission
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.routers._deps import (
    require_question_authoring_access,
    require_quiz_authoring_access,
)
from abridgeai.features.quizzes.routers.curated_question_bank import (
    router as curated_question_bank_router,
)
from abridgeai.features.quizzes.schemas import (  # noqa: F401
    CourseAssessmentSummaryRead,
    ManualGradeIn,
    QuizAttemptIntegrityEvent,
    QuizAttemptReviewOption,
    QuizAttemptReviewQuestion,
    QuizAttemptTeacherPage,
    QuizAttemptTeacherRead,
    QuizAttemptTeacherReview,
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizOverrideIn,
    QuizQuestionAuthoring,
)
from abridgeai.features.quizzes.services import (
    authoring as authoring_service,
)
from abridgeai.infrastructure.s3 import create_stream_url

router = APIRouter(prefix="/teacher", tags=["quizzes-authoring"])
router.include_router(curated_question_bank_router)

_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_QUIZ = require_quiz_authoring_access()
_REQUIRE_QUESTION = require_question_authoring_access()

# The Assessments tab's Result buckets. Validated at the boundary rather than
# passed through to SQL: an unrecognised value would otherwise match nothing
# and read to the teacher as "no attempts" rather than as a bad request.
_QUIZ_RESULT_PATTERN = r"^(in_progress|passed|not_passed|grading)$"


@dataclass(frozen=True)
class _StudentAvatarTarget:
    bucket: str
    object_key: str


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
    """Return the app-overridable ARQ Redis pool dependency."""
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
    """Create a draft quiz on a module under ``course_id``."""
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
        outcome_ids = {q.learning_outcome_id for q in questions if q.learning_outcome_id}
        positions = await _resolve_outcome_positions(db, outcome_ids)
        _fill_outcome_positions(questions, positions)

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
    contacts = await _resolve_student_contacts(db, student_ids)
    return {
        student_id: display_name or email or str(student_id)
        for student_id, (display_name, email, _avatar_url) in contacts.items()
    }


async def _resolve_student_contacts(
    db: AsyncSession,
    student_ids: set[UUID],
    *,
    include_avatar: bool = False,
) -> dict[UUID, tuple[str | None, str | None, str | None]]:
    """Batch-resolve result identities and optionally mint avatar URLs."""
    if not student_ids:
        return {}
    from sqlalchemy import text as _text  # noqa: PLC0415

    rows = (
        await db.execute(
            _text(
                "SELECT u.id, p.display_name, u.primary_email, so.bucket, so.object_key "
                "FROM users u "
                "LEFT JOIN user_profiles p ON p.user_id = u.id "
                "LEFT JOIN storage_objects so ON so.id = p.avatar_object_id "
                "WHERE u.id = ANY(:ids)"
            ),
            {"ids": list(student_ids)},
        )
    ).all()
    contacts: dict[UUID, tuple[str | None, str | None, str | None]] = {
        row[0]: (row[1], row[2], None) for row in rows
    }
    if include_avatar:
        for row in rows:
            if not row[3] or not row[4]:
                continue
            try:
                avatar_url, _ = await create_stream_url(
                    _StudentAvatarTarget(bucket=row[3], object_key=row[4])  # type: ignore[arg-type]
                )
            except Exception:  # noqa: BLE001 -- avatar storage is optional
                avatar_url = None
            if avatar_url:
                contacts[row[0]] = (row[1], row[2], avatar_url)
    return contacts


def _attempt_teacher_view(
    attempt: Any,  # noqa: ANN401  -- ORM row
    quiz_title: str,
    student_name: str | None,
    integrity_flags: int = 0,
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
        integrity_flags=integrity_flags,
        integrity_score=int(getattr(attempt, "integrity_score", 0) or 0),
        integrity_score_threshold=int(
            (getattr(attempt, "integrity_policy_snapshot", None) or {}).get("score_threshold", 0)
            or 0
        ),
        integrity_flagged=bool(getattr(attempt, "integrity_warning_issued", False)),
    )


@router.get(
    "/courses/{course_id}/quiz-attempts",
    response_model=QuizAttemptTeacherPage,
)
async def list_course_quiz_attempts(  # noqa: PLR0913 -- one parameter per client control
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    cursor: Annotated[str | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    title: Annotated[str | None, Query(max_length=255)] = None,
    result: Annotated[str | None, Query(pattern=_QUIZ_RESULT_PATTERN)] = None,
    since: Annotated[datetime | None, Query()] = None,
) -> QuizAttemptTeacherPage:
    """One page of quiz attempts (any student, any quiz) in this course."""
    del current_user
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    page = await _analytics_q.list_attempts_for_course(
        db,
        course_id,
        limit=limit,
        cursor=cursor,
        search=search,
        title=title,
        result=result,
        since=since,
    )
    rows = page.items
    names = await _resolve_student_names(db, {row.QuizAttempt.student_id for row in rows})
    return QuizAttemptTeacherPage(
        items=[
            _attempt_teacher_view(
                row.QuizAttempt,
                row.title,
                names.get(row.QuizAttempt.student_id),
                int(row.integrity_flags or 0),
            )
            for row in rows
        ],
        next_cursor=page.next_cursor,
    )


@router.get(
    "/courses/{course_id}/assessment-summary",
    response_model=CourseAssessmentSummaryRead,
)
async def course_assessment_summary(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAssessmentSummaryRead:
    """Whole-course assessment aggregates for the Assessments tab's tiles.

    Deliberately independent of the two paginated lists and of whichever tab
    is open: the tiles describe the course, not the page, and the title
    dropdowns must offer every title rather than only those on screen.
    """
    del current_user
    from abridgeai.features.interviews.api import public as _interviews_api  # noqa: PLC0415
    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415

    facets = await _analytics_q.course_assessment_facets(db, course_id)
    quiz_student_ids = await _analytics_q.course_quiz_student_ids(db, course_id)
    interview_facets = await _interviews_api.course_interview_facets(db, course_id)
    return CourseAssessmentSummaryRead(
        students_assessed=len(quiz_student_ids | interview_facets["student_ids"]),
        quiz_attempt_count=facets["quiz_attempt_count"],
        quiz_pass_rate=facets["quiz_pass_rate"],
        interview_session_count=interview_facets["session_count"],
        quiz_titles=facets["quiz_titles"],
        interview_titles=interview_facets["titles"],
    )


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
    return [
        _attempt_teacher_view(
            row.QuizAttempt, row.title, student_name, int(row.integrity_flags or 0)
        )
        for row in rows
    ]


@router.get(
    "/courses/{course_id}/quiz-attempts/{attempt_id}",
    response_model=QuizAttemptTeacherReview,
)
async def get_course_quiz_attempt_detail(
    course_id: UUID,
    attempt_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizAttemptTeacherReview:
    """Teacher-facing detail for a single attempt.

    Powers the quiz-attempt detail page. Combines the attempt summary
    (student name + quiz title), the full per-question review (prompt,
    options, the student's answer, correctness), and any integrity events.

    404s when the attempt doesn't exist or belongs to a quiz outside this
    course (enforced by the course-scoped query + the ``course.update``
    permission guard).
    """
    del current_user  # permission enforced by Depends
    from decimal import Decimal  # noqa: PLC0415

    from abridgeai.features.quizzes.queries import analytics as _analytics_q  # noqa: PLC0415
    from abridgeai.features.quizzes.queries import published as _published_q  # noqa: PLC0415

    row = await _analytics_q.get_course_attempt_for_review(db, course_id, attempt_id)
    if row is None:
        raise _not_found("quiz_attempt", attempt_id)

    attempt = row.QuizAttempt
    names = await _resolve_student_names(db, {attempt.student_id})

    answers_by_question = {a.question_id: a for a in attempt.answers}
    questions_with_options = await _published_q.list_quiz_questions_with_options(
        db, attempt.quiz_id
    )
    review_questions = [
        QuizAttemptReviewQuestion(
            question_id=question.id,
            position=question.position,
            question_type=question.question_type,
            prompt_text=question.prompt_text,
            explanation=question.explanation,
            hint_text=question.hint_text,
            options=[QuizAttemptReviewOption.model_validate(opt) for opt in options],
            selected_option_id=(
                answers_by_question[question.id].selected_option_id
                if question.id in answers_by_question
                else None
            ),
            answer_text=(
                answers_by_question[question.id].answer_text
                if question.id in answers_by_question
                else None
            ),
            is_correct=(
                answers_by_question[question.id].is_correct
                if question.id in answers_by_question
                else False
            ),
            points_awarded=(
                answers_by_question[question.id].points_awarded
                if question.id in answers_by_question
                else Decimal("0")
            ),
            hint_used=(
                answers_by_question[question.id].hint_used
                if question.id in answers_by_question
                else False
            ),
            t_actual_ms=(
                answers_by_question[question.id].t_actual_ms
                if question.id in answers_by_question
                else None
            ),
        )
        for question, options in questions_with_options
    ]

    integrity_rows = await _analytics_q.list_integrity_events_for_attempt(db, attempt_id)

    return QuizAttemptTeacherReview(
        attempt=_attempt_teacher_view(attempt, row.title, names.get(attempt.student_id)),
        questions=review_questions,
        integrity_events=[QuizAttemptIntegrityEvent.model_validate(ev) for ev in integrity_rows],
    )



# Keep the historical router module as the single public import surface.
from . import authoring_questions as _authoring_questions  # noqa: E402, F401, I001
from . import authoring_extended as _authoring_extended  # noqa: E402, F401, I001
from . import authoring_admin as _authoring_admin  # noqa: E402, F401, I001
from .authoring_questions import (  # noqa: E402, F401, I001
    _AttrShim as _AttrShim,
    _fill_outcome_positions as _fill_outcome_positions,
    _generation_run_view as _generation_run_view,
    _resolve_outcome_positions as _resolve_outcome_positions,
)

# Compatibility exports: callers historically imported the route handlers and
# request models directly from this module. Keep that public surface while the
# implementations live in the split modules.
from .authoring_questions import (  # noqa: E402, F401, I001
    create_question,
    delete_question,
    duplicate_question,
    import_questions_from_bank,
    list_question_bank,
    regenerate_question,
    update_question,
)
from .authoring_extended import (  # noqa: E402, F401, I001
    BulkApproveRequest,
    BulkApproveResponse,
    BulkSetExpectedTimeRequest,
    BulkSetExpectedTimeResponse,
    BulkSetItem,
    archive_quiz,
    bulk_approve_questions,
    bulk_set_expected_time,
    delete_quiz,
    get_generation_run,
    get_latest_quiz_generation_run,
    get_quiz_results,
    publish_quiz,
    start_generation,
    update_quiz,
)
from .authoring_admin import (  # noqa: E402, F401, I001
    _AuditEventRow as _AuditEventRow,
    _FeedbackBandsBody as _FeedbackBandsBody,
    _ImportBody as _ImportBody,
    _report_download as _report_download,
    _serialize_regrade_run as _serialize_regrade_run,
    commit_regrade_run,
    create_quiz_override,
    delete_quiz_override,
    export_quiz_gradebook,
    export_quiz_questions,
    get_quiz_gradebook,
    get_regrade_run,
    get_responses_report,
    get_statistics_report,
    grade_answer_manually,
    import_questions_from_file,
    list_feedback_bands,
    list_needs_grading,
    list_quiz_audit_events,
    list_quiz_overrides,
    regrade_dry_run,
    set_feedback_bands,
    update_quiz_override,
)
from .authoring_questions import question_bank_service as question_bank_service  # noqa: E402, F401, I001
