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
from abridgeai.core.security import CurrentUser
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
from abridgeai.features.quizzes.services import authoring as authoring_service
from abridgeai.features.quizzes.services import question_bank as question_bank_service
from abridgeai.features.quizzes.services.authoring import QuizPublishValidationError
from abridgeai.features.quizzes.services.publish_gate import QuizApprovalRequiredError
from abridgeai.infrastructure.s3 import create_stream_url

def _serialize_regrade_run(run: Any) -> RegradeRunRead:  # noqa: ANN401 -- ORM projection
    """Project a QuizRegradeRun (+ loaded items) to the API DTO."""
    from abridgeai.features.quizzes.schemas import RegradeItemRead  # noqa: PLC0415

    return RegradeRunRead(
        id=run.id,
        quiz_id=run.quiz_id,
        status=run.status,
        attempts_affected=run.attempts_affected,
        answers_changed=run.answers_changed,
        created_at=run.created_at,
        committed_at=run.committed_at,
        items=[
            RegradeItemRead(
                attempt_id=it.attempt_id,
                question_id=it.question_id,
                old_is_correct=it.old_is_correct,
                new_is_correct=it.new_is_correct,
                old_points=it.old_points,
                new_points=it.new_points,
            )
            for it in run.items
        ],
    )


@router.post(
    "/quizzes/{quiz_id}/regrade/dry-run",
    response_model=RegradeRunRead,
)
async def regrade_dry_run(
    quiz_id: UUID,
    body: RegradeScopeIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegradeRunRead:
    """Preview a regrade: compute per-answer deltas against the CURRENT question
    definitions and persist a dry-run run. Does NOT mutate attempts."""
    from abridgeai.features.quizzes.services import regrade as _regrade  # noqa: PLC0415

    try:
        run = await _regrade.compute_regrade(
            db,
            quiz_id=quiz_id,
            attempt_ids=body.attempt_ids or None,
            question_ids=body.question_ids or None,
            requested_by=current_user.user_id,
        )
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    await db.commit()
    run = await _regrade.get_regrade_run(db, quiz_id=quiz_id, run_id=run.id)
    return _serialize_regrade_run(run)


@router.get(
    "/quizzes/{quiz_id}/regrade/runs/{run_id}",
    response_model=RegradeRunRead,
)
async def get_regrade_run(
    quiz_id: UUID,
    run_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegradeRunRead:
    """Read a regrade run (with its per-answer delta items)."""
    del current_user
    from abridgeai.features.quizzes.services import regrade as _regrade  # noqa: PLC0415

    run = await _regrade.get_regrade_run(db, quiz_id=quiz_id, run_id=run_id)
    if run is None:
        raise _not_found("regrade run", run_id)
    return _serialize_regrade_run(run)


@router.post(
    "/quizzes/{quiz_id}/regrade/runs/{run_id}/commit",
    response_model=RegradeRunRead,
)
async def commit_regrade_run(
    quiz_id: UUID,
    run_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegradeRunRead:
    """Commit a dry run: apply deltas to answers, recompute affected attempt
    scores, mark the run committed. A committed run cannot be re-committed (409)."""
    del current_user
    from abridgeai.features.quizzes.services import regrade as _regrade  # noqa: PLC0415

    try:
        await _regrade.commit_regrade(db, quiz_id=quiz_id, run_id=run_id)
    except NotFoundError as exc:
        raise _not_found("regrade run", run_id) from exc
    except AppError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await db.commit()
    run = await _regrade.get_regrade_run(db, quiz_id=quiz_id, run_id=run_id)
    return _serialize_regrade_run(run)


@router.get(
    "/quizzes/{quiz_id}/needs-grading",
    response_model=list[NeedsGradingRow],
)
async def list_needs_grading(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[NeedsGradingRow]:
    """Teacher grading queue: open-response answers awaiting a human mark."""
    del current_user
    from abridgeai.features.quizzes.services import (  # noqa: PLC0415
        manual_grading as _manual,
    )

    rows = await _manual.list_needs_grading(db, quiz_id=quiz_id)
    return [
        NeedsGradingRow(
            answer_id=answer.id,
            attempt_id=attempt.id,
            question_id=question.id,
            student_id=attempt.student_id,
            question_type=question.question_type,
            prompt_text=question.prompt_text,
            answer_text=answer.answer_text,
            submitted_at=attempt.submitted_at,
        )
        for answer, question, attempt in rows
    ]


@router.patch(
    "/quizzes/{quiz_id}/answers/{answer_id}/grade",
    response_model=ManualGradeRead,
)
async def grade_answer_manually(
    quiz_id: UUID,
    answer_id: UUID,
    body: ManualGradeIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ManualGradeRead:
    """Record a teacher mark + feedback on one open-response answer, recompute
    the attempt score, and flip the attempt to graded when nothing else on it
    still needs a human."""
    from abridgeai.features.quizzes.services import (  # noqa: PLC0415
        manual_grading as _manual,
    )

    try:
        answer = await _manual.grade_answer_manually(
            db,
            quiz_id=quiz_id,
            answer_id=answer_id,
            score=body.score,
            feedback=body.feedback,
            grader_id=current_user.user_id,
        )
    except NotFoundError as exc:
        raise _not_found("answer", answer_id) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    await db.refresh(answer)
    return ManualGradeRead.model_validate(answer)


@router.get(
    "/quizzes/{quiz_id}/overrides",
    response_model=list[QuizOverrideRead],
)
async def list_quiz_overrides(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizOverrideRead]:
    """List all user/group overrides for a quiz (Phase 5)."""
    from abridgeai.features.quizzes.queries import overrides as _ov_q  # noqa: PLC0415

    rows = await _ov_q.list_overrides(db, quiz_id)
    return [QuizOverrideRead.model_validate(r) for r in rows]


@router.post(
    "/quizzes/{quiz_id}/overrides",
    response_model=QuizOverrideRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_quiz_override(
    quiz_id: UUID,
    body: QuizOverrideIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizOverrideRead:
    """Create a per-user or per-group override for a quiz's timing/retake policy."""
    from abridgeai.features.quizzes.queries import overrides as _ov_q  # noqa: PLC0415

    try:
        row = await _ov_q.create_override(db, quiz_id, body.model_dump())
        await db.flush()
    except Exception as exc:  # noqa: BLE001
        # A duplicate (quiz, scope, user/group) trips a unique constraint.
        await db.rollback()
        raise _conflict("an override for this target already exists") from exc
    from abridgeai.features.quizzes.services import audit as _audit  # noqa: PLC0415

    await _audit.record_event(
        db,
        event_name="override_created",
        quiz_id=quiz_id,
        actor_user_id=current_user.user_id,
        subject_user_id=body.user_id,
        payload={"scope": body.scope, "override_id": str(row.id)},
    )
    await db.commit()
    await db.refresh(row)
    return QuizOverrideRead.model_validate(row)


@router.patch(
    "/quizzes/{quiz_id}/overrides/{override_id}",
    response_model=QuizOverrideRead,
)
async def update_quiz_override(
    quiz_id: UUID,
    override_id: UUID,
    body: QuizOverrideIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QuizOverrideRead:
    """Update an existing override row."""
    from abridgeai.features.quizzes.queries import overrides as _ov_q  # noqa: PLC0415

    row = await _ov_q.update_override(db, override_id, body.model_dump(), quiz_id=quiz_id)
    if row is None:
        raise _not_found("override", override_id)
    from abridgeai.features.quizzes.services import audit as _audit  # noqa: PLC0415

    await _audit.record_event(
        db,
        event_name="override_updated",
        quiz_id=quiz_id,
        actor_user_id=current_user.user_id,
        subject_user_id=row.user_id,
        payload={"scope": row.scope, "override_id": str(row.id)},
    )
    await db.commit()
    await db.refresh(row)
    return QuizOverrideRead.model_validate(row)


@router.delete(
    "/quizzes/{quiz_id}/overrides/{override_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_quiz_override(
    quiz_id: UUID,
    override_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete an override row (app-code delete, ondelete=NO ACTION convention)."""
    from abridgeai.features.quizzes.queries import overrides as _ov_q  # noqa: PLC0415

    deleted = await _ov_q.delete_override(db, override_id, quiz_id=quiz_id)
    if not deleted:
        raise _not_found("override", override_id)
    from abridgeai.features.quizzes.services import audit as _audit  # noqa: PLC0415

    await _audit.record_event(
        db,
        event_name="override_deleted",
        quiz_id=quiz_id,
        actor_user_id=getattr(current_user, "user_id", None),
        payload={"override_id": str(override_id)},
    )
    await db.commit()


class _FeedbackBandsBody(BaseModel):
    model_config = {"extra": "forbid"}
    bands: list[FeedbackBandIn] = Field(default_factory=list)


@router.get(
    "/quizzes/{quiz_id}/feedback-bands",
    response_model=list[FeedbackBandRead],
)
async def list_feedback_bands(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[FeedbackBandRead]:
    """List a quiz's grade-band feedback rows (Phase 8)."""
    del current_user
    from abridgeai.features.quizzes.services import feedback as _fb  # noqa: PLC0415

    rows = await _fb.list_bands(db, quiz_id)
    return [FeedbackBandRead.model_validate(r) for r in rows]


@router.put(
    "/quizzes/{quiz_id}/feedback-bands",
    response_model=list[FeedbackBandRead],
)
async def set_feedback_bands(
    quiz_id: UUID,
    body: _FeedbackBandsBody,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[FeedbackBandRead]:
    """Wholesale-replace a quiz's grade bands. Overlapping bands → 422."""
    del current_user
    from abridgeai.features.quizzes.services import feedback as _fb  # noqa: PLC0415

    try:
        rows = await _fb.set_feedback_bands(db, quiz_id=quiz_id, bands=body.bands)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    await db.commit()
    return [FeedbackBandRead.model_validate(r) for r in rows]


@router.get(
    "/quizzes/{quiz_id}/gradebook",
    response_model=list[QuizGradeRow],
)
async def get_quiz_gradebook(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[QuizGradeRow]:
    """List every student's materialised grade-of-record for a quiz (Phase 9)."""
    del current_user
    from abridgeai.features.quizzes.services import gradebook as _gb  # noqa: PLC0415

    rows = await _gb.list_quiz_grades(db, quiz_id)
    contacts = await _resolve_student_contacts(
        db, {row.student_id for row in rows}, include_avatar=True
    )
    return [
        QuizGradeRow(
            student_id=row.student_id,
            student_name=(
                contacts.get(row.student_id, (None, None, None))[0]
                or contacts.get(row.student_id, (None, None, None))[1]
            ),
            student_email=contacts.get(row.student_id, (None, None, None))[1],
            student_avatar_url=contacts.get(row.student_id, (None, None, None))[2],
            grade_percent=row.grade_percent,
            grade_points=row.grade_points,
            passed=row.passed,
            grading_method=row.grading_method,
            based_on_attempt_id=row.based_on_attempt_id,
            attempts_counted=row.attempts_counted,
        )
        for row in rows
    ]


@router.get("/quizzes/{quiz_id}/gradebook/export")
async def export_quiz_gradebook(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = "csv",
) -> Response:
    """Export the complete gradebook from the backend dataset."""
    del current_user
    if format not in ("csv", "xlsx"):
        raise _bad_request("format must be 'csv' or 'xlsx'")
    from abridgeai.features.quizzes.services import gradebook as _gb  # noqa: PLC0415

    rows = await _gb.list_quiz_grades(db, quiz_id)
    contacts = await _resolve_student_contacts(db, {row.student_id for row in rows})
    headers = ["Student", "Email", "Grade %", "Status", "Method", "Attempts"]
    table = [
        [
            contacts.get(row.student_id, (None, None, None))[0]
            or contacts.get(row.student_id, (None, None, None))[1]
            or str(row.student_id),
            contacts.get(row.student_id, (None, None, None))[1] or "",
            row.grade_percent,
            "Passed" if row.passed else "Failed",
            row.grading_method,
            row.attempts_counted,
        ]
        for row in rows
    ]
    return _report_download(headers, table, format, filename_stem=f"quiz-{quiz_id}-gradebook")


def _report_download(
    headers: list[str],
    rows: list[list[object]],
    fmt: str,
    *,
    filename_stem: str,
) -> Response:
    """Serialize a flattened report table to a CSV or XLSX download response."""
    from datetime import datetime  # noqa: PLC0415

    from abridgeai.features.quizzes.services import reports_export as _exp  # noqa: PLC0415

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    if fmt == "xlsx":
        content = _exp.build_xlsx(headers, rows)
        return Response(
            content=content,
            media_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            headers={
                "Content-Disposition": (f'attachment; filename="{filename_stem}-{stamp}.xlsx"')
            },
        )
    return StreamingResponse(
        _exp.stream_csv(headers, rows),
        media_type="text/csv",
        headers={"Content-Disposition": (f'attachment; filename="{filename_stem}-{stamp}.csv"')},
    )


@router.get("/quizzes/{quiz_id}/reports/responses")
async def get_responses_report(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = "json",
) -> object:
    """Per-student, per-question responses report (Phase 10). ?format=json|csv|xlsx."""
    del current_user
    from abridgeai.features.quizzes.services import reports as _rep  # noqa: PLC0415
    from abridgeai.features.quizzes.services import (  # noqa: PLC0415
        reports_export as _exp,
    )

    try:
        report = await _rep.build_responses_report(db, quiz_id)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    contacts = await _resolve_student_contacts(
        db, {row.student_id for row in report.rows}, include_avatar=True
    )
    report = report.model_copy(
        update={
            "rows": [
                row.model_copy(
                    update={
                        "student_email": contacts.get(row.student_id, (None, None, None))[1],
                        "student_avatar_url": contacts.get(row.student_id, (None, None, None))[2],
                    }
                )
                for row in report.rows
            ]
        }
    )
    if format in ("csv", "xlsx"):
        headers, rows = _exp.responses_to_table(report)
        return _report_download(headers, rows, format, filename_stem=f"quiz-{quiz_id}-responses")
    return report


@router.get("/quizzes/{quiz_id}/reports/statistics")
async def get_statistics_report(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = "json",
) -> object:
    """Per-question facility + discrimination statistics (Phase 10). ?format=json|csv|xlsx."""
    del current_user
    from abridgeai.features.quizzes.services import reports as _rep  # noqa: PLC0415
    from abridgeai.features.quizzes.services import (  # noqa: PLC0415
        reports_export as _exp,
    )

    try:
        report = await _rep.build_statistics_report(db, quiz_id)
    except NotFoundError as exc:
        raise _not_found("quiz", quiz_id) from exc
    if format in ("csv", "xlsx"):
        headers, rows = _exp.statistics_to_table(report)
        return _report_download(headers, rows, format, filename_stem=f"quiz-{quiz_id}-statistics")
    return report


class _AuditEventRow(BaseModel):
    model_config = {"from_attributes": True}
    id: UUID
    event_name: str
    quiz_id: UUID
    actor_user_id: UUID | None = None
    subject_attempt_id: UUID | None = None
    subject_question_id: UUID | None = None
    subject_user_id: UUID | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime


@router.get(
    "/quizzes/{quiz_id}/audit-events",
    response_model=list[_AuditEventRow],
)
async def list_quiz_audit_events(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 100,
) -> list[_AuditEventRow]:
    """Most-recent-first append-only audit trail for a quiz (Phase 13)."""
    del current_user
    from abridgeai.features.quizzes.services import audit as _audit  # noqa: PLC0415

    rows = await _audit.list_events_for_quiz(db, quiz_id, limit=limit)
    return [_AuditEventRow.model_validate(r) for r in rows]


class _ImportBody(BaseModel):
    model_config = {"extra": "forbid"}
    content: str
    format: str = "gift"  # gift | xml


@router.post("/quizzes/{quiz_id}/questions/import-file")
async def import_questions_from_file(
    quiz_id: UUID,
    body: _ImportBody,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Import questions from a Moodle GIFT or XML file (Phase 11).

    Additive + review-gated: imported questions land as ``pending``. A malformed
    file → 422 with no writes; per-question issues are returned as warnings.
    """
    from abridgeai.features.quizzes.services import quiz_io as _io  # noqa: PLC0415

    if body.format not in ("gift", "xml"):
        raise _bad_request("format must be 'gift' or 'xml'")
    try:
        result = await _io.import_questions_from_file(
            db,
            quiz_id=quiz_id,
            content=body.content,
            fmt=body.format,
            actor=current_user,
        )
    except ValueError as exc:
        # Parser error (malformed file) — abort, nothing written.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": "malformed_import_file", "message": str(exc)},
        ) from exc
    except AppError as exc:
        await db.rollback()
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return result


@router.get("/quizzes/{quiz_id}/questions/export")
async def export_quiz_questions(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = "gift",
) -> Response:
    """Export a quiz's questions to GIFT or Moodle XML (teacher-only download)."""
    del current_user
    from datetime import datetime  # noqa: PLC0415

    from abridgeai.features.quizzes.services import quiz_io as _io  # noqa: PLC0415

    if format not in ("gift", "xml"):
        raise _bad_request("format must be 'gift' or 'xml'")
    try:
        content = await _io.export_quiz_questions(db, quiz_id=quiz_id, fmt=format)
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    ext = "txt" if format == "gift" else "xml"
    media = "text/plain" if format == "gift" else "application/xml"
    return Response(
        content=content,
        media_type=media,
        headers={
            "Content-Disposition": (
                f'attachment; filename="quiz-{quiz_id}-questions-{stamp}.{ext}"'
            )
        },
    )


__all__ = [
    "get_arq_pool",
    "router",
]
