"""Report services (Phase 10): responses + statistics reports.

Builds the two teacher reports from completed attempts and their snapshots:

* Responses — one row per question served in each attempt. Prompt and answer
  key come from the revision used to grade that answer whenever available.
* Statistics — per-question facility index (% correct, reused from the existing
  analytics breakdown) + discrimination index (point-biserial, Task 10.3).

Layering: owns its own DB reads (precedent: services/taking.py), so routers call
here rather than touching queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
)
from abridgeai.features.quizzes.schemas.reports import (
    ResponsesReportRead,
    ResponsesReportRow,
    StatisticsReportRead,
    StatisticsReportRow,
)
from abridgeai.features.quizzes.services.statistics import point_biserial

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _require_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz:
    quiz = (await db.execute(select(Quiz).where(Quiz.id == quiz_id))).scalar_one_or_none()
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    return quiz


def _correct_answer_text(question: QuizQuestion, options: list[QuizQuestionOption]) -> str:
    """Render a human-readable correct answer from the live question definition."""
    if question.question_type in {"multiple_choice", "true_false"}:
        correct = [o.option_text for o in options if o.is_correct]
        return " ; ".join(correct) if correct else "(none)"
    gen = question.original_generated_payload or {}
    ca = gen.get("correct_answer")
    if isinstance(ca, list):
        return " | ".join(str(x) for x in ca)
    return str(ca) if ca is not None else "(manual)"


def _student_answer_text(
    answer: QuizAttemptAnswer | None,
    options_by_id: dict[UUID, QuizQuestionOption],
) -> str:
    if answer is None:
        return "(no answer)"
    if answer.selected_option_id is not None:
        opt = options_by_id.get(answer.selected_option_id)
        return opt.option_text if opt is not None else "(unknown option)"
    return answer.answer_text or "(no answer)"


def _live_snapshot(
    question: QuizQuestion, options: list[QuizQuestionOption]
) -> dict[str, object]:
    """Portable fallback for legacy answers that pre-date revision pinning."""
    return {
        "prompt_text": question.prompt_text,
        "question_type": question.question_type,
        "correct_answer": (question.original_generated_payload or {}).get("correct_answer"),
        "options": [
            {
                "option_key": getattr(option, "option_key", None),
                "option_text": option.option_text,
                "is_correct": option.is_correct,
            }
            for option in options
        ],
    }


def _correct_answer_from_snapshot(snapshot: dict[str, object]) -> str:
    if snapshot.get("question_type") in {"multiple_choice", "true_false"}:
        options = snapshot.get("options") or []
        correct = [
            str(option.get("option_text"))
            for option in options
            if isinstance(option, dict) and option.get("is_correct")
        ]
        return " ; ".join(correct) if correct else "(none)"
    answer = snapshot.get("correct_answer")
    if isinstance(answer, list):
        return " | ".join(str(value) for value in answer)
    return str(answer) if answer is not None else "(manual)"


def _student_answer_from_snapshot(
    answer: QuizAttemptAnswer | None,
    snapshot: dict[str, object],
    live_options_by_id: dict[UUID, QuizQuestionOption],
) -> str:
    if answer is None:
        return "(no answer)"
    selected_ids = list(getattr(answer, "selected_option_ids", None) or [])
    if answer.selected_option_id is not None:
        selected_ids = [answer.selected_option_id]
    if selected_ids:
        snapshot_by_key = {
            str(option.get("option_key")): str(option.get("option_text"))
            for option in (snapshot.get("options") or [])
            if isinstance(option, dict)
        }
        rendered: list[str] = []
        for option_id in selected_ids:
            try:
                normalized_id = option_id if isinstance(option_id, UUID) else UUID(str(option_id))
            except (TypeError, ValueError):
                rendered.append("(unknown option)")
                continue
            live_option = live_options_by_id.get(normalized_id)
            option_key = getattr(live_option, "option_key", None)
            rendered.append(
                snapshot_by_key.get(str(option_key), getattr(live_option, "option_text", None))
                or "(unknown option)"
            )
        return " ; ".join(rendered)
    return answer.answer_text or "(no answer)"


def _revision_snapshots(
    revisions: list[QuizQuestionRevision],
) -> dict[UUID, dict[str, object]]:
    """Fold legacy patch revisions into complete snapshots per revision id."""
    state_by_question: dict[UUID, dict[str, object]] = {}
    snapshots: dict[UUID, dict[str, object]] = {}
    for revision in sorted(revisions, key=lambda row: (str(row.question_id), row.revision_no)):
        state = state_by_question.setdefault(revision.question_id, {})
        state.update(dict(revision.payload_json or {}))
        snapshots[revision.id] = dict(state)
    return snapshots


async def build_responses_report(db: AsyncSession, quiz_id: UUID) -> ResponsesReportRead:
    """One row per question captured in a completed attempt's layout."""
    quiz = await _require_quiz(db, quiz_id)

    questions = list(
        (
            await db.execute(
                select(QuizQuestion)
                .where(
                    QuizQuestion.quiz_id == quiz.id,
                    QuizQuestion.review_status == "approved",
                )
                .order_by(QuizQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    q_ids = [q.id for q in questions]
    options = list(
        (
            await db.execute(
                select(QuizQuestionOption).where(QuizQuestionOption.question_id.in_(q_ids))
            )
        )
        .scalars()
        .all()
    ) if q_ids else []
    options_by_q: dict[UUID, list[QuizQuestionOption]] = {}
    options_by_id: dict[UUID, QuizQuestionOption] = {}
    for o in options:
        options_by_q.setdefault(o.question_id, []).append(o)
        options_by_id[o.id] = o

    attempts = list(
        (
            await db.execute(
                select(QuizAttempt)
                .where(
                    QuizAttempt.quiz_id == quiz.id,
                    QuizAttempt.status.in_(["submitted", "graded"]),
                )
                .order_by(QuizAttempt.student_id, QuizAttempt.attempt_number)
            )
        )
        .scalars()
        .all()
    )
    attempt_ids = [a.id for a in attempts]
    answers = list(
        (
            await db.execute(
                select(QuizAttemptAnswer).where(
                    QuizAttemptAnswer.attempt_id.in_(attempt_ids)
                )
            )
        )
        .scalars()
        .all()
    ) if attempt_ids else []
    answers_by_key: dict[tuple[UUID, UUID], QuizAttemptAnswer] = {
        (a.attempt_id, a.question_id): a for a in answers
    }
    revision_ids = {
        revision_id
        for answer in answers
        if (revision_id := getattr(answer, "graded_revision_id", None)) is not None
    }
    # A stored revision may be a legacy PATCH rather than a full snapshot. Load
    # the question's complete revision chain so the fold can reconstruct it.
    revision_question_ids = {
        answer.question_id
        for answer in answers
        if getattr(answer, "graded_revision_id", None) is not None
    }
    revisions = list(
        (
            await db.execute(
                select(QuizQuestionRevision)
                .where(QuizQuestionRevision.question_id.in_(revision_question_ids))
                .order_by(QuizQuestionRevision.question_id, QuizQuestionRevision.revision_no)
            )
        )
        .scalars()
        .all()
    ) if revision_ids else []
    snapshots_by_revision = _revision_snapshots(revisions)
    questions_by_id = {question.id: question for question in questions}

    # Resolve student names via the same batch resolver the results endpoint uses.
    from abridgeai.features.quizzes.routers.authoring import (  # noqa: PLC0415
        _resolve_student_names,
    )

    names = await _resolve_student_names(db, {a.student_id for a in attempts})

    rows: list[ResponsesReportRow] = []
    for attempt in attempts:
        layout_order = list((getattr(attempt, "layout", None) or {}).get("question_order", []))
        attempt_question_ids = (
            [UUID(str(value)) for value in layout_order]
            if layout_order
            else [question.id for question in questions]
        )
        for layout_position, question_id in enumerate(attempt_question_ids, start=1):
            question = questions_by_id.get(question_id)
            ans = answers_by_key.get((attempt.id, question_id))
            opts = options_by_q.get(question_id, [])
            live_snapshot = _live_snapshot(question, opts) if question is not None else {}
            snapshot = dict(live_snapshot)
            revision_id = getattr(ans, "graded_revision_id", None) if ans else None
            if revision_id in snapshots_by_revision:
                snapshot.update(snapshots_by_revision[revision_id])
            rows.append(
                ResponsesReportRow(
                    student_id=attempt.student_id,
                    student_name=names.get(attempt.student_id),
                    attempt_number=attempt.attempt_number,
                    question_id=question_id,
                    question_position=(
                        question.position if question is not None else layout_position
                    ),
                    prompt_text=str(snapshot.get("prompt_text") or "(question unavailable)"),
                    question_type=str(snapshot.get("question_type") or "unknown"),
                    student_answer=_student_answer_from_snapshot(ans, snapshot, options_by_id),
                    correct_answer=_correct_answer_from_snapshot(snapshot),
                    is_correct=bool(ans.is_correct) if ans else False,
                    points_awarded=float(ans.points_awarded) if ans else 0.0,
                )
            )
    return ResponsesReportRead(quiz_id=quiz.id, rows=rows)


async def build_statistics_report(db: AsyncSession, quiz_id: UUID) -> StatisticsReportRead:
    """Per-question facility + discrimination over the completed-attempt set."""
    quiz = await _require_quiz(db, quiz_id)

    questions = list(
        (
            await db.execute(
                select(QuizQuestion)
                .where(
                    QuizQuestion.quiz_id == quiz.id,
                    QuizQuestion.review_status == "approved",
                )
                .order_by(QuizQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    attempts = list(
        (
            await db.execute(
                select(QuizAttempt).where(
                    QuizAttempt.quiz_id == quiz.id,
                    QuizAttempt.status.in_(["submitted", "graded"]),
                )
            )
        )
        .scalars()
        .all()
    )
    attempt_ids = [a.id for a in attempts]
    total_by_attempt: dict[UUID, float] = {
        a.id: float(a.score_points) if a.score_points is not None else 0.0 for a in attempts
    }
    answers = list(
        (
            await db.execute(
                select(QuizAttemptAnswer).where(
                    QuizAttemptAnswer.attempt_id.in_(attempt_ids)
                )
            )
        )
        .scalars()
        .all()
    ) if attempt_ids else []
    graded_revision_ids = {
        revision_id
        for answer in answers
        if (revision_id := getattr(answer, "graded_revision_id", None)) is not None
    }
    revisions = list(
        (
            await db.execute(
                select(QuizQuestionRevision).where(
                    QuizQuestionRevision.question_id.in_({answer.question_id for answer in answers})
                )
            )
        )
        .scalars()
        .all()
    ) if graded_revision_ids else []
    snapshots = _revision_snapshots(revisions)
    # question_id -> {attempt_id: is_correct}
    by_question: dict[UUID, dict[UUID, bool]] = {}
    snapshot_by_question: dict[UUID, dict[str, object]] = {}
    for a in answers:
        by_question.setdefault(a.question_id, {})[a.attempt_id] = bool(a.is_correct)
        revision_id = getattr(a, "graded_revision_id", None)
        if revision_id in snapshots:
            snapshot_by_question.setdefault(a.question_id, snapshots[revision_id])

    rows: list[StatisticsReportRow] = []
    for q in questions:
        per_attempt = by_question.get(q.id, {})
        correct_flags: list[bool] = []
        totals: list[float] = []
        for aid, is_correct in per_attempt.items():
            correct_flags.append(is_correct)
            totals.append(total_by_attempt.get(aid, 0.0))
        answered = len(correct_flags)
        correct = sum(1 for c in correct_flags if c)
        facility = (correct / answered) if answered else None
        disc, note = point_biserial(correct_flags, totals)
        rows.append(
            StatisticsReportRow(
                question_id=q.id,
                question_position=q.position,
                prompt_text=str(
                    snapshot_by_question.get(q.id, {}).get("prompt_text") or q.prompt_text
                ),
                answered_count=answered,
                correct_count=correct,
                facility_index=facility,
                discrimination_index=disc,
                discrimination_note=note,
            )
        )
    return StatisticsReportRead(
        quiz_id=quiz.id, attempts_analyzed=len(attempts), rows=rows
    )


__all__ = ["build_responses_report", "build_statistics_report"]
