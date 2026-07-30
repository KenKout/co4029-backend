"""Report services (Phase 10): responses + statistics reports.

Builds the two teacher reports from live question data + completed attempts:

* Responses — one row per student answer (what they answered vs the correct
  answer). Uses the LIVE question definition (revision payloads are opaque blobs
  in this deployment, so the live definition is authoritative).
* Statistics — per-question facility index (% correct, reused from the existing
  analytics breakdown) + discrimination index (point-biserial, Task 10.3).

Layering: owns its own DB reads (precedent: services/taking.py), so routers call
here rather than touching queries.
"""

from __future__ import annotations

from decimal import Decimal
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


async def build_responses_report(db: AsyncSession, quiz_id: UUID) -> ResponsesReportRead:
    """One row per (student attempt, approved question)."""
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

    # Resolve student names via the same batch resolver the results endpoint uses.
    from abridgeai.features.quizzes.routers.authoring import (  # noqa: PLC0415
        _resolve_student_names,
    )

    names = await _resolve_student_names(db, {a.student_id for a in attempts})

    rows: list[ResponsesReportRow] = []
    for attempt in attempts:
        for question in questions:
            ans = answers_by_key.get((attempt.id, question.id))
            opts = options_by_q.get(question.id, [])
            rows.append(
                ResponsesReportRow(
                    student_id=attempt.student_id,
                    student_name=names.get(attempt.student_id),
                    attempt_number=attempt.attempt_number,
                    question_id=question.id,
                    question_position=question.position,
                    prompt_text=question.prompt_text,
                    question_type=question.question_type,
                    student_answer=_student_answer_text(ans, options_by_id),
                    correct_answer=_correct_answer_text(question, opts),
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
    # question_id -> {attempt_id: is_correct}
    by_question: dict[UUID, dict[UUID, bool]] = {}
    for a in answers:
        by_question.setdefault(a.question_id, {})[a.attempt_id] = bool(a.is_correct)

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
                prompt_text=q.prompt_text,
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
