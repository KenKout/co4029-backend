"""Read-only attempt projection service (history / review / resume).

Split out of :mod:`services.taking` so the attempt lifecycle (start →
answer → submit) and the attempt READ surface (history, post-submit
review, in-flight resume) stay under the feature's god-file cap. The
three functions are pure projections over rows the learner already owns —
no mutation, no gates other than ownership.

Cross-module helpers that the write path also needs
(``_load_quiz_questions_for_taking``, ``_renumber_display_positions``)
stay in :mod:`services.taking` and are imported lazily here to avoid a
module-level import cycle (taking re-exports both functions).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
)
from abridgeai.features.quizzes.queries import published as published_queries
from abridgeai.features.quizzes.schemas.attempt import (
    QuizAttemptProgressAnswer,
    QuizAttemptProgressRead,
    QuizAttemptRead,
    QuizAttemptReviewOption,
    QuizAttemptReviewQuestion,
    QuizAttemptReviewRead,
)
from abridgeai.features.quizzes.schemas.public import (
    QuizForTakingPublic,
    QuizPublic,
    QuizQuestionPublic,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def project_attempt_summary(
    db: AsyncSession,
    attempt: QuizAttempt,
    *,
    now: datetime | None = None,
) -> QuizAttemptRead:
    """Return a learner-safe attempt summary for the active review window."""
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.quizzes.services.review_visibility import (  # noqa: PLC0415
        resolve_review_visibility,
    )

    pending = (
        await db.execute(
            select(QuizAttemptAnswer.id).where(
                QuizAttemptAnswer.attempt_id == attempt.id,
                QuizAttemptAnswer.needs_manual_grade.is_(True),
            )
        )
    ).first() is not None
    payload = QuizAttemptRead.model_validate(attempt).model_copy(
        update={"grading_pending": pending}
    )
    if attempt.status == "in_progress":
        return payload

    quiz = await db.get(Quiz, attempt.quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {attempt.quiz_id} not found")
    visibility = resolve_review_visibility(quiz, attempt, now or utcnow())
    if pending or not visibility.show_score:
        payload.score_points = None
        payload.score_percent = None
        payload.passed = None
        payload.correct_count = None
    return payload


async def get_attempt_history(
    db: AsyncSession,
    quiz_id: UUID,
    actor: CurrentUser,
) -> list[QuizAttemptRead]:
    """Return every attempt the calling student has against ``quiz_id``."""
    from sqlalchemy import select  # noqa: PLC0415

    attempts = (
        (
            await db.execute(
                select(QuizAttempt)
                .where(
                    QuizAttempt.quiz_id == quiz_id,
                    QuizAttempt.student_id == actor.user_id,
                )
                .order_by(QuizAttempt.attempt_number)
            )
        )
        .scalars()
        .all()
    )
    return [await project_attempt_summary(db, attempt) for attempt in attempts]


async def get_attempt_review(
    db: AsyncSession,
    *,
    attempt_id: UUID,
    actor: CurrentUser,
) -> QuizAttemptReviewRead | None:
    """Project a submitted attempt + its answers + correct options.

    Returns ``None`` (router → 404) when the attempt doesn't belong to the
    caller or hasn't been submitted yet — review surface only opens after
    the student locks in their answers.
    """
    attempt = await published_queries.get_attempt_for_review(
        db, attempt_id=attempt_id, user_id=actor.user_id
    )
    if attempt is None:
        return None

    answers_by_question: dict[UUID, QuizAttemptAnswer] = {a.question_id: a for a in attempt.answers}

    questions_with_options = await published_queries.list_quiz_questions_with_options(
        db, attempt.quiz_id
    )

    # Phase 2: resolve the teacher-configured review-visibility flags for the
    # active time-window (immediately_after / later_while_open / after_close) and
    # mask the payload server-side so a hidden field never leaves the service.
    from abridgeai.features.quizzes.schemas.attempt import (  # noqa: PLC0415
        ReviewVisibilityFlags,
    )
    from abridgeai.features.quizzes.services.review_visibility import (  # noqa: PLC0415
        resolve_review_visibility,
    )

    quiz = await db.get(Quiz, attempt.quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {attempt.quiz_id} not found")
    vis = resolve_review_visibility(quiz, attempt, utcnow())
    attempt_read = await project_attempt_summary(db, attempt)
    if attempt_read.grading_pending:
        vis = vis.model_copy(
            update={
                "show_score": False,
                "show_correctness": False,
                "show_points": False,
            }
        )

    review_questions: list[QuizAttemptReviewQuestion] = []
    for question, options in questions_with_options:
        ans = answers_by_question.get(question.id)
        # Options carry is_correct; strip the correct-answer signal when hidden.
        review_options = [QuizAttemptReviewOption.model_validate(opt) for opt in options]
        if not vis.show_correct_answers:
            for opt in review_options:
                opt.is_correct = False
        # Structured correct answers (matching pairs / ordering sequence) are
        # disclosed under the same visibility flag as option correctness.
        show_correct = vis.show_correct_answers
        matching_correct = (
            [dict(p) for p in (question.match_pairs or [])]
            if show_correct and question.match_pairs
            else None
        )
        ordering_correct = (
            list(question.ordering_sequence)
            if show_correct and question.ordering_sequence
            else None
        )
        # fill_blank / short_answer carry their answer in
        # original_generated_payload.correct_answer (a positional list for
        # fill_blank, a plain string for short_answer).
        payload = question.original_generated_payload or {}
        fill_blank_correct: list[str] | None = None
        short_answer_correct: str | None = None
        if show_correct and isinstance(payload, dict):
            correct = payload.get("correct_answer")
            if question.question_type == "fill_blank" and isinstance(correct, list):
                fill_blank_correct = [str(b) for b in correct if isinstance(b, str)]
            elif question.question_type == "short_answer" and isinstance(correct, str):
                short_answer_correct = correct
        review_questions.append(
            QuizAttemptReviewQuestion(
                question_id=question.id,
                position=question.position,
                question_type=question.question_type,
                prompt_text=question.prompt_text,
                explanation=question.explanation if vis.show_explanation else None,
                hint_text=question.hint_text,
                options=review_options,
                selected_option_id=ans.selected_option_id if ans else None,
                answer_text=ans.answer_text if ans else None,
                is_correct=(ans.is_correct if ans else False) if vis.show_correctness else False,
                points_awarded=(ans.points_awarded if ans else Decimal("0"))
                if vis.show_points
                else Decimal("0"),
                hint_used=ans.hint_used if ans else False,
                t_actual_ms=ans.t_actual_ms if ans else None,
                matching_correct=matching_correct,
                ordering_correct=ordering_correct,
                fill_blank_correct=fill_blank_correct,
                short_answer_correct=short_answer_correct,
            )
        )

    if not vis.show_score:
        attempt_read.score_points = None
        attempt_read.score_percent = None
        attempt_read.passed = None
        attempt_read.correct_count = None

    # Phase 8: attach the matched overall grade-band feedback, but only when the
    # score is visible (feedback is a review-time disclosure like the score).
    overall_text: str | None = None
    overall_format: str | None = None
    if vis.show_score:
        from abridgeai.features.quizzes.services import feedback as _fb  # noqa: PLC0415

        band = await _fb.select_overall_feedback(
            db, quiz_id=attempt.quiz_id, score_percent=attempt.score_percent
        )
        if band is not None:
            overall_text = band.feedback_text
            overall_format = band.feedback_format

    return QuizAttemptReviewRead(
        attempt=attempt_read,
        questions=review_questions,
        visibility=ReviewVisibilityFlags(
            show_score=vis.show_score,
            show_correctness=vis.show_correctness,
            show_correct_answers=vis.show_correct_answers,
            show_explanation=vis.show_explanation,
            show_points=vis.show_points,
        ),
        overall_feedback_text=overall_text,
        overall_feedback_format=overall_format,
    )


async def get_attempt_progress(
    db: AsyncSession,
    *,
    attempt_id: UUID,
    actor: CurrentUser,
) -> QuizAttemptProgressRead | None:
    """Project an in-progress attempt into a no-leak resume payload.

    Returns ``None`` (router → 404) when the attempt doesn't belong to the
    caller or is no longer in flight. Only the student's own inputs are
    echoed — correctness / points are dropped by
    :class:`QuizAttemptProgressAnswer` because the attempt is still open
    and a leak here would let a student probe answers by save-and-read.
    """
    attempt = await published_queries.get_in_progress_attempt(
        db, attempt_id=attempt_id, user_id=actor.user_id
    )
    if attempt is None:
        return None

    # Rebuild the no-leak take payload so the client can re-render the quiz
    # without POSTing a fresh attempt (which would create a duplicate). We
    # load the quiz row directly (NOT via get_quiz_for_taking, which would
    # re-run the retake/cooldown gate and could 409/429 a legitimate resume)
    # and project through the same QuizQuestionPublic schema that drops
    # is_correct. All published questions are shown; the saved answers are a
    # subset keyed by question_id.
    from abridgeai.features.quizzes.services.taking import (  # noqa: PLC0415
        _load_quiz_questions_for_taking,
        _renumber_display_positions,
        _require_quiz,
    )

    quiz = await _require_quiz(db, attempt.quiz_id)
    questions = await _load_quiz_questions_for_taking(db, attempt.quiz_id)
    # Re-apply the per-attempt shuffle layout persisted at start time so a
    # resume (refresh / re-open) shows the SAME shuffled order the student saw,
    # not the authored order. Without this the resume payload leaks the
    # canonical sequence and disagrees with the order answers were given in.
    if attempt.layout:
        from abridgeai.features.quizzes.services.shuffle import apply_layout  # noqa: PLC0415

        questions = apply_layout(questions, attempt.layout)
    public_questions = [QuizQuestionPublic.model_validate(q) for q in questions]
    _renumber_display_positions(public_questions)
    take_payload = QuizForTakingPublic(
        quiz=QuizPublic.model_validate(quiz),
        questions=public_questions,
    )

    return QuizAttemptProgressRead.model_validate(
        {
            "attempt_id": attempt.id,
            "quiz_id": attempt.quiz_id,
            "status": attempt.status,
            "started_at": attempt.started_at,
            "take": take_payload,
            "answers": [
                QuizAttemptProgressAnswer(
                    question_id=ans.question_id,
                    selected_option_id=ans.selected_option_id,
                    answer_text=ans.answer_text,
                    hint_used=ans.hint_used,
                    t_actual_ms=ans.t_actual_ms,
                )
                for ans in attempt.answers
            ],
        }
    )


__all__ = [
    "get_attempt_history",
    "get_attempt_progress",
    "get_attempt_review",
    "project_attempt_summary",
]
