"""Publish-gate validation for quiz authoring (T7.5.9).

Enforces the invariant that every ``QuizQuestion`` in a quiz has a
positive ``expected_response_time_ms`` before the quiz can be
published. Pulled into its own module to keep
:mod:`features.quizzes.services.authoring` under the feature LOC cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import Quiz, QuizQuestion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class QuizPublishValidationError(AppError):
    """Raised when the publish gate validation fails.

    Carries the list of question IDs missing a positive
    ``expected_response_time_ms`` so the frontend can highlight them in
    the authoring UI. The router maps this to HTTP 422 with structured
    detail.
    """

    def __init__(self, missing_t_exp_question_ids: list[UUID]) -> None:
        self.missing_t_exp_question_ids = missing_t_exp_question_ids
        super().__init__(
            f"Cannot publish quiz: {len(missing_t_exp_question_ids)} question(s) "
            f"missing expected_response_time_ms"
        )


async def assert_t_exp_set_for_all_questions(db: AsyncSession, quiz_id: UUID) -> None:
    """Assert every APPROVED question on ``quiz_id`` has positive ``expected_response_time_ms``.

    Only approved questions are ever served to students (see
    ``taking._load_quiz_questions_for_taking``), so the expected-time gate
    only needs to hold for them — a pending/rejected draft the student will
    never see must not block publishing the approved set.

    The ``deleted_at IS NULL`` predicate is applied automatically by the
    SoftDeleteMixin SELECT filter (``core.db.soft_delete``) — no need to
    repeat it here.
    """
    from sqlalchemy import or_, select  # noqa: PLC0415

    stmt = select(QuizQuestion.id).where(
        QuizQuestion.quiz_id == quiz_id,
        QuizQuestion.review_status == "approved",
        or_(
            QuizQuestion.expected_response_time_ms.is_(None),
            QuizQuestion.expected_response_time_ms <= 0,
        ),
    )
    result = await db.execute(stmt)
    missing = list(result.scalars().all())
    if missing:
        raise QuizPublishValidationError(missing_t_exp_question_ids=missing)


class QuizApprovalRequiredError(AppError):
    """Raised when publish is attempted with no approved questions.

    Partial publish (chosen product behaviour): a quiz publishes as soon as
    it has at least one approved question. Pending/rejected questions are
    retained as drafts and simply never served to students (see
    ``taking._load_quiz_questions_for_taking``), so a teacher can generate a
    surplus, approve the subset they want, publish, and keep the rest for
    later reuse. The only thing that blocks publish is having *zero* approved
    questions — an empty quiz. The router maps this to HTTP 422 with
    structured detail (code ``pending_review``).
    """

    def __init__(self, pending_question_ids: list[UUID]) -> None:
        # Kept for response-shape compatibility with the router/frontend;
        # empty here because the gate is now "needs ≥1 approved", not
        # "these specific questions are pending".
        self.pending_question_ids = pending_question_ids
        super().__init__("Cannot publish quiz: at least one question must be approved")


async def assert_all_questions_approved(db: AsyncSession, quiz_id: UUID) -> None:
    """Assert ``quiz_id`` has at least one ``review_status='approved'`` question.

    Human-in-the-loop gate for AI-generated content: students only ever see
    approved questions, so publishing is safe as long as at least one exists.
    Un-approved questions stay on the quiz as reusable drafts. Soft-deleted
    rows are auto-filtered by the SoftDeleteMixin SELECT filter.
    """
    from sqlalchemy import select  # noqa: PLC0415

    stmt = select(QuizQuestion.id).where(
        QuizQuestion.quiz_id == quiz_id,
        QuizQuestion.review_status == "approved",
    )
    result = await db.execute(stmt)
    approved = list(result.scalars().all())
    if not approved:
        raise QuizApprovalRequiredError(pending_question_ids=[])


async def bulk_set_expected_response_time(
    db: AsyncSession,
    quiz_id: UUID,
    items: list[tuple[UUID, int]],
    actor: CurrentUser,
) -> int:
    """Set ``expected_response_time_ms`` on each ``(question_id, ms)`` pair.

    Migrated from raw ``UPDATE`` to load-then-mutate (T8): we fetch each
    question via :func:`db.get`, verify it belongs to ``quiz_id``, and
    assign the column. Soft-deleted rows are auto-filtered. Triggers
    refresh ``updated_at`` (T1) and stamp ``updated_by`` (T3); we never
    touch them by hand.
    """
    del actor
    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    if not items:
        return 0
    for question_id, ms in items:
        if ms <= 0:
            raise AppError(f"expected_response_time_ms must be > 0 for question {question_id}")

    updated = 0
    for question_id, ms in items:
        question = await db.get(QuizQuestion, question_id)
        if question is None or question.quiz_id != quiz_id:
            continue
        question.expected_response_time_ms = ms
        updated += 1
    await db.flush()
    return updated


__all__ = [
    "QuizApprovalRequiredError",
    "QuizPublishValidationError",
    "assert_all_questions_approved",
    "assert_t_exp_set_for_all_questions",
    "bulk_set_expected_response_time",
]
