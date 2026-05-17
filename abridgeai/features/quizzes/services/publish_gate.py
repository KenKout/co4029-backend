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
    """Assert every live question on ``quiz_id`` has positive ``expected_response_time_ms``.

    The ``deleted_at IS NULL`` predicate is applied automatically by the
    SoftDeleteMixin SELECT filter (``core.db.soft_delete``) — no need to
    repeat it here.
    """
    from sqlalchemy import or_, select  # noqa: PLC0415

    stmt = select(QuizQuestion.id).where(
        QuizQuestion.quiz_id == quiz_id,
        or_(
            QuizQuestion.expected_response_time_ms.is_(None),
            QuizQuestion.expected_response_time_ms <= 0,
        ),
    )
    result = await db.execute(stmt)
    missing = list(result.scalars().all())
    if missing:
        raise QuizPublishValidationError(missing_t_exp_question_ids=missing)


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
    "QuizPublishValidationError",
    "assert_t_exp_set_for_all_questions",
    "bulk_set_expected_response_time",
]
