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
from abridgeai.features.quizzes.models import Quiz

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
    from sqlalchemy import text  # noqa: PLC0415

    rows = await db.execute(
        text(
            "SELECT id FROM quiz_questions "
            "WHERE quiz_id = :qid AND deleted_at IS NULL "
            "AND (expected_response_time_ms IS NULL OR expected_response_time_ms <= 0)"
        ),
        {"qid": str(quiz_id)},
    )
    missing = [UUID(str(row[0])) for row in rows.all()]
    if missing:
        raise QuizPublishValidationError(missing_t_exp_question_ids=missing)


async def bulk_set_expected_response_time(
    db: AsyncSession,
    quiz_id: UUID,
    items: list[tuple[UUID, int]],
    actor: CurrentUser,
) -> int:
    del actor
    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    if not items:
        return 0
    for question_id, ms in items:
        if ms <= 0:
            raise AppError(f"expected_response_time_ms must be > 0 for question {question_id}")

    from sqlalchemy import text  # noqa: PLC0415

    updated = 0
    for question_id, ms in items:
        result = await db.execute(
            text(
                "UPDATE quiz_questions SET expected_response_time_ms = :ms, "
                "updated_at = NOW() "
                "WHERE id = :qid AND quiz_id = :quiz_id AND deleted_at IS NULL"
            ),
            {"ms": ms, "qid": str(question_id), "quiz_id": str(quiz_id)},
        )
        updated += result.rowcount or 0  # type: ignore[attr-defined]
    await db.flush()
    return updated


__all__ = [
    "QuizPublishValidationError",
    "assert_t_exp_set_for_all_questions",
    "bulk_set_expected_response_time",
]
