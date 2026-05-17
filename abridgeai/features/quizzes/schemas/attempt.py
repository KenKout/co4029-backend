"""Student attempt DTOs for quizzes (T5.2).

Back the learner-facing attempt endpoints (T5.11):

* ``POST /quizzes/{id}/attempts`` — start a new attempt. Body:
  :class:`QuizAttemptStart`.
* ``POST /quizzes/{id}/attempts/{attempt_id}/submit`` — submit answers
  in bulk. Body: :class:`QuizAttemptSubmit`.
* ``GET  /quizzes/{id}/attempts/{attempt_id}`` — read the student's
  own attempt. Response: :class:`QuizAttemptRead`.

Reconciliation §C4 — :class:`QuizAttemptStart` carries an
``idempotency_key`` (mirrors ``QuizAttempt.idempotency_key`` UNIQUE
column) so retried POSTs on flaky networks resolve to the same attempt
row. Reconciliation §C1 — :class:`QuizAttemptSubmitAnswer.t_actual_ms`
mirrors the renamed ``QuizAttemptAnswer.t_actual_ms`` column (was
``response_time_ms`` in the legacy schema).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

QuizAttemptStatusLiteral = Literal[
    "in_progress",
    "submitted",
    "graded",
    "abandoned",
    "expired",
]


class QuizAttemptStart(BaseModel):
    """Body for ``POST /quizzes/{id}/attempts``.

    The idempotency key is opaque to the schema layer; the service
    layer (T5.11) enforces uniqueness at the DB level via the
    ``QuizAttempt.idempotency_key`` UNIQUE constraint.
    """

    model_config = ConfigDict(extra="forbid")

    quiz_id: UUID
    idempotency_key: UUID | None = None


class QuizAttemptSubmitAnswer(BaseModel):
    """One answer slice inside :class:`QuizAttemptSubmit`.

    All response fields are optional individually — different question
    types use different combinations:

    * multiple_choice / true_false → ``selected_option_ids``;
    * short_answer / fill_blank / code → ``text_answer``.

    Cross-field validity is enforced by the grading service (T5.11), not
    by the schema, because the constraint depends on the linked
    ``QuizQuestion.question_type``.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: UUID
    selected_option_ids: list[UUID] | None = None
    text_answer: str | None = None
    t_actual_ms: int | None = Field(default=None, ge=0)
    hint_used: bool = False


class QuizAttemptSubmit(BaseModel):
    """Body for ``POST /quizzes/{id}/attempts/{attempt_id}/submit``."""

    model_config = ConfigDict(extra="forbid")

    answers: list[QuizAttemptSubmitAnswer] = []


class QuizAttemptRead(BaseModel):
    """Student's view of their own attempt.

    Score columns surface only after grading completes; they are
    nullable in the ORM (``score_points`` / ``score_percent`` /
    ``passed`` / ``submitted_at`` / ``graded_at``).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    attempt_number: int
    status: QuizAttemptStatusLiteral
    started_at: datetime
    submitted_at: datetime | None = None
    graded_at: datetime | None = None
    time_taken_seconds: int | None = None
    score_points: Decimal | None = None
    score_percent: Decimal | None = None
    passed: bool | None = None
    total_questions: int | None = None
    correct_count: int | None = None


__all__ = [
    "QuizAttemptRead",
    "QuizAttemptStart",
    "QuizAttemptStatusLiteral",
    "QuizAttemptSubmit",
    "QuizAttemptSubmitAnswer",
]
