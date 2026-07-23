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

from abridgeai.features.quizzes.schemas.public import QuizForTakingPublic

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


class QuizAttemptReviewOption(BaseModel):
    """Option projection used for attempt review.

    Surfaces ``is_correct`` (which the student is normally not allowed to
    see at quiz-take time) because review happens *after* the attempt is
    submitted/graded — there's nothing left to cheat at.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    option_key: str
    option_text: str
    is_correct: bool
    position: int


class QuizAttemptReviewQuestion(BaseModel):
    """One question + the student's answer + the correct answer."""

    model_config = ConfigDict(from_attributes=True)

    question_id: UUID
    position: int
    question_type: str
    prompt_text: str
    explanation: str | None = None
    hint_text: str | None = None

    options: list[QuizAttemptReviewOption] = []

    # Student's submitted answer (None when they skipped the question).
    selected_option_id: UUID | None = None
    answer_text: str | None = None
    is_correct: bool = False
    points_awarded: Decimal = Decimal("0")
    hint_used: bool = False
    t_actual_ms: int | None = None


class QuizAttemptReviewRead(BaseModel):
    """Full review payload — attempt summary + per-question breakdown."""

    model_config = ConfigDict(from_attributes=True)

    attempt: QuizAttemptRead
    questions: list[QuizAttemptReviewQuestion]


class QuizAttemptIntegrityEvent(BaseModel):
    """One proctoring signal recorded during an attempt (teacher review).

    Projection of :class:`features.interviews.models.AssessmentIntegrityEvent`
    rows scoped to ``assessment_kind='quiz'``. Surfaced only in the
    teacher-facing attempt detail — never to the student.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_type: str
    severity: str
    metadata_json: dict = {}
    created_at: datetime


class QuizAttemptTeacherReview(BaseModel):
    """Teacher-facing detail for a single attempt.

    Backs ``GET /teacher/courses/{course_id}/quiz-attempts/{attempt_id}``.
    Combines the self-describing attempt summary (student name + quiz title),
    the full per-question review (prompt, options, the student's answer,
    correctness), and any integrity events captured during the take.
    """

    model_config = ConfigDict(from_attributes=True)

    attempt: QuizAttemptTeacherRead
    questions: list[QuizAttemptReviewQuestion]
    integrity_events: list[QuizAttemptIntegrityEvent] = []


class QuizAttemptProgressAnswer(BaseModel):
    """One saved answer for an in-progress attempt (resume payload).

    SECURITY INVARIANT: this is served WHILE the attempt is still in
    flight, so it MUST NOT expose ``is_correct`` / ``points_awarded`` (that
    would let a student probe correctness by saving an answer and reading it
    back). Only the student's own inputs are echoed — the same no-leak
    discipline as :class:`QuizQuestionOptionPublic`.
    """

    model_config = ConfigDict(from_attributes=True)

    question_id: UUID
    selected_option_id: UUID | None = None
    answer_text: str | None = None
    hint_used: bool = False
    t_actual_ms: int | None = None


class QuizAttemptProgressRead(BaseModel):
    """Resume payload for an in-progress attempt.

    Lets the client rehydrate its per-question state after an interruption
    (power-off, reload, critical notification) instead of wiping progress.
    Carries the attempt identity + timing anchor (``started_at`` so the
    client can resume a timed quiz from elapsed wall-clock, not from full),
    the full no-leak take payload (quiz + questions, so the client can
    re-render without re-POSTing a new attempt), and the answers saved so
    far. Correctness fields are omitted by :class:`QuizAttemptProgressAnswer`
    and the embedded ``take`` reuses the ``QuizQuestionPublic`` projection
    which already drops ``is_correct``.
    """

    model_config = ConfigDict(from_attributes=True)

    attempt_id: UUID
    quiz_id: UUID
    status: QuizAttemptStatusLiteral
    started_at: datetime
    take: QuizForTakingPublic
    answers: list[QuizAttemptProgressAnswer] = []


class QuizAttemptTeacherRead(BaseModel):
    """One row in a teacher's cross-quiz / cross-student attempts list.

    Teacher-only: joins in ``quiz_id`` -> title and student identity so a
    single row is self-describing without extra client-side lookups.
    Backs the course-wide "Assessments" tab and the per-student profile's
    quiz-attempts section (student-dashboard brainstorm, 2026-07-11).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    quiz_title: str
    student_id: UUID
    student_name: str | None = None
    attempt_number: int
    status: QuizAttemptStatusLiteral
    started_at: datetime
    submitted_at: datetime | None = None
    time_taken_seconds: int | None = None
    score_percent: Decimal | None = None
    passed: bool | None = None


__all__ = [
    "QuizAttemptRead",
    "QuizAttemptReviewOption",
    "QuizAttemptReviewQuestion",
    "QuizAttemptReviewRead",
    "QuizAttemptStart",
    "QuizAttemptStatusLiteral",
    "QuizAttemptSubmit",
    "QuizAttemptSubmitAnswer",
    "QuizAttemptTeacherRead",
]
