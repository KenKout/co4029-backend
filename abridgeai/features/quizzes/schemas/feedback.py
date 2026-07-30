"""Grade-band feedback DTOs (Phase 8).

Back the teacher grade-band CRUD endpoints and the student review-time
overall-feedback surface:

* ``GET /teacher/quizzes/{quiz_id}/feedback-bands``  -> list[FeedbackBandRead]
* ``PUT /teacher/quizzes/{quiz_id}/feedback-bands``  (FeedbackBandsIn) -> list[FeedbackBandRead]

Per-option feedback rides the existing question-update path and appears on the
review option schema — never on the public/take schema (security invariant).
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class FeedbackBandIn(BaseModel):
    """One grade band on write. ``[min_grade, max_grade)`` half-open, percent-space."""

    model_config = ConfigDict(extra="forbid")

    min_grade: Decimal
    max_grade: Decimal
    feedback_text: str
    feedback_format: str = "markdown"


class FeedbackBandsIn(BaseModel):
    """Wholesale replace payload for a quiz's grade bands."""

    model_config = ConfigDict(extra="forbid")

    bands: list[FeedbackBandIn] = []


class FeedbackBandRead(BaseModel):
    """A grade band as returned to the teacher."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    min_grade: Decimal
    max_grade: Decimal
    feedback_text: str
    feedback_format: str


class OverallFeedbackRead(BaseModel):
    """The single band matched for an attempt's score_percent (review-time)."""

    model_config = ConfigDict(from_attributes=True)

    feedback_text: str
    feedback_format: str


class QuizGradeRow(BaseModel):
    """One student's materialised grade-of-record for a quiz (Phase 9)."""

    model_config = ConfigDict(from_attributes=True)

    student_id: UUID
    grade_percent: Decimal
    grade_points: Decimal
    passed: bool
    grading_method: str
    based_on_attempt_id: UUID | None = None
    attempts_counted: int


__all__ = [
    "FeedbackBandIn",
    "FeedbackBandRead",
    "OverallFeedbackRead",
    "QuizGradeRow",
]
