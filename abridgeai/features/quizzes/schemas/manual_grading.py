"""Manual-grading DTOs (Phase 4).

Back the teacher grading-queue endpoints:

* ``GET  /teacher/quizzes/{quiz_id}/needs-grading`` -> list[NeedsGradingRow]
* ``PATCH /teacher/quizzes/{quiz_id}/answers/{answer_id}/grade`` (ManualGradeIn)
  -> ManualGradeRead
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class NeedsGradingRow(BaseModel):
    """One open-response answer awaiting a human grader."""

    model_config = ConfigDict(from_attributes=True)

    answer_id: UUID
    attempt_id: UUID
    question_id: UUID
    student_id: UUID
    question_type: str
    prompt_text: str
    answer_text: str | None = None
    submitted_at: datetime | None = None


class ManualGradeIn(BaseModel):
    """Teacher's mark + feedback for one answer."""

    model_config = ConfigDict(extra="forbid")

    score: Decimal = Field(ge=0)
    feedback: str | None = None


class ManualGradeRead(BaseModel):
    """The graded answer after a manual mark is applied."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    attempt_id: UUID
    question_id: UUID
    manual_score: Decimal | None = None
    manual_feedback: str | None = None
    is_correct: bool
    points_awarded: Decimal
    needs_manual_grade: bool
    graded_by: UUID | None = None
    graded_at: datetime | None = None


__all__ = ["ManualGradeIn", "ManualGradeRead", "NeedsGradingRow"]
