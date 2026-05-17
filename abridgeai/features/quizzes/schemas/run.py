"""Generation / regeneration request + run-status DTOs for quizzes (T5.2).

These schemas back the teacher-facing generation endpoints (T5.10):

* ``POST /teacher/quizzes/{id}/generate`` — trigger a generation run.
  Body: :class:`QuizGenerationRequest`.
* ``GET  /teacher/quizzes/{id}/runs/{run_id}`` — poll status. Response:
  :class:`QuizGenerationRunRead`.
* ``POST /teacher/quizzes/{id}/questions/{q_id}/regenerate`` —
  re-roll one question. Body: :class:`QuestionRegenerationRequest`.

Reconciliation §A13: the generation-run row lives in
``backend/app/models/ai_processing.py`` (will be ported to
``features/ai/models.py`` in a later T5.x). T5.2 only declares the
schema shape; the generation pipeline itself remains untouched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

GenerationModeLiteral = Literal["full", "coverage", "manual", "regenerate"]
GenerationRunStatusLiteral = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]


class QuizGenerationRequest(BaseModel):
    """Body for ``POST /teacher/quizzes/{id}/generate``.

    ``regenerate_question_id`` is required when ``mode == "regenerate"``;
    the service layer (T5.10) enforces the cross-field constraint.
    """

    model_config = ConfigDict(extra="forbid")

    mode: GenerationModeLiteral
    target_count: int | None = Field(default=None, ge=1, le=200)
    focus_topics: list[str] = []
    source_lessons: list[UUID] = []
    regenerate_question_id: UUID | None = None


class QuizGenerationRunRead(BaseModel):
    """Status-poll projection of a quiz generation run.

    Service layer (T5.10) joins the
    ``backend/app/models/ai_processing.py:GenerationRun`` row with the
    quiz / pipeline-run linkage to populate this DTO.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    status: GenerationRunStatusLiteral
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    pipeline_run_id: UUID | None = None


class QuestionRegenerationRequest(BaseModel):
    """Body for ``POST /teacher/quizzes/{id}/questions/{q_id}/regenerate``."""

    model_config = ConfigDict(extra="forbid")

    question_id: UUID


__all__ = [
    "GenerationModeLiteral",
    "GenerationRunStatusLiteral",
    "QuestionRegenerationRequest",
    "QuizGenerationRequest",
    "QuizGenerationRunRead",
]
