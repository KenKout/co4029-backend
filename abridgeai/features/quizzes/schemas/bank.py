"""Question bank DTOs (cross-quiz reuse, T5-bank).

The question bank lets teachers browse every authored quiz question
across the courses they manage and import a selection into a target
quiz. The DTOs here are projection-only — write-side endpoints accept
plain ``list[UUID]`` payloads.

Why a bank entry vs reusing :class:`QuizQuestionAuthoring`?
  The browse view also needs the parent quiz title + module/course
  identifiers for breadcrumbs and filters. Carrying those on the
  authoring schema would couple it to the bank surface; instead we
  compose a thin wrapper that embeds the question payload alongside
  the parent context.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from abridgeai.features.quizzes.schemas.authoring import QuizQuestionAuthoring


class QuestionBankEntry(BaseModel):
    """One row in a question-bank listing — question + parent context."""

    model_config = ConfigDict(from_attributes=True)

    question: QuizQuestionAuthoring
    quiz_id: UUID
    quiz_title: str
    module_id: UUID
    module_title: str
    course_id: UUID


class QuestionBankImportRequest(BaseModel):
    """Body for ``POST /teacher/quizzes/{quiz_id}/questions/import``."""

    source_question_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "IDs of bank questions to clone into the target quiz. Each "
            "source must be authored under a course the actor can edit."
        ),
    )


__all__ = [
    "QuestionBankEntry",
    "QuestionBankImportRequest",
]
