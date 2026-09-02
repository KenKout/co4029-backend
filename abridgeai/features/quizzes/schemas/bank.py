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

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from abridgeai.features.quizzes.schemas.authoring import QuizQuestionAuthoring
from abridgeai.features.quizzes.schemas.public import QuestionTypeLiteral


class QuestionBankEntry(BaseModel):
    """One row in a question-bank listing — question + parent context."""

    model_config = ConfigDict(from_attributes=True)

    question: QuizQuestionAuthoring
    quiz_id: UUID
    quiz_title: str
    module_id: UUID
    module_title: str
    course_id: UUID


class QuestionBankPage(BaseModel):
    """Cursor-paginated question-bank listing.

    ``next_cursor`` is opaque and round-trips through subsequent calls.
    Set when the page filled to ``limit`` (more rows may exist); ``None``
    otherwise. Reconciliation §A10/§D2: cursor pagination, not offset.
    """

    items: list[QuestionBankEntry]
    next_cursor: str | None = None


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


BankStatus = Literal["draft", "approved", "archived"]
Difficulty = Literal["easy", "medium", "hard"]
BloomLevel = Literal[
    "remember", "understand", "apply", "analyze", "evaluate", "create"
]


class QuizQuestionBankOptionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_key: str
    option_text: str
    is_correct: bool = False
    position: int
    option_format: str = "plain"
    grade_fraction: Decimal | None = None
    feedback_text: str | None = None
    feedback_format: str | None = None


class QuizQuestionBankOptionRead(QuizQuestionBankOptionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID


class QuizQuestionBankItemCreate(BaseModel):
    """Portable Quiz question content authored directly in the course bank."""

    model_config = ConfigDict(extra="forbid")

    question_type: QuestionTypeLiteral
    prompt_text: str
    hint_text: str | None = None
    explanation: str | None = None
    difficulty: Difficulty | None = None
    bloom_level: BloomLevel | None = None
    expected_response_time_ms: int | None = None
    expected_ef_ceiling: Decimal | None = None
    learning_outcome_id: UUID | None = None
    source_refs: list[Any] = []
    original_generated_payload: dict[str, Any] | None = None
    prompt_format: str = "plain"
    hint_format: str = "plain"
    explanation_format: str = "plain"
    single_answer: bool = True
    answer_numbering: str = "abc"
    numeric_answer: Decimal | None = None
    numeric_tolerance: Decimal | None = None
    match_pairs: list[dict[str, Any]] | None = None
    match_distractors: list[str] | None = None
    ordering_sequence: list[Any] | None = None
    category_id: UUID | None = None
    options: list[QuizQuestionBankOptionCreate] = []
    status: BankStatus = "draft"


class QuizQuestionBankItemUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_type: QuestionTypeLiteral | None = None
    prompt_text: str | None = None
    hint_text: str | None = None
    explanation: str | None = None
    difficulty: Difficulty | None = None
    bloom_level: BloomLevel | None = None
    expected_response_time_ms: int | None = None
    expected_ef_ceiling: Decimal | None = None
    learning_outcome_id: UUID | None = None
    prompt_format: str | None = None
    hint_format: str | None = None
    explanation_format: str | None = None
    single_answer: bool | None = None
    answer_numbering: str | None = None
    numeric_answer: Decimal | None = None
    numeric_tolerance: Decimal | None = None
    match_pairs: list[dict[str, Any]] | None = None
    match_distractors: list[str] | None = None
    ordering_sequence: list[Any] | None = None
    category_id: UUID | None = None
    options: list[QuizQuestionBankOptionCreate] | None = None


class QuizQuestionBankItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    course_id: UUID
    source_question_id: UUID | None = None
    status: BankStatus
    content_hash: str
    question_type: QuestionTypeLiteral
    prompt_text: str
    hint_text: str | None = None
    explanation: str | None = None
    difficulty: Difficulty | None = None
    bloom_level: BloomLevel | None = None
    expected_response_time_ms: int | None = None
    expected_ef_ceiling: Decimal | None = None
    learning_outcome_id: UUID | None = None
    source_refs: list[Any] = []
    original_generated_payload: dict[str, Any] | None = None
    prompt_format: str = "plain"
    hint_format: str = "plain"
    explanation_format: str = "plain"
    single_answer: bool = True
    answer_numbering: str = "abc"
    numeric_answer: Decimal | None = None
    numeric_tolerance: Decimal | None = None
    match_pairs: list[dict[str, Any]] | None = None
    match_distractors: list[str] | None = None
    ordering_sequence: list[Any] | None = None
    category_id: UUID | None = None
    options: list[QuizQuestionBankOptionRead] = []
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime


class QuizQuestionBankPage(BaseModel):
    items: list[QuizQuestionBankItemRead]
    next_cursor: str | None = None


class QuizQuestionBankCopyRequest(BaseModel):
    question_ids: list[UUID] = Field(min_length=1, max_length=100)


class QuizQuestionBankCopyResult(BaseModel):
    """Outcome of copying Quiz questions into the curated bank.

    ``skipped`` lists the SOURCE question ids whose content already has a
    live bank copy in this course — they are not copied again, and the
    caller can report them to the teacher instead of failing the batch.
    """

    created: list[QuizQuestionBankItemRead]
    skipped: list[UUID]


class QuizQuestionBankImportRequest(BaseModel):
    item_ids: list[UUID] = Field(min_length=1, max_length=100)


__all__ = [
    "QuestionBankEntry",
    "QuestionBankImportRequest",
    "QuestionBankPage",
    "QuizQuestionBankCopyRequest",
    "QuizQuestionBankCopyResult",
    "QuizQuestionBankImportRequest",
    "QuizQuestionBankItemCreate",
    "QuizQuestionBankItemRead",
    "QuizQuestionBankItemUpdate",
    "QuizQuestionBankOptionCreate",
    "QuizQuestionBankOptionRead",
    "QuizQuestionBankPage",
]
