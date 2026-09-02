"""Interview question-bank DTOs (§QBank-1).

Extracted from ``schemas.authoring`` (2026-09-01) to keep that module under
the interviews LOC ratchet. Re-exported through ``schemas.__init__`` so
downstream imports keep working.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from abridgeai.features.interviews.schemas.authoring import (
    DifficultyLiteral,
    InterviewQuestionAuthoring,
    QuestionTypeLiteral,
)


class InterviewQuestionBankItemCreate(BaseModel):
    """Body for ``POST /teacher/courses/{course_id}/interview-question-bank``.

    Either supplied directly by a teacher or forwarded from an existing
    interview question ("add to bank"). ``source_config_id`` is provenance
    only.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    question_type: QuestionTypeLiteral
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None
    source_config_id: UUID | None = None


class InterviewQuestionBankItemRead(BaseModel):
    """Read projection of a course-scoped interview question-bank item."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    course_id: UUID
    prompt_text: str
    question_type: QuestionTypeLiteral
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None
    variant_group_id: UUID | None = None
    source_config_id: UUID | None = None
    created_at: datetime
    updated_at: datetime


LogicalQuestionAngleLiteral = Literal["technical", "system_design", "situational", "behavioral"]


class InterviewQuestionBankLogicalGroupCreate(BaseModel):
    """Atomically add one complete, four-angle logical question to the bank."""

    model_config = ConfigDict(extra="forbid")

    items: list[InterviewQuestionBankItemCreate] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _validate_angles(self) -> InterviewQuestionBankLogicalGroupCreate:
        angles = [item.question_type for item in self.items]
        expected = {"technical", "system_design", "situational", "behavioral"}
        if set(angles) != expected or len(set(angles)) != 4:
            raise ValueError("logical question requires exactly one of each four angle types")
        return self


class InterviewQuestionBankSiblingCreate(InterviewQuestionBankItemCreate):
    """One missing angle to append to a bank singleton or partial group."""

    question_type: LogicalQuestionAngleLiteral


class InterviewQuestionBankImportRequest(BaseModel):
    """Import bank entries; selecting one grouped child imports every sibling."""

    model_config = ConfigDict(extra="forbid")

    item_ids: list[UUID] = Field(min_length=1)


class InterviewQuestionBankImportResult(BaseModel):
    """Questions copied into a target config by one atomic bank import."""

    created: list[InterviewQuestionAuthoring]
    imported_group_count: int = 0


class InterviewQuestionBankItemUpdate(BaseModel):
    """Partial edit of a bank item (management page). All fields optional."""

    model_config = ConfigDict(extra="forbid")

    prompt_text: str | None = None
    question_type: QuestionTypeLiteral | None = None
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None


# Suppress unused-import warning — Decimal is exported in case downstream
# generation-config payloads carry numeric thresholds. Keep available.
_DECIMAL_AVAILABLE = Decimal

__all__ = [
    "InterviewQuestionBankImportRequest",
    "InterviewQuestionBankImportResult",
    "InterviewQuestionBankItemCreate",
    "InterviewQuestionBankItemRead",
    "InterviewQuestionBankItemUpdate",
    "InterviewQuestionBankLogicalGroupCreate",
    "InterviewQuestionBankSiblingCreate",
    "_DECIMAL_AVAILABLE",
]
