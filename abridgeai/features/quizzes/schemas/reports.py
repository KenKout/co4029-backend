"""Report DTOs (Phase 10): responses + statistics reports."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ResponsesReportRow(BaseModel):
    """One student answer in the responses report."""

    model_config = ConfigDict(from_attributes=True)

    student_id: UUID
    student_name: str | None = None
    attempt_number: int
    question_id: UUID
    question_position: int
    prompt_text: str
    question_type: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    points_awarded: float


class ResponsesReportRead(BaseModel):
    quiz_id: UUID
    rows: list[ResponsesReportRow] = []


class StatisticsReportRow(BaseModel):
    """Per-question item statistics."""

    question_id: UUID
    question_position: int
    prompt_text: str
    answered_count: int
    correct_count: int
    facility_index: float | None = None  # % correct (0..1)
    discrimination_index: float | None = None  # point-biserial
    discrimination_note: str | None = None


class StatisticsReportRead(BaseModel):
    quiz_id: UUID
    attempts_analyzed: int
    rows: list[StatisticsReportRow] = []


__all__ = [
    "ResponsesReportRead",
    "ResponsesReportRow",
    "StatisticsReportRead",
    "StatisticsReportRow",
]
