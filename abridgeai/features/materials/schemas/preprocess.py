"""Pydantic models for the teacher-facing preprocessing report + overrides."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class QuarantinedUnit(BaseModel):
    """One recorded preprocessing decision, with the removed text."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    unit_kind: Literal["page", "line", "chunk"]
    page_number: int | None = None
    ordinal: int
    content: str
    occurrences: int = 1
    rule_name: str
    reason_code: str
    action: str
    rule_score: float | None = None
    detector_stage: str
    teacher_action: Literal["restore", "confirm"] | None = None
    teacher_action_at: datetime | None = None
    created_at: datetime


class PreprocessReportView(BaseModel):
    """Everything the teacher needs to audit one material's preprocessing.

    ``summary`` is the cascade's own report (page counts, deck score, role
    counts, capped decision log) as persisted in
    ``extracted_metadata['preprocess']`` — ``None`` for materials ingested
    before the preprocessing stage existed.
    """

    material_version_id: UUID
    preprocess_mode: Literal["full", "normalize_only", "off"]
    summary: dict[str, Any] | None = None
    units: list[QuarantinedUnit] = Field(default_factory=list)
    # Overrides apply on the next reprocess, never in place — surfaced so the
    # UI can pair every action with the reprocess button.
    requires_reprocess: bool = True


class TeacherActionRequest(BaseModel):
    action: Literal["restore", "confirm"]


class PreprocessModeRequest(BaseModel):
    mode: Literal["full", "normalize_only", "off"]


class CourseFilterSummaryRow(BaseModel):
    """Per-reason aggregate across one course's quarantine rows."""

    model_config = ConfigDict(from_attributes=True)

    reason_code: str
    unit_count: int
    occurrence_count: int
    restored: int
    confirmed: int


__all__ = [
    "CourseFilterSummaryRow",
    "PreprocessModeRequest",
    "PreprocessReportView",
    "QuarantinedUnit",
    "TeacherActionRequest",
]
