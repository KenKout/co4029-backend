from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class StudentProgressRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: UUID
    total_lessons: int
    completed_lessons: int
    in_progress_lessons: int
    not_started_lessons: int
    completion_percent: Decimal
    total_time_seconds: int


class RosterProgressRead(BaseModel):
    course_id: UUID
    students: list[StudentProgressRow]


class AtRiskReason(BaseModel):
    code: str
    detail: str


class AtRiskStudent(BaseModel):
    user_id: UUID
    completion_percent: Decimal
    days_since_last_engagement: int | None
    reasons: list[AtRiskReason]


class AtRiskListRead(BaseModel):
    course_id: UUID
    students: list[AtRiskStudent]


__all__ = [
    "AtRiskListRead",
    "AtRiskReason",
    "AtRiskStudent",
    "RosterProgressRead",
    "StudentProgressRow",
]
