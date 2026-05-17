from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class LessonProgressPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    lesson_id: UUID
    status: str
    completion_percent: Decimal
    last_activity_at: datetime | None
    total_time_seconds: int


class MaterialEngagementCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material_version_id: UUID
    engagement_seconds: int = Field(ge=0)
    scroll_position_percent: Decimal | None = Field(
        default=None, ge=Decimal("0"), le=Decimal("100")
    )
    started_at: datetime
    ended_at: datetime


class MaterialEngagementPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    material_version_id: UUID
    engagement_seconds: int
    scroll_position_percent: Decimal | None
    started_at: datetime
    ended_at: datetime
    created_at: datetime


class LessonProgressSummary(BaseModel):
    lesson_id: UUID
    status: str
    completion_percent: Decimal
    last_activity_at: datetime | None
    total_time_seconds: int


class MyCourseProgressSummary(BaseModel):
    course_id: UUID
    total_lessons: int
    completed_lessons: int
    in_progress_lessons: int
    not_started_lessons: int
    completion_percent: Decimal
    total_time_seconds: int
    last_activity_at: datetime | None
    lessons: list[LessonProgressSummary]


__all__ = [
    "LessonProgressPublic",
    "LessonProgressSummary",
    "MaterialEngagementCreate",
    "MaterialEngagementPublic",
    "MyCourseProgressSummary",
]
