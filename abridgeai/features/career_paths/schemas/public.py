from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CareerPathCoursePublic(BaseModel):
    course_id: UUID
    slug: str
    title: str
    position: int
    is_required: bool

    model_config = ConfigDict(from_attributes=True)


class CareerPathPublic(BaseModel):
    id: UUID
    slug: str
    name: str
    description: str | None = None
    status: Literal["published"]
    courses: list[CareerPathCoursePublic] = []

    model_config = ConfigDict(from_attributes=True)


class MyCareerEnrollmentRead(BaseModel):
    career_path_id: UUID
    slug: str
    name: str
    status: Literal["active", "completed", "dropped"]
    started_at: datetime
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class CourseProgressSummary(BaseModel):
    course_id: UUID
    slug: str
    title: str
    status: str
    completion_percent: float


class CareerPathProgressRead(BaseModel):
    career_path_id: UUID
    overall_percent: float
    course_count: int
    completed_courses: int
    in_progress_courses: int
    courses: list[CourseProgressSummary]


__all__ = [
    "CareerPathCoursePublic",
    "CareerPathProgressRead",
    "CareerPathPublic",
    "CourseProgressSummary",
    "MyCareerEnrollmentRead",
]
