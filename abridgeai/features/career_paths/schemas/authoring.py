from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CareerPathAuthoring(BaseModel):
    id: UUID
    organization_id: UUID
    org_unit_id: UUID | None = None
    slug: str
    name: str
    description: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    created_by: UUID | None = None
    updated_by: UUID | None = None
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class CareerPathCourseAuthoring(BaseModel):
    career_path_id: UUID
    course_id: UUID
    position: int
    is_required: bool
    course_slug: str
    course_title: str
    course_status: str

    model_config = ConfigDict(from_attributes=True)


class CareerPathCreate(BaseModel):
    organization_id: UUID
    org_unit_id: UUID | None = None
    slug: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class CareerPathUpdate(BaseModel):
    org_unit_id: UUID | None = None
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class CareerPathCourseAdd(BaseModel):
    course_id: UUID
    position: int | None = Field(default=None, gt=0)
    is_required: bool = True

    model_config = ConfigDict(extra="forbid")


class CareerPathCourseReorder(BaseModel):
    course_ids: list[UUID]

    model_config = ConfigDict(extra="forbid")


class CareerPathStudentEnroll(BaseModel):
    student_id: UUID

    model_config = ConfigDict(extra="forbid")


class StudentCareerEnrollmentAuthoring(BaseModel):
    id: UUID
    career_path_id: UUID
    student_id: UUID
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    created_by: UUID | None = None
    updated_by: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class StudentPathProgressAuthoring(BaseModel):
    student_id: UUID
    student_email: str
    overall_percent: float
    completed_courses: int
    course_count: int


__all__ = [
    "CareerPathAuthoring",
    "CareerPathCourseAdd",
    "CareerPathCourseAuthoring",
    "CareerPathCourseReorder",
    "CareerPathCreate",
    "CareerPathStudentEnroll",
    "CareerPathUpdate",
    "StudentCareerEnrollmentAuthoring",
    "StudentPathProgressAuthoring",
]
