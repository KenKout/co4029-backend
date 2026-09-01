"""Pydantic v2 DTOs returned by ``features.courses.api.public``.

ORM models stay private to the feature; cross-feature consumers bind
to these immutable shapes. Soft-deleted rows are filtered upstream by
``core/db/soft_delete.py`` -- audit columns are not exposed.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _DTOBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True, extra="ignore")


class OrgDTO(_DTOBase):
    id: UUID


class CourseDTO(_DTOBase):
    id: UUID
    organization_id: UUID
    faculty_id: UUID | None
    owner_user_id: UUID
    slug: str
    title: str
    status: str


class ModuleDTO(_DTOBase):
    id: UUID
    course_id: UUID
    title: str
    position: int
    status: str


class LessonDTO(_DTOBase):
    id: UUID
    module_id: UUID
    slug: str
    title: str
    status: str


class ModuleItemDTO(_DTOBase):
    id: UUID
    module_id: UUID
    item_type: str
    lesson_id: UUID | None
    quiz_id: UUID | None
    interview_config_id: UUID | None
    position: int


class ContentTreeItemDTO(_DTOBase):
    id: UUID
    module_id: UUID
    item_type: str
    lesson_id: UUID | None
    quiz_id: UUID | None
    interview_config_id: UUID | None
    position: int
    lesson: LessonDTO | None = None


class ContentTreeDTO(_DTOBase):
    course: CourseDTO
    modules: list[ModuleDTO]
    items: list[ContentTreeItemDTO]


__all__ = [
    "ContentTreeDTO",
    "ContentTreeItemDTO",
    "CourseDTO",
    "LessonDTO",
    "ModuleDTO",
    "ModuleItemDTO",
    "OrgDTO",
]
