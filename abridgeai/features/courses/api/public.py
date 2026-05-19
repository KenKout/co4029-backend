"""Typed cross-feature read API for the courses feature.

Sibling features (sr, interviews, career_paths, materials, quizzes,
admin) import from this module instead of issuing raw SQL against
courses-owned tables. Reads return Pydantic DTOs (the immutable
contract); ORM models stay private.

All ORM access is delegated to the queries layer
(queries/cross_feature.py). This module is a thin DTO-wrapping surface.
Soft-delete: every read inherits the soft-delete loader-criteria filter
automatically via the ORM queries layer.

``require_lesson_authoring_access`` is re-exported so consumers
(materials.routers.authoring) can depend on it via this public path
without crossing into ``courses.routers._deps`` directly.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.courses.queries import cross_feature as queries
from abridgeai.features.courses.queries.published import (
    get_published_course_content,
)
from abridgeai.features.courses.routers._deps import require_lesson_authoring_access

from ._dto import (
    ContentTreeDTO,
    ContentTreeItemDTO,
    CourseDTO,
    LessonDTO,
    ModuleDTO,
    ModuleItemDTO,
    OrgDTO,
)


async def get_course_by_id(db: AsyncSession, course_id: UUID) -> CourseDTO | None:
    course = await queries.get_course(db, course_id)
    return CourseDTO.model_validate(course) if course else None


async def get_lesson_by_id(db: AsyncSession, lesson_id: UUID) -> LessonDTO | None:
    lesson = await queries.get_lesson(db, lesson_id)
    return LessonDTO.model_validate(lesson) if lesson else None


async def get_module_by_id(db: AsyncSession, module_id: UUID) -> ModuleDTO | None:
    module = await queries.get_module(db, module_id)
    return ModuleDTO.model_validate(module) if module else None


async def walk_resource_to_course(db: AsyncSession, resource_id: UUID) -> CourseDTO | None:
    course = await queries.walk_resource_to_course(db, resource_id)
    return CourseDTO.model_validate(course) if course else None


async def get_published_content_tree(db: AsyncSession, course_id: UUID) -> ContentTreeDTO | None:
    raw = await get_published_course_content(db, course_id)
    if raw is None:
        return None
    return ContentTreeDTO.model_validate(
        {
            "course": raw["course"],
            "modules": raw["modules"],
            "items": [ContentTreeItemDTO.model_validate(item) for item in raw["items"]],
        }
    )


async def get_published_lessons_for_course(db: AsyncSession, course_id: UUID) -> list[LessonDTO]:
    """Return published lessons in a course regardless of ``courses.status``."""
    rows = await queries.get_published_lessons_for_course(db, course_id)
    return [LessonDTO.model_validate(row) for row in rows]


async def find_module_items(db: AsyncSession, *, lesson_id: UUID) -> list[ModuleItemDTO]:
    rows = await queries.find_module_items_by_lesson(db, lesson_id)
    return [ModuleItemDTO.model_validate(row) for row in rows]


async def next_module_item_position(db: AsyncSession, module_id: UUID) -> int:
    return await queries.next_module_item_position(db, module_id)


async def insert_module_item(
    db: AsyncSession,
    *,
    module_id: UUID,
    item_type: str,
    position: int,
    lesson_id: UUID | None = None,
    quiz_id: UUID | None = None,
    interview_config_id: UUID | None = None,
) -> ModuleItemDTO:
    """Insert a ``ModuleItem`` row and flush."""
    row = await queries.insert_module_item(
        db,
        module_id=module_id,
        item_type=item_type,
        position=position,
        lesson_id=lesson_id,
        quiz_id=quiz_id,
        interview_config_id=interview_config_id,
    )
    return ModuleItemDTO.model_validate(row)


async def get_lesson_title(db: AsyncSession, lesson_id: UUID) -> str | None:
    return await queries.get_lesson_title(db, lesson_id)


async def get_course_slug(db: AsyncSession, course_id: UUID) -> str | None:
    return await queries.get_course_slug(db, course_id)


async def get_user_primary_org(db: AsyncSession, user_id: UUID) -> OrgDTO | None:
    """Resolve the user's primary organization for catalog scoping."""
    org_id = await queries.get_user_primary_org_id(db, user_id)
    return OrgDTO(id=org_id) if org_id is not None else None


__all__ = [
    "ContentTreeDTO",
    "ContentTreeItemDTO",
    "CourseDTO",
    "LessonDTO",
    "ModuleDTO",
    "ModuleItemDTO",
    "OrgDTO",
    "find_module_items",
    "get_course_by_id",
    "get_course_slug",
    "get_lesson_by_id",
    "get_lesson_title",
    "get_module_by_id",
    "get_published_content_tree",
    "get_published_lessons_for_course",
    "get_user_primary_org",
    "insert_module_item",
    "next_module_item_position",
    "require_lesson_authoring_access",
    "walk_resource_to_course",
]
