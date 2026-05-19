"""Sub-resource → course resolution queries for the authoring permission layer.

These ORM queries walk FK chains from a sub-resource id up to the owning
course, returning ``(course_id, owner_user_id)`` for the permission check
in :mod:`routers._deps`. The soft-delete loader filter is auto-applied to
every :class:`SoftDeleteMixin` table touched by the join.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)


async def resolve_module_to_course(db: AsyncSession, module_id: UUID) -> tuple[UUID, UUID] | None:
    """Walk ``module_id -> course_id`` via the FK column.

    Returns ``(course_id, owner_user_id)`` or ``None`` when the module
    is missing / soft-deleted.
    """
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Course, Course.id == Module.course_id)
        .where(Module.id == module_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def resolve_lesson_to_course(db: AsyncSession, lesson_id: UUID) -> tuple[UUID, UUID] | None:
    """Walk ``lesson_id -> module_id -> course_id`` via FK columns."""
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .where(Lesson.id == lesson_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def resolve_resource_to_course(
    db: AsyncSession, resource_id: UUID
) -> tuple[UUID, UUID] | None:
    """Walk ``resource_id -> lesson -> module -> course`` via FK columns."""
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .join(LessonResource, LessonResource.lesson_id == Lesson.id)
        .where(LessonResource.id == resource_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def resolve_module_item_to_course(
    db: AsyncSession, module_item_id: UUID
) -> tuple[UUID, UUID] | None:
    """Walk ``module_item_id -> module -> course`` via FK columns."""
    stmt = (
        select(Module.course_id, Course.owner_user_id)
        .join(Course, Course.id == Module.course_id)
        .join(ModuleItem, ModuleItem.module_id == Module.id)
        .where(ModuleItem.id == module_item_id)
    )
    return (await db.execute(stmt)).tuples().first()


async def resolve_outcome_to_course(db: AsyncSession, outcome_id: UUID) -> tuple[UUID, UUID] | None:
    """Walk ``outcome_id -> course`` via the FK column."""
    stmt = (
        select(CourseLearningOutcome.course_id, Course.owner_user_id)
        .join(Course, Course.id == CourseLearningOutcome.course_id)
        .where(CourseLearningOutcome.id == outcome_id)
    )
    return (await db.execute(stmt)).tuples().first()


__all__ = [
    "resolve_lesson_to_course",
    "resolve_module_item_to_course",
    "resolve_module_to_course",
    "resolve_outcome_to_course",
    "resolve_resource_to_course",
]
