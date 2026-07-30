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


# ---------------------------------------------------------------------------
# Resource -> owning organization_id (tenancy scoping for learner reads).
#
# The learner catalog endpoints address courses / modules / lessons / resources
# by id under a bare ``get_current_user`` (students hold no course permission),
# so ``require_course_permission`` cannot gate them. Without an organization
# check a student in org A could read org B's published content by id. These
# resolvers return the owning ``organization_id`` (or ``None`` when the row is
# missing / soft-deleted) so the router can 404 a cross-tenant read.
# ---------------------------------------------------------------------------


async def organization_id_for_course(db: AsyncSession, course_id: UUID) -> UUID | None:
    stmt = select(Course.organization_id).where(Course.id == course_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def organization_id_for_module(db: AsyncSession, module_id: UUID) -> UUID | None:
    stmt = (
        select(Course.organization_id)
        .join(Module, Module.course_id == Course.id)
        .where(Module.id == module_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def organization_id_for_lesson(db: AsyncSession, lesson_id: UUID) -> UUID | None:
    stmt = (
        select(Course.organization_id)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .where(Lesson.id == lesson_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def organization_id_for_resource(db: AsyncSession, resource_id: UUID) -> UUID | None:
    stmt = (
        select(Course.organization_id)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .join(LessonResource, LessonResource.lesson_id == Lesson.id)
        .where(LessonResource.id == resource_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


__all__ = [
    "organization_id_for_course",
    "organization_id_for_lesson",
    "organization_id_for_module",
    "organization_id_for_resource",
    "resolve_lesson_to_course",
    "resolve_module_item_to_course",
    "resolve_module_to_course",
    "resolve_outcome_to_course",
    "resolve_resource_to_course",
]
