"""Cross-feature query helpers backing api/public.py.

These ORM queries serve the typed cross-feature read surface. They are
separated from the authoring/published query modules because they serve
a different consumer (sibling features, not routers).
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.models import UserRoleAssignment
from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)


async def get_course(db: AsyncSession, course_id: UUID) -> Course | None:
    return await db.get(Course, course_id)


async def get_lesson(db: AsyncSession, lesson_id: UUID) -> Lesson | None:
    return await db.get(Lesson, lesson_id)


async def get_module(db: AsyncSession, module_id: UUID) -> Module | None:
    return await db.get(Module, module_id)


async def walk_resource_to_course(db: AsyncSession, resource_id: UUID) -> Course | None:
    stmt = (
        select(Course)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .join(LessonResource, LessonResource.lesson_id == Lesson.id)
        .where(LessonResource.id == resource_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_published_lessons_for_course(db: AsyncSession, course_id: UUID) -> Sequence[Lesson]:
    """Published lessons in a course (modules must also be published)."""
    stmt = (
        select(Lesson)
        .join(Module, Module.id == Lesson.module_id)
        .join(Course, Course.id == Module.course_id)
        .where(
            Course.id == course_id,
            Module.status == "published",
            Lesson.status == "published",
        )
        .order_by(Module.position, Lesson.id)
    )
    return (await db.execute(stmt)).scalars().all()


async def find_module_items_by_lesson(db: AsyncSession, lesson_id: UUID) -> Sequence[ModuleItem]:
    stmt = select(ModuleItem).where(ModuleItem.lesson_id == lesson_id)
    return (await db.execute(stmt)).scalars().all()


async def next_module_item_position(db: AsyncSession, module_id: UUID) -> int:
    stmt = select(func.coalesce(func.max(ModuleItem.position), 0)).where(
        ModuleItem.module_id == module_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def insert_module_item(
    db: AsyncSession,
    *,
    module_id: UUID,
    item_type: str,
    position: int,
    lesson_id: UUID | None = None,
    quiz_id: UUID | None = None,
    interview_config_id: UUID | None = None,
) -> ModuleItem:
    """Insert a ModuleItem row and flush."""
    row = ModuleItem(
        module_id=module_id,
        item_type=item_type,
        lesson_id=lesson_id,
        quiz_id=quiz_id,
        interview_config_id=interview_config_id,
        position=position,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


async def get_lesson_title(db: AsyncSession, lesson_id: UUID) -> str | None:
    stmt = select(Lesson.title).where(Lesson.id == lesson_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_course_slug(db: AsyncSession, course_id: UUID) -> str | None:
    stmt = select(Course.slug).where(Course.id == course_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_user_primary_org_id(db: AsyncSession, user_id: UUID) -> UUID | None:
    """Resolve the user's primary organization for catalog scoping.

    Picks the most recent active scoped assignment from
    ``user_role_assignments``. ``scope_kind='global'`` is excluded --
    platform admins have no implicit org.
    """
    stmt = (
        select(UserRoleAssignment.organization_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.scope_kind.in_(["organization", "org_unit", "course"]),
            UserRoleAssignment.organization_id.is_not(None),
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
        .order_by(UserRoleAssignment.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


__all__ = [
    "find_module_items_by_lesson",
    "get_course",
    "get_course_slug",
    "get_lesson",
    "get_lesson_title",
    "get_module",
    "get_published_lessons_for_course",
    "get_user_primary_org_id",
    "insert_module_item",
    "next_module_item_position",
    "walk_resource_to_course",
]
