from __future__ import annotations

from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
    ModulePrerequisite,
)

_AUTHORING_CONTENT_TREE_SQL = text(
    resources.files("abridgeai.features.courses.queries.sql")
    .joinpath("authoring_content_tree.sql")
    .read_text(encoding="utf-8")
)


def _archived_filter(
    include_archived: bool, status_col: InstrumentedAttribute[str]
) -> ColumnElement[bool]:
    if include_archived:
        return true()
    return status_col != "archived"


async def list_courses_for_owner(
    db: AsyncSession,
    user_id: UUID,
    *,
    include_archived: bool = False,
) -> list[Course]:
    """All courses (any status) owned by ``user_id``.

    No visibility filter — drafts surface for the author. ``include_archived``
    defaults FALSE per plan §4119; pass TRUE for the "all my courses" admin
    view.
    """
    stmt = (
        select(Course)
        .where(
            Course.owner_user_id == user_id,
            _archived_filter(include_archived, Course.status),
        )
        .order_by(Course.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_courses_assigned_to_teacher(
    db: AsyncSession,
    user_id: UUID,
    *,
    include_archived: bool = False,
) -> list[Course]:
    """All courses where ``user_id`` holds an active ``role=teacher`` course-scoped assignment.

    Mirrors :func:`list_courses_for_owner` for the "co-author" path:
    ``user_role_assignments`` rows with ``scope_kind='course'`` and
    ``role.code='teacher'`` give a teacher edit access without owning
    the course. Active = not soft-deleted AND ``active_until`` IS NULL
    or in the future.
    """
    join_sql = text(
        """
        SELECT c.id
        FROM courses c
        JOIN user_role_assignments ura ON ura.course_id = c.id
        JOIN roles r ON r.id = ura.role_id
        WHERE ura.user_id = :user_id
          AND ura.scope_kind = 'course'
          AND r.code = 'teacher'
          AND ura.deleted_at IS NULL
          AND (ura.active_until IS NULL OR ura.active_until > NOW())
        """
    )
    ids = [row[0] for row in await db.execute(join_sql, {"user_id": user_id})]
    if not ids:
        return []
    stmt = (
        select(Course)
        .where(
            Course.id.in_(ids),
            _archived_filter(include_archived, Course.status),
        )
        .order_by(Course.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_courses_in_org_unit(db: AsyncSession, org_unit_id: UUID) -> list[Course]:
    stmt = (
        select(Course).where(Course.org_unit_id == org_unit_id).order_by(Course.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_course_for_authoring(db: AsyncSession, course_id: UUID) -> Course | None:
    """Course by id without status filter (returns drafts and archived)."""
    return await db.get(Course, course_id)


async def get_course_content_authoring(
    db: AsyncSession,
    course_id: UUID,
    *,
    include_archived: bool = False,
) -> dict[str, Any] | None:
    """Full authoring tree (drafts included).

    Mirror of :func:`get_published_course_content` shape, but no
    status='published' gate. ``include_archived`` toggles the archived
    filter at every level (course, modules, lessons). Soft-deleted rows
    are still excluded.
    """
    result = await db.execute(
        _AUTHORING_CONTENT_TREE_SQL,
        {"course_id": course_id, "include_archived": include_archived},
    )
    row = result.one_or_none()
    if row is None or row.course is None:
        return None
    return {"course": row.course, "modules": row.modules, "items": row.items}


async def list_modules_for_authoring(db: AsyncSession, course_id: UUID) -> list[Module]:
    stmt = select(Module).where(Module.course_id == course_id).order_by(Module.position)
    return list((await db.execute(stmt)).scalars().all())


async def list_lessons_for_authoring(db: AsyncSession, module_id: UUID) -> list[Lesson]:
    stmt = select(Lesson).where(Lesson.module_id == module_id).order_by(Lesson.title)
    return list((await db.execute(stmt)).scalars().all())


async def list_all_lesson_resources(db: AsyncSession, lesson_id: UUID) -> list[LessonResource]:
    """All resources on the lesson (no visible_to_students filter)."""
    stmt = (
        select(LessonResource)
        .where(LessonResource.lesson_id == lesson_id)
        .order_by(LessonResource.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_module(db: AsyncSession, module_id: UUID) -> Module | None:
    return await db.get(Module, module_id)


async def course_slug_exists(db: AsyncSession, *, organization_id: UUID, slug: str) -> bool:
    """Whether an active (non-soft-deleted) course already uses ``slug`` in ``organization_id``.

    Mirrors the partial UNIQUE INDEX behind ``uq_courses_org_slug`` (see
    migration 0002): the constraint is scoped to rows where
    ``deleted_at IS NULL``, so the check must apply the same predicate.
    """
    stmt = (
        select(func.count())
        .select_from(Course)
        .where(
            Course.organization_id == organization_id,
            Course.slug == slug,
            Course.deleted_at.is_(None),
        )
    )
    return bool(int((await db.execute(stmt)).scalar_one()))


async def get_lesson(db: AsyncSession, lesson_id: UUID) -> Lesson | None:
    return await db.get(Lesson, lesson_id)


async def get_lesson_resource(db: AsyncSession, resource_id: UUID) -> LessonResource | None:
    return await db.get(LessonResource, resource_id)


async def get_module_item(db: AsyncSession, item_id: UUID) -> ModuleItem | None:
    return await db.get(ModuleItem, item_id)


async def next_module_item_position(db: AsyncSession, module_id: UUID) -> int:
    """Return ``MAX(position) + 1`` for ``module_items`` under ``module_id``.

    Mirrors the legacy ``backend/app/routes/courses/service.py:create_lesson``
    helper used to auto-place a new ``ModuleItem`` at the end of its module.
    Returns 1 when the module has no items yet.
    """
    stmt = select(func.coalesce(func.max(ModuleItem.position), 0)).where(
        ModuleItem.module_id == module_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def list_module_items(db: AsyncSession, module_id: UUID) -> list[ModuleItem]:
    """All ``ModuleItem`` rows under ``module_id`` ordered by ``position``."""
    stmt = select(ModuleItem).where(ModuleItem.module_id == module_id).order_by(ModuleItem.position)
    return list((await db.execute(stmt)).scalars().all())


async def list_module_prerequisites(db: AsyncSession, module_id: UUID) -> list[UUID]:
    """Return the list of prerequisite module ids for ``module_id``."""
    stmt = select(ModulePrerequisite.prerequisite_module_id).where(
        ModulePrerequisite.module_id == module_id
    )
    return [row[0] for row in (await db.execute(stmt)).all()]


async def replace_module_prerequisites(
    db: AsyncSession, module_id: UUID, prereq_module_ids: list[UUID]
) -> None:
    """Idempotent: clear the existing prereq set and insert the new one."""
    await db.execute(delete(ModulePrerequisite).where(ModulePrerequisite.module_id == module_id))
    for prereq_id in prereq_module_ids:
        db.add(ModulePrerequisite(module_id=module_id, prerequisite_module_id=prereq_id))
    await db.flush()


__all__ = [
    "get_course_content_authoring",
    "get_course_for_authoring",
    "get_lesson",
    "get_lesson_resource",
    "get_module",
    "get_module_item",
    "list_all_lesson_resources",
    "list_courses_for_owner",
    "list_courses_in_org_unit",
    "list_lessons_for_authoring",
    "list_module_items",
    "list_module_prerequisites",
    "list_modules_for_authoring",
    "next_module_item_position",
    "replace_module_prerequisites",
]
