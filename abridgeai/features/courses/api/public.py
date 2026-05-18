"""Typed cross-feature read API for the courses feature.

Sibling features (sr, interviews, career_paths, materials, quizzes,
admin) import from this module instead of issuing raw ``text(...)``
SQL against courses-owned tables. Reads return Pydantic DTOs (the
immutable contract); ORM models stay private.

Soft-delete: every read here uses ORM ``select()`` and inherits the
soft-delete loader-criteria filter automatically. No manual
``deleted_at IS NULL`` is needed.

Cross-feature WRITE wrappers: ``next_module_item_position`` and
``insert_module_item`` are the ONLY allowed cross-feature writers
into ``module_items`` (used by quizzes / interviews authoring during
Wave 5). T1+T3 triggers stamp ``updated_at`` / ``created_by`` on the
flushed row -- callers do not set them.

``require_lesson_authoring_access`` is re-exported so consumers
(materials.routers.authoring) can depend on it via this public path
without crossing into ``courses.routers._deps`` directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)
from abridgeai.features.courses.queries.published import (
    get_published_course_content as _get_published_course_content,
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
    course = await db.get(Course, course_id)
    return CourseDTO.model_validate(course) if course else None


async def get_lesson_by_id(db: AsyncSession, lesson_id: UUID) -> LessonDTO | None:
    lesson = await db.get(Lesson, lesson_id)
    return LessonDTO.model_validate(lesson) if lesson else None


async def get_module_by_id(db: AsyncSession, module_id: UUID) -> ModuleDTO | None:
    module = await db.get(Module, module_id)
    return ModuleDTO.model_validate(module) if module else None


async def walk_resource_to_course(db: AsyncSession, resource_id: UUID) -> CourseDTO | None:
    stmt = (
        select(Course)
        .join(Module, Module.course_id == Course.id)
        .join(Lesson, Lesson.module_id == Module.id)
        .join(LessonResource, LessonResource.lesson_id == Lesson.id)
        .where(LessonResource.id == resource_id)
    )
    course = (await db.execute(stmt)).scalar_one_or_none()
    return CourseDTO.model_validate(course) if course else None


async def get_published_content_tree(db: AsyncSession, course_id: UUID) -> ContentTreeDTO | None:
    raw = await _get_published_course_content(db, course_id)
    if raw is None:
        return None
    return ContentTreeDTO.model_validate(
        {
            "course": raw["course"],
            "modules": raw["modules"],
            "items": [ContentTreeItemDTO.model_validate(item) for item in raw["items"]],
        }
    )


async def find_module_items(db: AsyncSession, *, lesson_id: UUID) -> list[ModuleItemDTO]:
    stmt = select(ModuleItem).where(ModuleItem.lesson_id == lesson_id)
    rows: Sequence[ModuleItem] = (await db.execute(stmt)).scalars().all()
    return [ModuleItemDTO.model_validate(row) for row in rows]


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
) -> ModuleItemDTO:
    """Insert a ``ModuleItem`` row and flush.

    Exactly one of ``lesson_id`` / ``quiz_id`` / ``interview_config_id``
    must be non-NULL (enforced by the DB-level XOR CHECK). Caller does
    NOT set ``created_by`` / ``updated_at`` -- the T1 + T3 listener-
    backed triggers stamp those from ``current_actor_var`` on flush.
    The session ``flush()`` here preserves the caller's UoW boundary;
    no ``commit()`` happens here.
    """
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
    return ModuleItemDTO.model_validate(row)


async def get_lesson_title(db: AsyncSession, lesson_id: UUID) -> str | None:
    stmt = select(Lesson.title).where(Lesson.id == lesson_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_course_slug(db: AsyncSession, course_id: UUID) -> str | None:
    stmt = select(Course.slug).where(Course.id == course_id)
    return (await db.execute(stmt)).scalar_one_or_none()


_USER_PRIMARY_ORG_SQL = text(
    """
    SELECT organization_id
    FROM user_role_assignments
    WHERE user_id = :user_id
      AND scope_kind IN ('organization', 'org_unit', 'course')
      AND organization_id IS NOT NULL
      AND deleted_at IS NULL
      AND (active_until IS NULL OR active_until > NOW())
    ORDER BY created_at DESC NULLS LAST
    LIMIT 1
    """
)


async def get_user_primary_org(db: AsyncSession, user_id: UUID) -> OrgDTO | None:
    """Resolve the user's primary organization for catalog scoping.

    Picks the most recent active scoped assignment from
    ``user_role_assignments``. ``scope_kind='global'`` is excluded --
    platform admins have no implicit org. The ``user_role_assignments``
    table is owned by ``features.access_control`` and we deliberately
    do NOT import its ORM model here (that would re-introduce the
    cross-feature import that wave 4 / 5 are eliminating); raw SQL with
    the explicit ``deleted_at IS NULL`` filter satisfies the T4 lint.
    """
    org_id = (await db.execute(_USER_PRIMARY_ORG_SQL, {"user_id": user_id})).scalar_one_or_none()
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
    "get_user_primary_org",
    "insert_module_item",
    "next_module_item_position",
    "require_lesson_authoring_access",
    "walk_resource_to_course",
]
