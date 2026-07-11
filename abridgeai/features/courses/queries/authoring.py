from __future__ import annotations

from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute, selectinload
from sqlalchemy.sql.elements import ColumnElement

from abridgeai.features.access_control.models import Role, UserRoleAssignment
from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
    ModulePrerequisite,
)
from abridgeai.features.enrollments.models import Enrollment
from abridgeai.features.identity.models import StorageObject, User, UserProfile

_ROSTER_WITH_PROGRESS_SQL = text(
    resources.files("abridgeai.features.courses.queries.sql")
    .joinpath("roster_with_progress.sql")
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
    id_stmt = (
        select(UserRoleAssignment.course_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
    )
    ids = [row[0] for row in (await db.execute(id_stmt)).all() if row[0] is not None]
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


async def get_course_with_content_tree(
    db: AsyncSession,
    course_id: UUID,
    *,
    include_archived: bool = False,
) -> Course | None:
    """Course + modules + items + polymorphic targets (lesson/quiz/interview).

    Eager-loads modules → items → lesson/quiz/interview_config in 4 queries
    (one selectinload per branch). Soft-deleted rows are excluded by callers.
    Returns the ORM ``Course`` instance with relationships hydrated, or
    ``None`` if the course is missing or soft-deleted.
    """
    stmt = (
        select(Course)
        .where(Course.id == course_id, Course.deleted_at.is_(None))
        .options(
            selectinload(Course.modules).selectinload(Module.items).selectinload(ModuleItem.lesson),
            selectinload(Course.modules).selectinload(Module.items).selectinload(ModuleItem.quiz),
            selectinload(Course.modules)
            .selectinload(Module.items)
            .selectinload(ModuleItem.interview_config),
        )
    )
    if not include_archived:
        stmt = stmt.where(Course.status != "archived")
    return (await db.execute(stmt)).scalar_one_or_none()


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


async def get_authoring_resource_storage_target(
    db: AsyncSession, resource_id: UUID
) -> tuple[str, str] | None:
    """Bucket + object_key for a lesson resource the teacher can edit.

    Authoring sibling of :func:`published.get_visible_resource_storage_target`:
    no learner publish-chain gates and no ``visible_to_students``
    filter — the teacher must reach hidden / draft resources during
    course assembly. Soft-deleted rows are still excluded. Returns
    ``None`` for missing resources or resources without an attached
    storage object.
    """
    stmt = (
        select(StorageObject.bucket, StorageObject.object_key)
        .join(LessonResource, LessonResource.storage_object_id == StorageObject.id)
        .where(
            LessonResource.id == resource_id,
            LessonResource.deleted_at.is_(None),
        )
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    return row.bucket, row.object_key


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


async def list_course_roster(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Enrolled students for a course with user profile info."""
    stmt = (
        select(
            Enrollment.id.label("enrollment_id"),
            Enrollment.student_id,
            User.primary_email,
            UserProfile.display_name,
            Enrollment.status,
            Enrollment.enrolled_at,
            Enrollment.completed_at,
            Enrollment.dropped_at,
        )
        .join(User, User.id == Enrollment.student_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(Enrollment.course_id == course_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(row) for row in rows]


async def list_course_roster_with_progress(
    db: AsyncSession, course_id: UUID
) -> list[dict[str, Any]]:
    """Enrolled students for the teacher roster view, with progress + risk.

    Composes ``sql/roster_with_progress.sql`` — a richer projection than
    :func:`list_course_roster` (used by the ``/dept`` HOD-scope roster,
    which stays on the thin shape). Adds ``progress_percent``,
    ``last_activity_at`` and a derived ``at_risk_level``, matching the
    frontend's ``RosterStudent`` DTO 1:1. The same lesson-progress
    aggregation as ``progress/queries/sql/roster_progress.sql`` so the
    Students and Progress pages never disagree on a student's percent.
    """
    rows = (await db.execute(_ROSTER_WITH_PROGRESS_SQL, {"course_id": course_id})).mappings().all()
    return [dict(row) for row in rows]


__all__ = [
    "get_authoring_resource_storage_target",
    "get_course_for_authoring",
    "get_course_with_content_tree",
    "get_lesson",
    "get_lesson_resource",
    "get_module",
    "get_module_item",
    "list_all_lesson_resources",
    "list_course_roster",
    "list_course_roster_with_progress",
    "list_courses_assigned_to_teacher",
    "list_courses_for_owner",
    "list_courses_in_org_unit",
    "list_lessons_for_authoring",
    "list_module_items",
    "list_module_prerequisites",
    "list_modules_for_authoring",
    "next_module_item_position",
    "replace_module_prerequisites",
]
