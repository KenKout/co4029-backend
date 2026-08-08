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


async def list_courses_by_org(db: AsyncSession, organization_id: UUID) -> list[CourseDTO]:
    """All non-deleted courses of an organization (any status), newest first.

    Unlike the learner catalogue this deliberately includes draft/archived
    courses: career-path authoring attaches courses to draft paths before
    publishing, so its picker needs the full org catalogue.
    """
    courses = await queries.list_courses_by_org(db, organization_id)
    return [CourseDTO.model_validate(course) for course in courses]


async def list_course_outcome_texts(db: AsyncSession, course_id: UUID) -> list[str]:
    """Course-level learning-outcome statements, ordered by position.

    Returns just the text (interview outcomes carry their own type/weight),
    so a sibling feature can seed rubric outcomes without importing the
    courses ORM model.
    """
    rows = await queries.list_course_outcomes(db, course_id)
    return [row.outcome_text for row in rows]


async def get_lesson_by_id(db: AsyncSession, lesson_id: UUID) -> LessonDTO | None:
    lesson = await queries.get_lesson(db, lesson_id)
    return LessonDTO.model_validate(lesson) if lesson else None


async def get_module_by_id(db: AsyncSession, module_id: UUID) -> ModuleDTO | None:
    module = await queries.get_module(db, module_id)
    return ModuleDTO.model_validate(module) if module else None


async def list_lesson_ids_for_modules(db: AsyncSession, module_ids: list[UUID]) -> list[UUID]:
    """All lesson ids under the given modules — backs module-scoped generation."""
    return await queries.list_lesson_ids_for_modules(db, module_ids)


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


async def can_view_course_content(
    db: AsyncSession,
    *,
    user_id: UUID,
    course_id: UUID,
) -> bool:
    """Tenant + enrollment + manage gate for learner content reads (materials, quizzes).

    ``True`` iff the caller holds course-management rights on the course
    (owner short-circuit included, so a teacher previewing their own
    course's material passes; ``system.administer`` passes through the
    global-scope grant), OR is an active member of the course's owning
    organization AND has an ``active`` / ``completed`` enrollment. An
    org member without an enrollment row — or with a ``dropped`` /
    ``waitlisted`` row — gets ``False``: per BR, an unenrolled student
    must not reach any course item. This is the org-membership side of
    the courses-learner ``_ensure_org_access`` perimeter, exported here
    so by-id content reads in sibling features (``/materials/{id}``,
    ``/quizzes/{id}``) can apply the same isolation instead of trusting
    a visibility flag alone.

    Lazy imports keep the cross-feature edges (access_control api.public,
    policies, enrollments api.public) inside the function body; all are
    authorised by the import-linter ignore list.
    """
    from abridgeai.features.access_control.api import public as access_api
    from abridgeai.features.access_control.policies import can_manage_course
    from abridgeai.features.enrollments.api import public as enrollments_api

    if await can_manage_course(db, user_id, course_id):
        return True
    org = await queries.get_course_org(db, course_id)
    if org is None:
        return False
    if not await access_api.is_user_member_of_org(db, user_id=user_id, org_id=org):
        return False
    return await enrollments_api.has_active_or_completed_enrollment(
        db, student_id=user_id, course_id=course_id
    )


# Cross-feature notification helpers. Re-exported from courses.services.notify
# so the enrollments feature (which enrolls students into courses) can emit the
# "enrolled in a published course" notification through the blessed public
# surface instead of importing a courses service directly.
from abridgeai.features.courses.services.notify import (  # noqa: E402
    notify_student_enrolled,
)

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
    "list_course_outcome_texts",
    "list_lesson_ids_for_modules",
    "next_module_item_position",
    "notify_student_enrolled",
    "require_lesson_authoring_access",
    "walk_resource_to_course",
    "can_view_course_content",
]
