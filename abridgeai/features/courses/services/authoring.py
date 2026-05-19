"""Teacher-side authoring service for the courses aggregate.

Composes :mod:`features.courses.queries.authoring` for reads and applies
business rules + ORM writes for course / module / lesson / resource CRUD,
reordering, and soft-deletion. The legacy ``backend/app/routes/courses/
service.py`` god-file is split here by concern and routed via the queries
layer per the import-linter contract (services do not import sqlalchemy).

§A5 — :func:`add_lesson` auto-creates a ``ModuleItem`` at the next
position (single transaction; rollback on either failure).

§A6 — :func:`reorder_module_items` uses the ``_OFFSET=100_000`` two-phase
swap pattern to avoid the ``uq_module_items_position`` unique-constraint
collision mid-update.

Soft-delete — :func:`delete_lesson_resource` uses
:func:`abridgeai.core.db.recursive_delete.soft_delete_cascade`. The legacy
``db.delete(...)`` is intentionally NOT used.

Audit columns (``created_by`` / ``updated_by``) are populated automatically
by the T0.8 audit listener via the request-scoped contextvar; services
accept ``actor: CurrentUser`` to permit explicit ownership writes
(e.g. ``Course.owner_user_id`` is set from ``actor.user_id`` on create).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict as _flush_or_conflict,
)
from abridgeai.core.db.conflict_mapper import (
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)
from abridgeai.features.courses.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.courses.queries import (
    get_user_primary_organization_id,
)
from abridgeai.features.courses.schemas import (
    CourseAuthoring,
    CourseCreate,
    CourseUpdate,
    LessonAuthoring,
    LessonCreate,
    LessonResourceAuthoring,
    LessonResourceCreate,
    LessonUpdate,
    ModuleAuthoring,
    ModuleCreate,
    ModuleItemAuthoring,
    ModuleUpdate,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_OFFSET = 100_000


register_conflict_mappings(
    {
        "uq_courses_org_slug": "course_slug_taken: a course with this slug already exists in this organization",  # noqa: E501
        "modules_course_id_position_key": "module_position_taken: another module already occupies this position in the course",  # noqa: E501
        "uq_modules_course_position": "module_position_taken: another module already occupies this position in the course",  # noqa: E501
        "lessons_module_id_slug_key": "lesson_slug_taken: a lesson with this slug already exists in this module",  # noqa: E501
        "uq_lessons_module_slug": "lesson_slug_taken: a lesson with this slug already exists in this module",  # noqa: E501
        "lesson_resources_lesson_id_position_key": "lesson_resource_position_taken: another resource already occupies this position in the lesson",  # noqa: E501
        "uq_lesson_resources_position": "lesson_resource_position_taken: another resource already occupies this position in the lesson",  # noqa: E501
        "module_items_module_id_position_key": "module_item_position_taken: another item already occupies this position in the module",  # noqa: E501
        "uq_module_items_position": "module_item_position_taken: another item already occupies this position in the module",  # noqa: E501
        "uq_course_learning_outcomes_position": "course_outcome_position_taken: another outcome already occupies this position in the course",  # noqa: E501
    }
)


def _apply_patch(model: object, payload: object) -> None:
    """Apply ``payload.model_dump(exclude_unset=True)`` onto ``model`` in place."""
    data = payload.model_dump(exclude_unset=True)  # type: ignore[attr-defined]
    for key, value in data.items():
        setattr(model, key, value)


async def _require_course(db: AsyncSession, course_id: UUID) -> Course:
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")
    return course


async def _require_module(db: AsyncSession, module_id: UUID) -> Module:
    module = await authoring_queries.get_module(db, module_id)
    if module is None:
        raise NotFoundError(f"Module {module_id} not found")
    return module


async def _require_lesson(db: AsyncSession, lesson_id: UUID) -> Lesson:
    lesson = await authoring_queries.get_lesson(db, lesson_id)
    if lesson is None:
        raise NotFoundError(f"Lesson {lesson_id} not found")
    return lesson


async def _require_resource(db: AsyncSession, resource_id: UUID) -> LessonResource:
    resource = await authoring_queries.get_lesson_resource(db, resource_id)
    if resource is None:
        raise NotFoundError(f"Lesson resource {resource_id} not found")
    return resource


async def create_course(
    db: AsyncSession, payload: CourseCreate, owner: CurrentUser
) -> CourseAuthoring:
    """Create a new course owned by ``owner`` in their primary organization.

    Both ``organization_id`` and ``owner_user_id`` are server-authoritative:
    the org is resolved from the token via the access-control public surface,
    and ownership always tracks the requesting principal. This prevents a
    teacher in Org A from creating a course in Org B (or under another
    teacher's name) by sending a forged payload.

    A duplicate ``(organization_id, slug)`` is mapped to :class:`ConflictError`
    (HTTP 409) instead of bubbling the raw ``IntegrityError`` up to a 500.
    """
    org_id = await _resolve_owner_org(db, owner)
    data = payload.model_dump()
    data["organization_id"] = org_id
    data["owner_user_id"] = owner.user_id
    course = Course(**data)
    db.add(course)
    await _flush_or_conflict(db)
    await db.refresh(course)
    return CourseAuthoring.model_validate(course)


async def check_course_slug_available(db: AsyncSession, *, slug: str, owner: CurrentUser) -> bool:
    """Pre-flight check used by the SPA before submitting a new-course form.

    Resolves the owner's primary org the same way :func:`create_course`
    does so the answer matches what the actual write would do; returns
    ``True`` when the slug is free for that org. Soft-deleted rows are
    excluded (the partial UNIQUE INDEX behind ``uq_courses_org_slug``
    already excludes them).
    """
    org_id = await _resolve_owner_org(db, owner)
    return not await authoring_queries.course_slug_exists(db, organization_id=org_id, slug=slug)


async def _resolve_owner_org(db: AsyncSession, owner: CurrentUser) -> UUID:
    org_id = await get_user_primary_organization_id(db, owner.user_id)
    if org_id is None:
        raise AppError(f"User {owner.user_id} has no primary organization; cannot create a course.")
    return org_id


async def update_course(
    db: AsyncSession,
    course_id: UUID,
    payload: CourseUpdate,
    actor: CurrentUser,
) -> CourseAuthoring:
    del actor
    course = await _require_course(db, course_id)
    _apply_patch(course, payload)
    await _flush_or_conflict(db)
    await db.refresh(course)
    return CourseAuthoring.model_validate(course)


async def publish_course(db: AsyncSession, course_id: UUID, actor: CurrentUser) -> CourseAuthoring:
    """Transition a course's status to ``published``.

    The plan body suggests checking "all modules ready" before publish.
    For T3.5 the gate is intentionally minimal (status transition only)
    so the API surface is stable; tighter gates will land alongside
    quizzes / interviews when those features can publish independently.
    """
    del actor
    course = await _require_course(db, course_id)
    if course.status == "archived":
        raise AppError(f"Cannot publish archived course {course_id}")
    course.status = "published"
    await db.flush()
    await db.refresh(course)
    return CourseAuthoring.model_validate(course)


async def archive_course(db: AsyncSession, course_id: UUID, actor: CurrentUser) -> CourseAuthoring:
    del actor
    course = await _require_course(db, course_id)
    course.status = "archived"
    await db.flush()
    await db.refresh(course)
    return CourseAuthoring.model_validate(course)


async def add_module(
    db: AsyncSession,
    course_id: UUID,
    payload: ModuleCreate,
    actor: CurrentUser,
) -> ModuleAuthoring:
    del actor
    await _require_course(db, course_id)
    data = payload.model_dump()
    data["course_id"] = course_id
    module = Module(**data)
    db.add(module)
    await _flush_or_conflict(db)
    await db.refresh(module)
    return ModuleAuthoring.model_validate(module)


async def update_module(
    db: AsyncSession,
    module_id: UUID,
    payload: ModuleUpdate,
    actor: CurrentUser,
) -> ModuleAuthoring:
    del actor
    module = await _require_module(db, module_id)
    _apply_patch(module, payload)
    await _flush_or_conflict(db)
    await db.refresh(module)
    return ModuleAuthoring.model_validate(module)


async def add_lesson(
    db: AsyncSession,
    module_id: UUID,
    payload: LessonCreate,
    actor: CurrentUser,
) -> LessonAuthoring:
    """Create a lesson under ``module_id`` AND auto-create the matching
    :class:`ModuleItem` at the next free position (Reconciliation §A5).

    Both inserts run in the same transaction. If the ``ModuleItem``
    insert fails the caller's outer transaction will roll back the
    lesson too; we ``flush`` (not commit) so service composition stays
    atomic at the router boundary. Either flush may surface a UNIQUE
    collision (duplicate ``(module_id, slug)`` on the lesson, duplicate
    ``(module_id, position)`` on the auto-item) which
    :func:`_flush_or_conflict` translates to :class:`ConflictError`.
    """
    del actor
    module = await _require_module(db, module_id)
    data = payload.model_dump()
    data["module_id"] = module.id
    lesson = Lesson(**data)
    db.add(lesson)
    await _flush_or_conflict(db)

    next_pos = await authoring_queries.next_module_item_position(db, module.id)
    item = ModuleItem(
        module_id=module.id,
        item_type="lesson",
        lesson_id=lesson.id,
        position=next_pos,
    )
    db.add(item)
    await _flush_or_conflict(db)
    await db.refresh(lesson)
    return LessonAuthoring.model_validate(lesson)


async def update_lesson(
    db: AsyncSession,
    lesson_id: UUID,
    payload: LessonUpdate,
    actor: CurrentUser,
) -> LessonAuthoring:
    del actor
    lesson = await _require_lesson(db, lesson_id)
    _apply_patch(lesson, payload)
    await _flush_or_conflict(db)
    await db.refresh(lesson)
    return LessonAuthoring.model_validate(lesson)


async def add_lesson_resource(
    db: AsyncSession,
    lesson_id: UUID,
    payload: LessonResourceCreate,
    actor: CurrentUser,
) -> LessonResourceAuthoring:
    del actor
    lesson = await _require_lesson(db, lesson_id)
    data = payload.model_dump()
    data["lesson_id"] = lesson.id
    resource = LessonResource(**data)
    db.add(resource)
    await _flush_or_conflict(db)
    await db.refresh(resource)
    return LessonResourceAuthoring.model_validate(resource)


async def delete_lesson_resource(db: AsyncSession, resource_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete a lesson resource via :func:`soft_delete_cascade` (T0.15)."""
    resource = await _require_resource(db, resource_id)
    await soft_delete_cascade(db, resource, actor_id=actor.user_id)


async def reorder_module_items(
    db: AsyncSession,
    module_id: UUID,
    item_ids: list[UUID],
    actor: CurrentUser,
) -> list[ModuleItemAuthoring]:
    """Reorder ``ModuleItem`` rows under ``module_id`` (Reconciliation §A6).

    Two-phase swap pattern:

    1. Bump every target row to ``_OFFSET + i`` (escapes the
       ``uq_module_items_position`` unique constraint).
    2. Re-assign final positions ``i + 1`` (1-indexed to match the
       legacy convention).

    Items not in ``item_ids`` keep their existing positions; callers are
    expected to send the FULL ordered list per plan §4207.
    """
    del actor
    items_by_id: dict[UUID, ModuleItem] = {}
    for idx, item_id in enumerate(item_ids):
        item = await authoring_queries.get_module_item(db, item_id)
        if item is None or item.module_id != module_id:
            raise NotFoundError(f"ModuleItem {item_id} not found in module {module_id}")
        item.position = _OFFSET + idx
        items_by_id[item_id] = item
    await _flush_or_conflict(db)

    for idx, item_id in enumerate(item_ids, start=1):
        items_by_id[item_id].position = idx
    await _flush_or_conflict(db)

    return [
        ModuleItemAuthoring.model_validate(item)
        for item in await authoring_queries.list_module_items(db, module_id)
    ]


async def set_module_prerequisites(
    db: AsyncSession,
    module_id: UUID,
    prereq_module_ids: list[UUID],
    actor: CurrentUser,
) -> ModuleAuthoring:
    """Idempotently replace the prereq set for ``module_id``.

    Existing rows are deleted, then the new set is inserted in one
    flush. Returns the refreshed module (the prereq list itself lives
    on ``ModuleAuthoring.prerequisites`` which the service hydrates
    separately when serializing the content tree — for the bare module
    view, the post-write state is enough).
    """
    del actor
    module = await _require_module(db, module_id)
    await authoring_queries.replace_module_prerequisites(db, module.id, prereq_module_ids)
    await db.refresh(module)
    return ModuleAuthoring.model_validate(module)


__all__ = [
    "add_lesson",
    "add_lesson_resource",
    "add_module",
    "archive_course",
    "check_course_slug_available",
    "create_course",
    "delete_lesson_resource",
    "publish_course",
    "reorder_module_items",
    "set_module_prerequisites",
    "update_course",
    "update_lesson",
    "update_module",
]
