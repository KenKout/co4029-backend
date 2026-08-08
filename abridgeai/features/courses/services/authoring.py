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

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict as _flush_or_conflict,
)
from abridgeai.core.db.conflict_mapper import (
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)
from abridgeai.features.courses.queries import (
    assignment as assignment_queries,
)
from abridgeai.features.courses.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.courses.queries import (
    find_active_teacher_assignment,
    get_teacher_role_id,
    get_user_primary_organization_id,
    insert_teacher_assignment,
)
from abridgeai.features.courses.schemas import (
    CourseAuthoring,
    CourseCreate,
    CourseLearningOutcomeAuthoring,
    CourseLearningOutcomeCreate,
    CourseLearningOutcomeUpdate,
    CourseUpdate,
    LessonAuthoring,
    LessonCreate,
    LessonResourceAuthoring,
    LessonResourceCreate,
    LessonUpdate,
    ModuleAuthoring,
    ModuleCreate,
    ModuleItemAuthoring,
    ModuleItemUpdate,
    ModuleUpdate,
    TeacherDashboardStats,
)
from abridgeai.features.identity.models import StorageObject
from abridgeai.features.interviews.api import public as interviews_public
from abridgeai.features.quizzes.api import public as quizzes_public
from abridgeai.infrastructure.s3 import create_stream_url, put_object_bytes


@dataclass
class _AuthoringStorageTarget:
    bucket: str
    object_key: str


# Course thumbnail upload constraints (mirrors the avatar upload feature).
_THUMBNAIL_MIME_TYPES: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}
_THUMBNAIL_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB — thumbnails can be larger than avatars.


class ThumbnailUploadError(ValueError):
    """Raised on an unsupported type or oversized course-thumbnail upload."""


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
    if course is None or course.deleted_at is not None:
        raise NotFoundError(f"Course {course_id} not found")
    return course


async def _require_module(db: AsyncSession, module_id: UUID) -> Module:
    module = await authoring_queries.get_module(db, module_id)
    if module is None or module.deleted_at is not None:
        raise NotFoundError(f"Module {module_id} not found")
    return module


async def _require_lesson(db: AsyncSession, lesson_id: UUID) -> Lesson:
    lesson = await authoring_queries.get_lesson(db, lesson_id)
    if lesson is None or lesson.deleted_at is not None:
        raise NotFoundError(f"Lesson {lesson_id} not found")
    return lesson


async def _require_resource(db: AsyncSession, resource_id: UUID) -> LessonResource:
    resource = await authoring_queries.get_lesson_resource(db, resource_id)
    if resource is None or resource.deleted_at is not None:
        raise NotFoundError(f"Lesson resource {resource_id} not found")
    return resource


async def _require_module_item(db: AsyncSession, item_id: UUID) -> ModuleItem:
    item = await authoring_queries.get_module_item(db, item_id)
    if item is None or item.deleted_at is not None:
        raise NotFoundError(f"ModuleItem {item_id} not found")
    return item


async def create_course(
    db: AsyncSession,
    payload: CourseCreate,
    owner: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> CourseAuthoring:
    """Create a new course owned by ``owner`` in their primary organization.

    Both ``organization_id`` and ``owner_user_id`` are server-authoritative:
    the org is resolved from the token via the access-control public surface,
    and ownership always tracks the requesting principal. This prevents a
    teacher in Org A from creating a course in Org B (or under another
    teacher's name) by sending a forged payload.

    A duplicate ``(organization_id, slug)`` is mapped to :class:`ConflictError`
    (HTTP 409) instead of bubbling the raw ``IntegrityError`` up to a 500.

    The creator is auto-assigned as a teacher ONLY when they actually hold the
    teacher role. A teacher self-creating a course wants it in their authoring
    list; a manager creating one on a teacher's behalf does not — assignment is
    purely additive (``assign_teacher_to_course`` never removes anyone), so
    auto-assigning the manager left them as a permanent co-teacher on every
    course they ever created, cluttering their authoring list and the dept
    teachers tab. The manager's real handle on the course is ownership plus
    ``course.delete``/``course.publish``, none of which depend on a teacher row.
    """
    org_id = await _resolve_owner_org(db, owner)
    data = payload.model_dump()
    data["organization_id"] = org_id
    data["owner_user_id"] = owner.user_id
    course = Course(**data)
    db.add(course)
    await _flush_or_conflict(db)
    await db.refresh(course)

    if await _creator_is_teacher(db, owner.user_id):
        existing = await find_active_teacher_assignment(
            db, course_id=course.id, user_id=owner.user_id
        )
        if existing is None:
            role_id = await get_teacher_role_id(db)
            await insert_teacher_assignment(
                db,
                assignment_id=uuid4(),
                course_id=course.id,
                user_id=owner.user_id,
                role_id=role_id,
                organization_id=org_id,
                granted_by=owner.user_id,
            )
            # Notify on THIS path too, not just the explicit assign route.
            # This branch writes a real teacher assignment, so skipping the
            # notification made the outcome depend on how the row happened to
            # be created: a manager assigning someone got a notification, a
            # teacher creating their own course did not — same assignment,
            # same inbox, different result, and nothing in the inbox to show
            # the course was ever handed over.
            #
            # Best-effort inside `notify`, so a dispatch failure can never
            # roll back the course that was just created.
            await _notify_teacher_assigned(
                db,
                teacher_user_id=owner.user_id,
                course_id=course.id,
                course_title=course.title,
                arq_pool=arq_pool,
            )

    return CourseAuthoring.model_validate(course)


async def _notify_teacher_assigned(
    db: AsyncSession,
    *,
    teacher_user_id: UUID,
    course_id: UUID,
    course_title: str,
    arq_pool: object | None,
) -> None:
    """Tell a teacher they now hold a course.

    Lazy import for the same reason as :func:`_notify_course_published` — a
    module-level ``courses.services -> notify`` edge would close an import
    cycle through ``enrollments``.
    """
    from abridgeai.features.courses.services import notify  # noqa: PLC0415

    await notify.notify_teacher_assigned(
        db,
        teacher_user_id=teacher_user_id,
        course_id=course_id,
        course_title=course_title,
        arq_pool=arq_pool,
    )


async def _creator_is_teacher(db: AsyncSession, user_id: UUID) -> bool:
    """Whether ``user_id`` holds the ``teacher`` role at any scope.

    Lazy import: ``courses.services`` reaching ``access_control.api.public`` at
    module level would add a cross-feature edge at import time; the same lazy
    pattern is used by ``courses.services.catalog``.
    """
    from abridgeai.features.access_control.api import public as access_api  # noqa: PLC0415

    codes = await access_api.get_role_codes_for_users(db, [user_id])
    return "teacher" in codes.get(user_id, [])


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
    # Publishing is a one-way door: a published course can never be reverted
    # to draft. Its learning outcomes double as the graded assessment scale,
    # so re-opening it for edits would move the goalposts under enrolled
    # students. (archived is a separate terminal state; only draft->published
    # and ->archived transitions are allowed.)
    new_status = payload.status
    if (
        new_status is not None
        and new_status != course.status
        and course.status == "published"
        and new_status == "draft"
    ):
        raise ConflictError(f"Course {course_id} is published and cannot be reverted to draft.")
    # PATCH is the second door into `published`. `POST /publish` is the first.
    # Both must apply the gradeable-unit gate or the gate is decorative — a
    # manager could publish an empty course through whichever door is not
    # guarded.
    if new_status == "published" and course.status != "published":
        await _require_gradeable_units(db, course_id)
    _apply_patch(course, payload)
    await _flush_or_conflict(db)
    await db.refresh(course)
    return CourseAuthoring.model_validate(course)


async def _require_gradeable_units(db: AsyncSession, course_id: UUID) -> None:
    """Refuse to publish a course no student could ever complete.

    A gradeable unit is a published lesson, quiz or interview config. With
    zero of them the completion writer can never promote an enrollment, so
    ``satisfied`` stays false forever: as a required course on a career path
    that locks its stage and every stage behind it permanently.

    The same rule already guards career-path publication, but that fires only
    once someone puts the course on a path — possibly weeks later, and with a
    message pointing at the path rather than the course. Publishing the course
    is the first moment the system can tell the manager, so it says it here.

    Lazy import keeps the courses -> enrollments edge out of module import
    time (same pattern as ``_notify_course_published``).
    """
    from abridgeai.features.enrollments.api import public as enrollments_api  # noqa: PLC0415

    units = await enrollments_api.count_course_gradeable_units(db, course_id=course_id)
    if units == 0:
        raise ConflictError(
            f"course_has_no_gradeable_units: course {course_id} has no published "
            "lessons, quizzes or interviews, so no student could ever complete it. "
            "Add and publish at least one before publishing the course."
        )


async def _require_learning_outcomes(db: AsyncSession, course_id: UUID) -> None:
    """Refuse to publish a course that never states what it teaches.

    Learning outcomes are what a student reads to decide whether to enrol and
    what a manager maps onto a career path. Publishing without one ships a
    course whose only description of itself is its title.

    Deliberately a SEPARATE gate from :func:`_require_gradeable_units` rather
    than one merged check: the two failures have different fixes and different
    owners. Content is the teacher's job, outcomes are the manager's (see the
    authoring ownership boundary), so a single blended message would send half
    the readers to the wrong place.

    Any outcome counts, at any depth. The hierarchy is an authoring
    convenience, not a quality bar — demanding a top-level one would reject a
    perfectly-stated course whose author happened to nest everything.
    """
    outcomes = await authoring_queries.count_course_outcomes(db, course_id)
    if outcomes == 0:
        raise ConflictError(
            f"course_has_no_learning_outcomes: course {course_id} defines no "
            "learning outcomes, so students cannot tell what it teaches. Add at "
            "least one before publishing the course."
        )


async def publish_course(db: AsyncSession, course_id: UUID, actor: CurrentUser) -> CourseAuthoring:
    """Transition a course's status to ``published``.

    Two gates, both skipped when re-publishing an already-published course
    (that is a no-op, and retro-actively blocking it would strand courses
    published before either rule existed):

    * At least one gradeable unit — a published lesson, quiz or interview
      config. A course with none can never be completed by anyone (the
      completion writer refuses to promote an empty course), and as a required
      course on a career path it would lock its stage permanently. See
      :func:`_require_gradeable_units`.
    * At least one learning outcome — otherwise the course never states what
      it teaches. See :func:`_require_learning_outcomes`.

    Checked in that order so the first 409 a manager sees is the one that
    blocks students outright, not the one about documentation.

    On an actual transition INTO ``published`` (not a re-publish), everyone
    already attached to the course is notified with a deep-link: assigned
    teachers and actively-enrolled students. Notification failures never roll
    back the publish.
    """
    del actor
    course = await _require_course(db, course_id)
    if course.status == "archived":
        raise AppError(f"Cannot publish archived course {course_id}")
    was_published = course.status == "published"
    if not was_published:
        await _require_gradeable_units(db, course_id)
        await _require_learning_outcomes(db, course_id)
    course.status = "published"
    await db.flush()
    await db.refresh(course)

    if not was_published:
        await _notify_course_published(db, course)

    return CourseAuthoring.model_validate(course)


async def _notify_course_published(db: AsyncSession, course: Course) -> None:
    """Notify attached teachers + enrolled students that ``course`` published.

    Lazy imports keep the module-load graph acyclic (enrollments.api.public →
    enrollments.services.manager → courses.api.public would otherwise close a
    cycle at import time). All dispatch is best-effort inside ``notify``.
    """
    from abridgeai.features.courses.queries import assignment as assignment_queries
    from abridgeai.features.courses.services import notify
    from abridgeai.features.enrollments.api import public as enrollments_api

    teacher_rows = await assignment_queries.list_teachers_for_course(db, course.id)
    teacher_ids = [row["user_id"] for row in teacher_rows]
    student_ids = await enrollments_api.list_active_student_ids(db, course_id=course.id)

    if not teacher_ids and not student_ids:
        return

    await notify.notify_course_published(
        db,
        course_id=course.id,
        course_title=course.title,
        course_slug=course.slug,
        teacher_user_ids=teacher_ids,
        student_user_ids=student_ids,
    )


async def archive_course(db: AsyncSession, course_id: UUID, actor: CurrentUser) -> CourseAuthoring:
    del actor
    course = await _require_course(db, course_id)
    # Archiving a course that sits on a PUBLISHED path would silently remove
    # it from enrolled students' stages — the permanent stage lock the
    # add-time published check (add_course_to_path) exists to prevent. The
    # invariant is enforced on entry, so it must be maintained on exit:
    # block the archive and name the affected paths.
    live_paths = [
        p
        for p in await assignment_queries.list_career_paths_containing_course(
            db, course_id
        )
        if p["career_path_status"] == "published"
    ]
    if live_paths:
        names = ", ".join(sorted({p["career_path_name"] for p in live_paths}))
        raise AppError(
            f"Course {course.title!r} is attached to published career path(s): {names}. "
            "Remove it from those paths before archiving — archiving it would lock "
            "the stage for every enrolled student."
        )
    course.status = "archived"
    await db.flush()
    await db.refresh(course)
    return CourseAuthoring.model_validate(course)


async def delete_course(db: AsyncSession, course_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete a course (manager-facing), cascading to its children.

    Reversible tombstone via :func:`soft_delete_cascade` — nothing is
    physically removed, the row is stamped ``deleted_at`` / ``deleted_by`` and
    filtered out of every non-admin ``Course`` SELECT. Mirrors the admin
    delete but is scoped to the caller's ``course.delete`` permission on this
    course. Raises ``NotFoundError`` when the course is missing or already
    soft-deleted (``_require_course`` enforces the active-course guard).
    """
    course = await _require_course(db, course_id)
    await soft_delete_cascade(db, course, actor_id=actor.user_id)


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


async def update_module_item(
    db: AsyncSession,
    item_id: UUID,
    payload: ModuleItemUpdate,
    actor: CurrentUser,
) -> ModuleItemAuthoring:
    """Patch a single ``ModuleItem`` row (only ``unlock_rule_json`` is mutable).

    Position changes go through :func:`reorder_module_items`; identity
    (lesson_id / quiz_id / interview_config_id) is immutable per the
    XOR check on the table. Service-level allowlist mirrors the schema.
    """
    del actor
    item = await _require_module_item(db, item_id)
    data = payload.model_dump(exclude_unset=True)
    if "unlock_rule_json" in data and data["unlock_rule_json"] is not None:
        item.unlock_rule_json = data["unlock_rule_json"]
    await _flush_or_conflict(db)
    await db.refresh(item)
    return ModuleItemAuthoring.model_validate(item)


async def delete_module_item(db: AsyncSession, item_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete a single ``ModuleItem`` row (does NOT cascade to the lesson).

    The polymorphic target (lesson / quiz / interview_config) survives
    because removing an item only un-pins it from this module's order.
    Sibling items keep their existing positions; callers who want a
    repacked 1..N ordering should follow up with
    :func:`reorder_module_items`.
    """
    item = await _require_module_item(db, item_id)
    await soft_delete_cascade(db, item, actor_id=actor.user_id)


# --- Duplicate (deep clone) ------------------------------------------------
#
# Duplicated content is ALWAYS unpublished: modules/lessons land in
# ``status='draft'``; cloned quiz/interview subtrees are forced to draft +
# ``review_status='pending'`` inside their feature's ``deep_clone_*`` helper.
# A teacher must explicitly re-publish a copy — a duplicate never inherits the
# source's published/approved state.

_DUP_SUFFIX = " (Copy)"


async def _deep_clone_lesson(
    db: AsyncSession,
    *,
    source_lesson: Lesson,
    target_module_id: UUID,
    actor: CurrentUser,
) -> UUID:
    """Clone a lesson (+ its resources) into ``target_module_id`` as draft.

    Slug must stay unique per module. When cloning inside the SAME module we
    append a short uuid fragment to dodge ``uq_lessons_module_slug``; across
    modules the original slug is free to reuse.

    Resource rows are copied by reference to the same ``storage_object_id`` —
    the underlying S3 object is shared, not re-uploaded (a duplicated lesson
    points at the same files, which is the desired behaviour and avoids a
    storage blow-up).
    """
    slug = source_lesson.slug
    if source_lesson.module_id == target_module_id:
        slug = f"{slug}-copy-{uuid4().hex[:8]}"

    lesson_clone = Lesson(
        module_id=target_module_id,
        slug=slug,
        title=f"{source_lesson.title}{_DUP_SUFFIX}",
        summary=source_lesson.summary,
        notes_markdown=source_lesson.notes_markdown,
        primary_material_id=source_lesson.primary_material_id,
        lesson_type=source_lesson.lesson_type,
        difficulty=source_lesson.difficulty,
        estimated_minutes=source_lesson.estimated_minutes,
        status="draft",
        ef_min_unlock=source_lesson.ef_min_unlock,
        tau_unlock=source_lesson.tau_unlock,
        requires_interview_pass=source_lesson.requires_interview_pass,
        unlock_rule_json=dict(source_lesson.unlock_rule_json or {}),
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(lesson_clone)
    await _flush_or_conflict(db)

    resources = await authoring_queries.list_all_lesson_resources(db, source_lesson.id)
    for res in resources:
        db.add(
            LessonResource(
                lesson_id=lesson_clone.id,
                title=res.title,
                resource_type=res.resource_type,
                storage_object_id=res.storage_object_id,
                position=res.position,
                visible_to_students=res.visible_to_students,
                created_by=actor.user_id,
                updated_by=actor.user_id,
            )
        )
    await _flush_or_conflict(db)
    return lesson_clone.id


async def _clone_item_target(
    db: AsyncSession,
    *,
    source_item: ModuleItem,
    target_module_id: UUID,
    actor: CurrentUser,
) -> tuple[str, dict[str, UUID]]:
    """Deep-clone the polymorphic target a module item points at.

    Returns ``(item_type, fk_kwargs)`` where ``fk_kwargs`` binds exactly one of
    ``lesson_id`` / ``quiz_id`` / ``interview_config_id`` — ready to splat into
    a new :class:`ModuleItem`. Cross-feature quiz/interview cloning goes through
    the respective ``api.public`` (feature-independence contract).
    """
    if source_item.item_type == "lesson":
        source_lesson = await _require_lesson(db, source_item.lesson_id)
        new_lesson_id = await _deep_clone_lesson(
            db,
            source_lesson=source_lesson,
            target_module_id=target_module_id,
            actor=actor,
        )
        return "lesson", {"lesson_id": new_lesson_id}

    if source_item.item_type == "quiz":
        new_quiz_id = await quizzes_public.deep_clone_quiz(
            db,
            source_quiz_id=source_item.quiz_id,
            target_module_id=target_module_id,
            actor_id=actor.user_id,
            title_suffix=_DUP_SUFFIX,
        )
        return "quiz", {"quiz_id": new_quiz_id}

    if source_item.item_type == "interview":
        new_config_id = await interviews_public.deep_clone_interview_config(
            db,
            source_config_id=source_item.interview_config_id,
            target_module_id=target_module_id,
            actor_id=actor.user_id,
            title_suffix=_DUP_SUFFIX,
        )
        return "interview", {"interview_config_id": new_config_id}

    raise AppError(f"Unknown module item type: {source_item.item_type!r}")


async def duplicate_module_item(
    db: AsyncSession,
    item_id: UUID,
    actor: CurrentUser,
) -> ModuleItemAuthoring:
    """Deep-clone a single module item into the SAME module, appended at the end.

    The item's polymorphic target (lesson / quiz / interview) is fully cloned as
    an independent draft; the new pin appends after the current last item.
    """
    source_item = await _require_module_item(db, item_id)
    item_type, fk_kwargs = await _clone_item_target(
        db,
        source_item=source_item,
        target_module_id=source_item.module_id,
        actor=actor,
    )
    next_pos = await authoring_queries.next_module_item_position(db, source_item.module_id)
    new_item = ModuleItem(
        module_id=source_item.module_id,
        item_type=item_type,
        position=next_pos,
        unlock_rule_json=dict(source_item.unlock_rule_json or {}),
        **fk_kwargs,
    )
    db.add(new_item)
    await _flush_or_conflict(db)
    await db.refresh(new_item)
    return ModuleItemAuthoring.model_validate(new_item)


async def duplicate_module(
    db: AsyncSession,
    module_id: UUID,
    actor: CurrentUser,
) -> ModuleAuthoring:
    """Deep-clone a whole module: the module row + every item + every target.

    The new module is created as ``status='draft'`` at the end of its course's
    module order. Each of the source module's items is cloned in position
    order, and each item's lesson/quiz/interview target is deep-cloned into the
    NEW module so the copy is fully self-contained (no shared child rows with
    the original). Module prerequisites are intentionally NOT copied — they
    reference sibling modules by id and a fresh draft starts with none.
    """
    source_module = await _require_module(db, module_id)

    next_module_pos = await authoring_queries.next_module_position(db, source_module.course_id)
    module_clone = Module(
        course_id=source_module.course_id,
        title=f"{source_module.title}{_DUP_SUFFIX}",
        description=source_module.description,
        position=next_module_pos,
        status="draft",
        estimated_minutes=source_module.estimated_minutes,
        requires_all_lessons_unlocked=source_module.requires_all_lessons_unlocked,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(module_clone)
    await _flush_or_conflict(db)

    source_items = await authoring_queries.list_module_items(db, source_module.id)
    for position, source_item in enumerate(source_items, start=1):
        if source_item.deleted_at is not None:
            continue
        item_type, fk_kwargs = await _clone_item_target(
            db,
            source_item=source_item,
            target_module_id=module_clone.id,
            actor=actor,
        )
        db.add(
            ModuleItem(
                module_id=module_clone.id,
                item_type=item_type,
                position=position,
                unlock_rule_json=dict(source_item.unlock_rule_json or {}),
                **fk_kwargs,
            )
        )
        await _flush_or_conflict(db)

    await db.refresh(module_clone)
    return ModuleAuthoring.model_validate(module_clone)


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


async def reorder_modules(
    db: AsyncSession,
    course_id: UUID,
    module_ids: list[UUID],
    actor: CurrentUser,
) -> list[ModuleAuthoring]:
    """Reorder ``Module`` rows under ``course_id``.

    Two-phase swap pattern (mirrors :func:`reorder_module_items`):

    1. Bump every target row to ``_OFFSET + i`` (escapes the
       ``uq_modules_course_position`` unique constraint).
    2. Re-assign final positions ``i + 1`` (1-indexed).

    Modules not in ``module_ids`` keep their existing positions; callers
    are expected to send the FULL ordered list.
    """
    del actor
    modules_by_id: dict[UUID, Module] = {}
    for idx, module_id in enumerate(module_ids):
        module = await authoring_queries.get_module(db, module_id)
        if module is None or module.course_id != course_id:
            raise NotFoundError(f"Module {module_id} not found in course {course_id}")
        module.position = _OFFSET + idx
        modules_by_id[module_id] = module
    await _flush_or_conflict(db)

    for idx, module_id in enumerate(module_ids, start=1):
        modules_by_id[module_id].position = idx
    await _flush_or_conflict(db)

    return [
        ModuleAuthoring.model_validate(module)
        for module in await authoring_queries.list_modules_for_authoring(db, course_id)
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


async def list_authoring_courses_for_user(
    db: AsyncSession,
    *,
    user: CurrentUser,
    include_archived: bool = False,
) -> list[CourseAuthoring]:
    """Return courses the caller can author.

    Two visibility paths combined:

    * Courses the caller owns (``Course.owner_user_id == user_id``) — the
      "my drafts" view.
    * Courses the caller has a ``role=teacher, scope=course`` assignment
      on — the "courses I co-author" view.

    Soft-deleted rows are excluded by the global filter; archived rows
    are filtered unless ``include_archived`` is true.
    """
    owned = await authoring_queries.list_courses_for_owner(
        db, user.user_id, include_archived=include_archived
    )
    assigned = await authoring_queries.list_courses_assigned_to_teacher(
        db, user.user_id, include_archived=include_archived
    )
    seen: set[UUID] = set()
    merged: list[Course] = []
    for course in (*owned, *assigned):
        if course.id in seen:
            continue
        seen.add(course.id)
        merged.append(course)
    merged.sort(key=lambda c: c.created_at, reverse=True)
    result = [CourseAuthoring.model_validate(c) for c in merged]
    # Batch-count students + modules once for the whole page (no N+1), then
    # attach to each DTO for the "My courses" grid's course-health line.
    counts = await authoring_queries.count_students_and_modules_for_courses(
        db, [c.id for c in merged]
    )
    for dto, orm in zip(result, merged, strict=True):
        dto.thumbnail_url = await _mint_thumbnail_url(db, orm.id)
        students, modules = counts.get(orm.id, (0, 0))
        dto.student_count = students
        dto.module_count = modules
    return result


async def get_teacher_dashboard_stats(
    db: AsyncSession, *, user: CurrentUser
) -> TeacherDashboardStats:
    """Aggregate the teacher dashboard's actionable counts.

    Scopes to every course the caller can author (owned + assigned), then
    returns the "needs attention" tallies that power the dashboard's
    clickable widgets: courses in draft, ungraded quiz attempts, and
    interview sessions awaiting evaluation. All aggregate queries are
    batched over the course-id set — no N+1.

    Also returns the human-in-the-loop review backlog (pending quiz cards /
    interview questions, published quizzes missing an expected response
    time, ingested materials with no quiz yet) and the spaced-repetition
    retention signal (students below the EF threshold, mean EF, overdue
    cards) — same batched-over-course-ids property.
    """
    owned = await authoring_queries.list_courses_for_owner(db, user.user_id, include_archived=False)
    assigned = await authoring_queries.list_courses_assigned_to_teacher(
        db, user.user_id, include_archived=False
    )
    seen: set[UUID] = set()
    course_ids: list[UUID] = []
    draft_courses = 0
    for course in (*owned, *assigned):
        if course.id in seen:
            continue
        seen.add(course.id)
        course_ids.append(course.id)
        if course.status == "draft":
            draft_courses += 1

    (
        ungraded_quizzes,
        pending_interviews,
    ) = await authoring_queries.count_pending_grading_for_courses(db, course_ids)
    pending_review_by_course = await authoring_queries.count_pending_review_by_course(
        db, course_ids
    )
    review = await authoring_queries.count_review_queue_and_retention_for_courses(db, course_ids)
    return TeacherDashboardStats(
        draft_courses=draft_courses,
        ungraded_quizzes=ungraded_quizzes,
        pending_interviews=pending_interviews,
        pending_review_by_course=pending_review_by_course,
        quiz_cards_pending_review=review.quiz_cards_pending_review,
        interview_questions_pending_review=review.interview_questions_pending_review,
        published_quizzes_missing_texp=review.published_quizzes_missing_texp,
        materials_ready_for_quiz_gen=review.materials_ready_for_quiz_gen,
        students_below_ef_threshold=review.students_below_ef_threshold,
        avg_retention_ef=review.avg_retention_ef,
        cards_overdue=review.cards_overdue,
    )


async def get_authoring_course(db: AsyncSession, course_id: UUID) -> CourseAuthoring:
    course = await _require_course(db, course_id)
    dto = CourseAuthoring.model_validate(course)
    dto.thumbnail_url = await _mint_thumbnail_url(db, course_id)
    return dto


async def upload_course_thumbnail(
    db: AsyncSession,
    course_id: UUID,
    *,
    data: bytes,
    content_type: str,
    uploaded_by: UUID,
) -> CourseAuthoring:
    """Validate + store a course thumbnail image and point the course at it.

    Uploads the bytes server-side to object storage, records a
    ``storage_objects`` row, and sets ``courses.thumbnail_object_id``. Mirrors
    the avatar upload flow (raw-body, server-side put). Raises
    :class:`ThumbnailUploadError` on an unsupported type or oversized file.
    """
    ext = _THUMBNAIL_MIME_TYPES.get(content_type)
    if ext is None:
        raise ThumbnailUploadError(
            "unsupported_thumbnail_type: allowed types are JPEG, PNG, WebP, GIF."
        )
    if len(data) == 0:
        raise ThumbnailUploadError("empty_thumbnail: the uploaded file is empty.")
    if len(data) > _THUMBNAIL_MAX_BYTES:
        raise ThumbnailUploadError("thumbnail_too_large: images must be 5 MiB or smaller.")

    course = await _require_course(db, course_id)

    settings = get_settings()
    bucket = settings.s3_bucket_name or "abridgeai-local"
    object_id = uuid4()
    object_key = f"course-thumbnails/{course_id}/{object_id}.{ext}"

    # Upload bytes first — if storage fails we never touch the DB.
    await put_object_bytes(
        _AuthoringStorageTarget(bucket=bucket, object_key=object_key),
        data,
        content_type=content_type,
    )

    storage = StorageObject(
        id=object_id,
        bucket=bucket,
        object_key=object_key,
        original_filename=f"thumbnail.{ext}",
        mime_type=content_type,
        size_bytes=len(data),
        uploaded_by=uploaded_by,
        uploaded_at=datetime.now(tz=UTC),
    )
    db.add(storage)
    # Flush the storage row before setting the FK — this session has
    # autoflush=False, so without this the courses UPDATE can hit the DB
    # before the storage_objects INSERT and trip the FK constraint.
    await db.flush()
    course.thumbnail_object_id = object_id

    await db.commit()
    await db.refresh(course)
    dto = CourseAuthoring.model_validate(course)
    dto.thumbnail_url = await _mint_thumbnail_url(db, course_id)
    return dto


async def _mint_thumbnail_url(db: AsyncSession, course_id: UUID) -> str | None:
    """Mint a short-TTL presigned GET URL for a course's thumbnail image.

    Returns ``None`` when the course has no thumbnail set, or a storage blip
    occurs (a blip must never break a course read — the SPA falls back to the
    gradient banner).
    """
    target = await authoring_queries.get_course_thumbnail_storage_target(db, course_id)
    if target is None:
        return None
    bucket, object_key = target
    try:
        url, _ = await create_stream_url(
            _AuthoringStorageTarget(bucket=bucket, object_key=object_key)
        )
        return url
    except Exception:  # noqa: BLE001 — a storage blip must not break the course read
        return None


async def get_authoring_content(
    db: AsyncSession,
    course_id: UUID,
    *,
    include_archived: bool = False,
) -> dict[str, Any]:
    """Authoring content tree for ``course_id`` (drafts included).

    Delegates eager-loading to :func:`authoring_queries.get_course_with_content_tree`
    and composes the dict shape the schema layer expects.
    """
    course = await authoring_queries.get_course_with_content_tree(
        db, course_id, include_archived=include_archived
    )
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")

    modules_out = []
    for module in sorted(course.modules, key=lambda m: m.position):
        if module.deleted_at is not None:
            continue
        if not include_archived and module.status == "archived":
            continue

        items_out = []
        for item in sorted(module.items, key=lambda i: i.position):
            if item.deleted_at is not None:
                continue
            target = None
            if item.lesson and item.lesson.deleted_at is None:
                target = {
                    "id": item.lesson.id,
                    "title": item.lesson.title,
                    "slug": item.lesson.slug,
                    "lesson_type": item.lesson.lesson_type,
                    "status": item.lesson.status,
                    "summary": item.lesson.summary,
                    "estimated_minutes": item.lesson.estimated_minutes,
                    "difficulty": item.lesson.difficulty,
                }
            elif item.quiz and item.quiz.deleted_at is None:
                target = {
                    "id": item.quiz.id,
                    "title": item.quiz.title,
                    "status": item.quiz.status,
                }
            elif item.interview_config and item.interview_config.deleted_at is None:
                target = {
                    "id": item.interview_config.id,
                    "title": item.interview_config.title,
                    "status": item.interview_config.status,
                }

            # An item whose target resolved to None points at a soft-deleted
            # (or missing) lesson/quiz/interview. Emitting it anyway shipped a
            # dangling ``quiz_id`` to the client, which rendered a clickable
            # entry that 404'd on open. Skip it — a content item with no
            # reachable target is not content.
            if target is None:
                continue

            items_out.append(
                {
                    "id": item.id,
                    "module_id": item.module_id,
                    "item_type": item.item_type,
                    "lesson_id": item.lesson_id,
                    "quiz_id": item.quiz_id,
                    "interview_config_id": item.interview_config_id,
                    "position": item.position,
                    "unlock_rule_json": item.unlock_rule_json,
                    "target": target,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "created_by": item.created_by,
                    "updated_by": item.updated_by,
                    "deleted_at": item.deleted_at,
                    "deleted_by": item.deleted_by,
                }
            )

        module_dict = {
            "id": module.id,
            "course_id": module.course_id,
            "title": module.title,
            "description": module.description,
            "position": module.position,
            "status": module.status,
            "estimated_minutes": module.estimated_minutes,
            "requires_all_lessons_unlocked": module.requires_all_lessons_unlocked,
            "items": items_out,
            "prerequisites": [],
            "created_by": module.created_by,
            "updated_by": module.updated_by,
            "created_at": module.created_at,
            "updated_at": module.updated_at,
            "deleted_at": module.deleted_at,
            "deleted_by": module.deleted_by,
        }
        modules_out.append(module_dict)

    course_dict = {
        "id": course.id,
        "organization_id": course.organization_id,
        "org_unit_id": course.org_unit_id,
        "owner_user_id": course.owner_user_id,
        "slug": course.slug,
        "title": course.title,
        "description": course.description,
        "status": course.status,
        "level": course.level,
        "thumbnail_object_id": course.thumbnail_object_id,
        "estimated_minutes": course.estimated_minutes,
        "expected_completion_days": course.expected_completion_days,
        "enrollment_cap": course.enrollment_cap,
        "created_at": course.created_at,
        "updated_at": course.updated_at,
    }

    return {"course": course_dict, "modules": modules_out}


async def get_authoring_lesson(db: AsyncSession, lesson_id: UUID) -> LessonAuthoring:
    lesson = await _require_lesson(db, lesson_id)
    return LessonAuthoring.model_validate(lesson)


async def list_authoring_lessons(db: AsyncSession, module_id: UUID) -> list[LessonAuthoring]:
    """All non-soft-deleted lessons under ``module_id`` (drafts included).

    Authoring sibling of :func:`catalog.list_published_lessons_for_module`:
    no ``status='published'`` filter, no module-published filter — teachers
    must see drafts to manage them. Caller layer enforces authoring
    permission (``require_module_authoring_access``).
    """
    await _require_module(db, module_id)
    rows = await authoring_queries.list_lessons_for_authoring(db, module_id)
    return [LessonAuthoring.model_validate(r) for r in rows]


async def list_authoring_lesson_resources(
    db: AsyncSession, lesson_id: UUID
) -> list[LessonResourceAuthoring]:
    """All non-soft-deleted resources for ``lesson_id`` (drafts + hidden included)."""
    await _require_lesson(db, lesson_id)
    rows = await authoring_queries.list_all_lesson_resources(db, lesson_id)
    return [LessonResourceAuthoring.model_validate(r) for r in rows]


async def get_authoring_resource_download_url(
    db: AsyncSession, resource_id: UUID
) -> tuple[str, datetime]:
    """Mint a presigned GET URL for a teacher-visible resource.

    Authoring sibling of :func:`catalog.get_lesson_resource_download_url`:
    no learner publish-chain gates and no ``visible_to_students``
    filter, since the teacher must see hidden / draft resources during
    course assembly. Raises :class:`NotFoundError` for missing resources
    or resources with no storage object attached.
    """
    target = await authoring_queries.get_authoring_resource_storage_target(db, resource_id)
    if target is None:
        raise NotFoundError(f"Lesson resource {resource_id} not found or has no storage object")
    bucket, object_key = target
    url, expires_at = await create_stream_url(
        _AuthoringStorageTarget(bucket=bucket, object_key=object_key)
    )
    return url, expires_at


async def list_course_roster(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Enrolled students for a course (teacher or HOD view), with presigned avatars.

    Same bucket/key → ``avatar_url`` swap as
    :func:`list_course_roster_with_progress`; the raw storage coordinates must
    not reach the response body.
    """
    rows = await authoring_queries.list_course_roster(db, course_id)
    for row in rows:
        bucket = row.pop("avatar_bucket", None)
        object_key = row.pop("avatar_object_key", None)
        avatar_url: str | None = None
        if bucket and object_key:
            try:
                url, _ = await create_stream_url(
                    _AuthoringStorageTarget(bucket=bucket, object_key=object_key)
                )
                avatar_url = url
            except Exception:  # noqa: BLE001 — a storage blip must not break the roster
                avatar_url = None
        row["avatar_url"] = avatar_url
    return rows


async def list_course_roster_with_progress(
    db: AsyncSession, course_id: UUID
) -> list[dict[str, Any]]:
    """Enrolled students for the teacher "Students" page, with progress + risk.

    Mints a short-TTL presigned ``avatar_url`` for each student that has an
    avatar image uploaded (the SQL projects the avatar object's bucket/key);
    students without an avatar get ``avatar_url = None`` and the SPA falls back
    to initials.
    """
    rows = await authoring_queries.list_course_roster_with_progress(db, course_id)
    for row in rows:
        bucket = row.pop("avatar_bucket", None)
        object_key = row.pop("avatar_object_key", None)
        avatar_url: str | None = None
        if bucket and object_key:
            try:
                url, _ = await create_stream_url(
                    _AuthoringStorageTarget(bucket=bucket, object_key=object_key)
                )
                avatar_url = url
            except Exception:  # noqa: BLE001 — a storage blip must not break the roster
                avatar_url = None
        row["avatar_url"] = avatar_url
    return rows


# ---------------------------------------------------------------------------
# Course learning outcomes (§LO-1/2) — teacher-side CRUD.
# ---------------------------------------------------------------------------
async def _require_outcome(
    db: AsyncSession, course_id: UUID, outcome_id: UUID
) -> CourseLearningOutcome:
    outcome = await authoring_queries.get_course_outcome(db, outcome_id)
    if outcome is None or outcome.deleted_at is not None or outcome.course_id != course_id:
        raise NotFoundError(f"Course outcome {outcome_id} not found")
    return outcome


def _assert_outcomes_editable(course: Course) -> None:
    """Learning outcomes are editable only while the course is a draft.

    Once a course is published its outcomes are frozen: they double as the
    graded assessment scale, so changing/removing them after students have
    started would silently move the goalposts. Archived courses are likewise
    read-only. Callers pass the already-loaded course row so this stays a
    pure guard (→ HTTP 409 via ConflictError at the router).
    """
    if course.status != "draft":
        raise ConflictError(
            "Learning outcomes can only be edited while the course is an "
            f"unpublished draft (course {course.id} is {course.status})."
        )


async def _project_outcomes(
    db: AsyncSession, course_id: UUID, outcomes: list[CourseLearningOutcome]
) -> list[CourseLearningOutcomeAuthoring]:
    """Validate ORM rows into authoring DTOs with derived code + depth.

    The dotted ``L.O.x.y`` code and tree depth are projection-only, so we
    compute them once over the whole list and stamp each DTO. Returned in
    tree order (parent before children, siblings by position) so a client
    can render the list top-to-bottom without re-sorting.
    """
    code_map = authoring_queries.build_outcome_code_map(outcomes)
    question_counts = await authoring_queries.count_questions_mapped_to_outcomes(
        db, course_id, {o.id for o in outcomes}
    )
    dtos: dict[UUID, CourseLearningOutcomeAuthoring] = {}
    for o in outcomes:
        dto = CourseLearningOutcomeAuthoring.model_validate(o)
        code, depth = code_map.get(o.id, (str(o.position), 0))
        dto.code = code
        dto.depth = depth
        dto.question_count = question_counts.get(o.id, 0)
        dtos[o.id] = dto

    # Tree order: sort by the dotted code split into ints so 1.2 < 1.10.
    def sort_key(dto: CourseLearningOutcomeAuthoring) -> list[int]:
        return [int(part) for part in (dto.code or "").split(".") if part.isdigit()]

    return sorted(dtos.values(), key=sort_key)


async def _project_one(
    db: AsyncSession,
    course_id: UUID,
    outcomes: list[CourseLearningOutcome],
    outcome_id: UUID,
) -> CourseLearningOutcomeAuthoring:
    """Project the whole course tree, then return the one DTO we care about.

    Single-outcome mutations (create/update) still need the full list to
    derive the dotted code, since a code depends on the outcome's ancestor
    chain and sibling positions.
    """
    for dto in await _project_outcomes(db, course_id, outcomes):
        if dto.id == outcome_id:
            return dto
    raise NotFoundError(f"Course outcome {outcome_id} not found")


async def list_course_outcomes(
    db: AsyncSession, course_id: UUID
) -> list[CourseLearningOutcomeAuthoring]:
    """All learning outcomes for a course in tree order with codes (§LO-1)."""
    await _require_course(db, course_id)
    outcomes = await authoring_queries.list_course_outcomes(db, course_id)
    return await _project_outcomes(db, course_id, outcomes)


async def add_course_outcome(
    db: AsyncSession,
    course_id: UUID,
    payload: CourseLearningOutcomeCreate,
    actor: CurrentUser,
) -> CourseLearningOutcomeAuthoring:
    """Append a new outcome under its parent at the next free position (§LO-1).

    ``parent_id`` (optional) nests the outcome; it must belong to the same
    course. ``position`` is server-assigned (MAX+1 among the parent's
    children); the dotted code is derived at display time and never stored.
    """
    del actor
    course = await _require_course(db, course_id)
    _assert_outcomes_editable(course)
    if payload.parent_id is not None:
        # Parent must exist in this course (guards cross-course nesting).
        await _require_outcome(db, course_id, payload.parent_id)
    next_pos = await authoring_queries.next_course_outcome_position(
        db, course_id, payload.parent_id
    )
    outcome = CourseLearningOutcome(
        course_id=course_id,
        parent_id=payload.parent_id,
        position=next_pos,
        outcome_text=payload.outcome_text,
    )
    db.add(outcome)
    await _flush_or_conflict(db)
    outcomes = await authoring_queries.list_course_outcomes(db, course_id)
    return await _project_one(db, course_id, outcomes, outcome.id)


async def update_course_outcome(
    db: AsyncSession,
    course_id: UUID,
    outcome_id: UUID,
    payload: CourseLearningOutcomeUpdate,
    actor: CurrentUser,
) -> CourseLearningOutcomeAuthoring:
    """Edit text, re-parent and/or reorder an outcome (§LO-2).

    Re-parenting is guarded against cycles (an outcome may not become its
    own descendant, nor its own parent). When the parent changes, the
    outcome is inserted at ``position`` (default: append) among the new
    parent's children and both the old and new sibling groups are
    re-indexed so positions/codes stay contiguous.

    Reordering without re-parenting (``position`` set, ``parent_id``
    unchanged) slides the outcome to that 1-based slot among its existing
    siblings — the outliner's "drop between rows" maps to this directly.

    The dotted ``L.O.x.y`` code is display-only and derived at read time,
    so a reorder merely changes what the code renders as; the outcome's
    UUID identity never changes. External references key on the id.
    """
    del actor
    course = await _require_course(db, course_id)
    _assert_outcomes_editable(course)
    outcome = await _require_outcome(db, course_id, outcome_id)
    reparenting = "parent_id" in payload.model_fields_set
    reordering = "position" in payload.model_fields_set
    new_parent_id = payload.parent_id if reparenting else outcome.parent_id
    old_parent_id = outcome.parent_id

    moved = reparenting and new_parent_id != old_parent_id

    if moved:
        if new_parent_id is not None:
            if new_parent_id == outcome_id:
                raise AppError("An outcome cannot be its own parent")
            await _require_outcome(db, course_id, new_parent_id)
            # Cycle guard: the new parent must not be a descendant of this
            # outcome (that would detach a subtree into a loop).
            all_outcomes = await authoring_queries.list_course_outcomes(db, course_id)
            descendants = authoring_queries.build_descendant_map(all_outcomes)
            if new_parent_id in descendants.get(outcome_id, set()):
                raise AppError("Cannot move an outcome under one of its own descendants")
        outcome.parent_id = new_parent_id

    if payload.outcome_text is not None:
        outcome.outcome_text = payload.outcome_text

    # Apply the target slot: either explicit (reorder/reparent-with-slot)
    # or append at the end of the (possibly new) sibling group.
    if reordering or moved:
        if payload.position is not None:
            target_position = payload.position
        else:
            target_position = await authoring_queries.next_course_outcome_position(
                db, course_id, new_parent_id
            )
        # Remove the outcome from its current sibling list, then insert at
        # the target slot (1-based, clamped). This handles all four cases —
        # move within same parent, move into a new parent, append, and
        # move with no explicit slot — without double-counting the outcome.
        siblings = [
            o
            for o in await authoring_queries.list_course_outcome_siblings(
                db, course_id, new_parent_id
            )
            if o.id != outcome_id
        ]
        insert_at = max(1, min(target_position, len(siblings) + 1))
        siblings.insert(insert_at - 1, outcome)
        # Two-phase offset shift (same pattern as
        # reindex_course_outcome_siblings): a plain 1..N renumber in place
        # trips the per-parent unique constraint mid-update.
        for idx, sibling in enumerate(siblings, start=1):
            sibling.position = _OFFSET + idx
        await _flush_or_conflict(db)
        for pos, sibling in enumerate(siblings, start=1):
            sibling.position = pos
        await _flush_or_conflict(db)

    if moved:
        # Old siblings gapped by the move; compact them.
        await authoring_queries.reindex_course_outcome_siblings(db, course_id, old_parent_id)
        await _flush_or_conflict(db)

    outcomes = await authoring_queries.list_course_outcomes(db, course_id)
    return await _project_one(db, course_id, outcomes, outcome_id)


async def delete_course_outcome(
    db: AsyncSession,
    course_id: UUID,
    outcome_id: UUID,
    actor: CurrentUser,
    *,
    promote_children: bool = False,
) -> None:
    """Soft-delete an outcome, then compact siblings (§LO-2).

    Default: the outcome's whole subtree goes with it (soft-delete cascade),
    so no child is orphaned to a dangling parent. With ``promote_children``,
    the outcome's immediate children are re-parented onto the outcome's own
    parent (their relative order preserved) before the outcome is deleted —
    the outliner's "keep children" delete. Deeper descendants stay nested
    under their (promoted) parents.

    The FK on ``quiz_questions.learning_outcome_id`` is ``ON DELETE SET
    NULL`` but fires only on hard DELETE; we soft-delete, so questions keep
    pointing at the now-deleted rows. That's benign: the projection layer
    treats a soft-deleted / missing outcome as "no outcome", and the
    question count shown in the confirmation UI tells the teacher exactly
    which questions lose their mapping. Surviving siblings of the removed
    node are re-indexed so codes never gap.
    """
    course = await _require_course(db, course_id)
    _assert_outcomes_editable(course)
    outcome = await _require_outcome(db, course_id, outcome_id)
    parent_id = outcome.parent_id
    all_outcomes = await authoring_queries.list_course_outcomes(db, course_id)
    descendants = authoring_queries.build_descendant_map(all_outcomes)
    by_id = {o.id: o for o in all_outcomes}
    children = sorted(
        (
            by_id[d]
            for d in descendants.get(outcome_id, set())
            if d in by_id and by_id[d].parent_id == outcome_id
        ),
        key=lambda o: o.position,
    )

    if promote_children and children:
        # Re-parent immediate children onto our parent, preserving their
        # sibling order, then delete just this node. The children take the
        # parent's own slot among the siblings — that is what "keep the
        # children" means in an outliner (Workflowy/Notion promote in place).
        # Positions collide across groups (a child's old 1..N and the
        # parent's existing 1..N live in the SAME per-parent sequence once
        # re-parented), so the whole target sibling group is offset together,
        # then renumbered 1..N.
        target_siblings = await authoring_queries.list_course_outcome_siblings(
            db, course_id, parent_id
        )
        merged = []
        inserted = False
        for sibling in target_siblings:
            if sibling.id == outcome_id:
                merged.extend(children)
                inserted = True
            else:
                merged.append(sibling)
        if not inserted:
            merged.extend(children)
        for child in children:
            child.parent_id = parent_id
        for idx, sibling in enumerate(merged, start=1):
            sibling.position = _OFFSET + idx
        await _flush_or_conflict(db)
        # Drop the parent BEFORE renumbering: it still occupies its old slot
        # in this sibling group, and a 1..N renumber would collide with it.
        await soft_delete_cascade(db, outcome, actor_id=actor.user_id)
        await _flush_or_conflict(db)
        for pos, sibling in enumerate(merged, start=1):
            sibling.position = pos
        await _flush_or_conflict(db)
    else:
        to_delete = [
            outcome,
            *(by_id[d] for d in descendants.get(outcome_id, set()) if d in by_id),
        ]
        for node in to_delete:
            await soft_delete_cascade(db, node, actor_id=actor.user_id)
        await authoring_queries.reindex_course_outcome_siblings(db, course_id, parent_id)


async def duplicate_course_outcome(
    db: AsyncSession,
    course_id: UUID,
    outcome_id: UUID,
    actor: CurrentUser,
) -> CourseLearningOutcomeAuthoring:
    """Deep-copy an outcome and its subtree, inserted after the original (§LO).

    The copy gets fresh UUIDs at every node, is inserted as the original's
    next sibling (same parent, position = original + 1), and reuses the
    exact outcome_text of each node. Question mappings are NOT copied —
    questions reference outcomes by id, and a duplicate with the same
    questions would double-grade; the teacher re-maps after duplicating.
    """
    del actor
    course = await _require_course(db, course_id)
    _assert_outcomes_editable(course)
    source = await _require_outcome(db, course_id, outcome_id)
    all_outcomes = await authoring_queries.list_course_outcomes(db, course_id)
    descendants = authoring_queries.build_descendant_map(all_outcomes)
    by_id = {o.id: o for o in all_outcomes}

    # Map old id -> new ORM row (created but not yet added), preserving the
    # subtree shape: parent references are remapped through the map.
    new_by_old: dict[UUID, CourseLearningOutcome] = {}

    def clone(node: CourseLearningOutcome) -> CourseLearningOutcome:
        new_id = uuid4()
        new_row = CourseLearningOutcome(
            id=new_id,
            course_id=course_id,
            parent_id=None,  # remapped below by the caller
            # Offset immediately: the copy shares its parent's position
            # space, and the first flush must not trip the per-parent
            # unique constraint before the re-slot below runs.
            position=node.position + _OFFSET,
            outcome_text=node.outcome_text,
        )
        new_by_old[node.id] = new_row
        for child_id in sorted(descendants.get(node.id, set())):
            child = by_id[child_id]
            if child.parent_id == node.id:
                clone(child)
        return new_row

    root_clone = clone(source)
    # Remap parents through the clone map (topological: children cloned
    # after parents, so new_by_old is fully populated).
    for old_id, new_row in new_by_old.items():
        old_row = by_id[old_id]
        new_parent = new_by_old.get(old_row.parent_id) if old_row.parent_id else None
        new_row.parent_id = new_parent.id if new_parent else None

    # Position the root copy right after the original among its siblings.
    siblings = await authoring_queries.list_course_outcome_siblings(
        db, course_id, source.parent_id
    )
    insert_at = next((i for i, s in enumerate(siblings) if s.id == source.id), len(siblings)) + 1
    db.add_all(new_by_old.values())
    await _flush_or_conflict(db)
    # Re-slot the whole sibling chain including the copy (offset-shift
    # dodges the unique constraint, same as reindex_course_outcome_siblings).
    all_siblings = await authoring_queries.list_course_outcome_siblings(
        db, course_id, source.parent_id
    )
    all_siblings.insert(insert_at, root_clone)
    for pos, sibling in enumerate(all_siblings, start=1):
        sibling.position = pos
    await _flush_or_conflict(db)

    outcomes = await authoring_queries.list_course_outcomes(db, course_id)
    return await _project_one(db, course_id, outcomes, root_clone.id)


__all__ = [
    "add_course_outcome",
    "add_lesson",
    "add_lesson_resource",
    "add_module",
    "archive_course",
    "check_course_slug_available",
    "create_course",
    "delete_course_outcome",
    "delete_lesson_resource",
    "duplicate_course_outcome",
    "delete_module_item",
    "get_authoring_content",
    "get_authoring_course",
    "get_authoring_lesson",
    "get_authoring_resource_download_url",
    "list_authoring_courses_for_user",
    "list_authoring_lesson_resources",
    "list_course_outcomes",
    "list_course_roster",
    "list_course_roster_with_progress",
    "publish_course",
    "reorder_module_items",
    "set_module_prerequisites",
    "update_course",
    "update_course_outcome",
    "update_lesson",
    "update_module",
    "update_module_item",
]
