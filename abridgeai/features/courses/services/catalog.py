"""Learner-side course catalog reads.

Composes :mod:`features.courses.queries.published` accessors and serializes
the resulting ORM rows / dict trees into Pydantic public DTOs. Per plan
§4192 the public surface is intentionally narrow: list / detail / content.

§A11 — slug lookups REQUIRE ``organization_id`` (slug uniqueness is
per-organization). :func:`get_published_course_detail` accepts UUID OR
slug; when a slug is passed the caller must supply ``organization_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.courses.queries import (
    CursorPage,
    get_course_instructor,
    get_published_course_by_id,
    get_published_course_by_slug,
    get_published_course_content,
    get_published_lesson_by_id,
    get_published_module_by_id,
    get_user_primary_organization_id,
    get_visible_resource_storage_target,
    list_enrolled_courses,
    list_published_course_outcomes,
    list_published_course_tags,
    list_published_courses,
    list_published_lessons,
    list_published_modules,
    list_visible_lesson_resources,
    list_visible_module_items,
)
from abridgeai.features.courses.schemas import (
    CourseContentPublic,
    CourseLearningOutcomePublic,
    CoursePublic,
    InstructorRead,
    LessonPublic,
    LessonResourcePublic,
    ModuleItemPublic,
    ModulePublic,
    TagPublic,
)
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class _StorageTarget:
    bucket: str
    object_key: str


@dataclass(frozen=True)
class ResourceDownloadUrl:
    """Presigned GET URL for a student-visible lesson resource."""

    url: str
    expires_at: datetime


def _looks_like_uuid(value: str | UUID) -> bool:
    if isinstance(value, UUID):
        return True
    try:
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


async def resolve_user_primary_organization_id(db: AsyncSession, user_id: UUID) -> UUID | None:
    """Return the user's most-recent active org membership (or ``None``).

    Routers use this for slug-lookup endpoints which need an
    ``organization_id`` per Reconciliation §A11. Platform admins
    (``scope_kind='global'``) intentionally resolve to ``None``.
    """
    return await get_user_primary_organization_id(db, user_id)


async def list_published_courses_for_user(
    db: AsyncSession,
    user_id: UUID,
    *,
    organization_id: UUID,
    limit: int = 20,
    cursor: str | None = None,
) -> CursorPage[CoursePublic]:
    """Cursor-paginated learner view of published courses.

    Currently delegates to :func:`list_published_courses`. ``user_id``
    is accepted for API symmetry and future "enrolled OR published"
    composition once the enrollments feature lands in Phase 7 — until
    then it is unused, by design.
    """
    del user_id
    page = await list_published_courses(
        db, organization_id=organization_id, limit=limit, cursor=cursor
    )
    return CursorPage(
        items=[CoursePublic.model_validate(course) for course in page.items],
        next_cursor=page.next_cursor,
    )


async def list_enrolled_courses_for_user(
    db: AsyncSession,
    user_id: UUID,
    *,
    limit: int = 20,
    cursor: str | None = None,
) -> CursorPage[CoursePublic]:
    """Active enrollments → published courses for the requesting student."""
    page = await list_enrolled_courses(db, user_id, limit=limit, cursor=cursor)
    return CursorPage(
        items=[CoursePublic.model_validate(course) for course in page.items],
        next_cursor=page.next_cursor,
    )


async def _course_with_instructor(db: AsyncSession, course: object) -> CoursePublic:
    """Validate ``course`` (ORM row OR dict) into :class:`CoursePublic` with
    the instructor block hydrated from ``UserProfile``.

    Single helper so list / detail / content paths share one composition
    rule. Falls back to ``instructor=None`` when the owner has no
    ``user_profiles`` row — the public DTO already declares
    ``instructor: InstructorRead | None = None``.
    """
    public = CoursePublic.model_validate(course)
    instructor_data = await get_course_instructor(db, public.id)
    if instructor_data is not None:
        public = public.model_copy(
            update={"instructor": InstructorRead.model_validate(instructor_data)}
        )
    return public


async def get_published_course_detail(
    db: AsyncSession,
    course_id_or_slug: str | UUID,
    *,
    organization_id: UUID | None = None,
) -> CoursePublic | None:
    """Single course detail, accepting UUID OR slug.

    Slug lookups are organization-scoped per Reconciliation §A11 — the
    caller MUST supply ``organization_id`` when ``course_id_or_slug`` is
    a slug. UUID lookups ignore the org param.
    """
    if _looks_like_uuid(course_id_or_slug):
        course_id = (
            course_id_or_slug if isinstance(course_id_or_slug, UUID) else UUID(course_id_or_slug)
        )
        course = await get_published_course_by_id(db, course_id)
    else:
        if organization_id is None:
            raise ValueError(
                "organization_id is required when looking up a course by slug "
                "(see Reconciliation §A11)"
            )
        course = await get_published_course_by_slug(db, str(course_id_or_slug), organization_id)
    if course is None:
        return None
    return await _course_with_instructor(db, course)


async def get_published_course_content_for_learner(
    db: AsyncSession, course_id: UUID
) -> CourseContentPublic | None:
    """Full published content tree (course + modules + items + lessons + quizzes).

    Mirror of :func:`get_published_course_content` — returns ``None``
    when the course is missing / unpublished / soft-deleted.

    Modules are converted to dicts before Pydantic validation to avoid
    triggering lazy-loaded ``Module.items`` relationship access outside
    an async greenlet context (MissingGreenlet).

    The course block is composed via :func:`_course_with_instructor` so
    the public DTO carries ``instructor`` alongside the modules.
    """
    tree = await get_published_course_content(db, course_id)
    if tree is None:
        return None

    # Group items_data by module_id so each ModulePublic gets its items
    # without touching the lazy ORM relationship. Quiz items map their
    # ``quiz`` row into ``target`` so the polymorphic field is populated
    # for both ``lesson`` and ``quiz`` rows.
    items_by_module: dict = {}
    for item in tree.get("items", []):
        mid = str(item["module_id"])
        target = item.get("quiz") if item["item_type"] == "quiz" else item.get("lesson")
        items_by_module.setdefault(mid, []).append({
            "id": item["id"],
            "module_id": item["module_id"],
            "item_type": item["item_type"],
            "position": item["position"],
            "target": target,
        })

    modules_data = []
    for m in tree["modules"]:
        modules_data.append({
            "id": m.id,
            "course_id": m.course_id,
            "title": m.title,
            "position": m.position,
            "items": items_by_module.get(str(m.id), []),
        })

    course_public = await _course_with_instructor(db, tree["course"])
    return CourseContentPublic.model_validate({
        "course": course_public,
        "modules": modules_data,
    })


async def list_published_modules_for_course(
    db: AsyncSession, course_id: UUID
) -> list[ModulePublic] | None:
    """Modules under a published course; ``None`` if the course is not visible."""
    course = await get_published_course_by_id(db, course_id)
    if course is None:
        return None
    modules = await list_published_modules(db, course_id)
    return [ModulePublic.model_validate(m) for m in modules]


async def list_published_course_tags_for_learner(
    db: AsyncSession, course_id: UUID
) -> list[TagPublic] | None:
    """Tags on a published course; ``None`` (→ 404) when not visible."""
    course = await get_published_course_by_id(db, course_id)
    if course is None:
        return None
    tags = await list_published_course_tags(db, course_id)
    return [TagPublic.model_validate(t) for t in tags]


async def list_published_course_outcomes_for_learner(
    db: AsyncSession, course_id: UUID
) -> list[CourseLearningOutcomePublic] | None:
    """Learning outcomes on a published course (§A12); ``None`` (→ 404) otherwise."""
    course = await get_published_course_by_id(db, course_id)
    if course is None:
        return None
    outcomes = await list_published_course_outcomes(db, course_id)
    return [CourseLearningOutcomePublic.model_validate(o) for o in outcomes]


async def get_published_module_for_learner(
    db: AsyncSession, module_id: UUID
) -> ModulePublic | None:
    module = await get_published_module_by_id(db, module_id)
    return None if module is None else ModulePublic.model_validate(module)


async def list_visible_module_items_for_learner(
    db: AsyncSession, module_id: UUID
) -> list[ModuleItemPublic] | None:
    """Visible items under a published module; ``None`` if the module is unpublished.

    Items come back as dicts with ``target`` already hydrated to the
    matching ``Lesson`` / ``Quiz`` row (see
    :func:`list_visible_module_items`); we just hand them to Pydantic.
    """
    module = await get_published_module_by_id(db, module_id)
    if module is None:
        return None
    items = await list_visible_module_items(db, module_id)
    return [ModuleItemPublic.model_validate(item) for item in items]


async def list_published_lessons_for_module(
    db: AsyncSession, module_id: UUID
) -> list[LessonPublic] | None:
    """Published lessons under a published module; ``None`` (→ 404) otherwise."""
    module = await get_published_module_by_id(db, module_id)
    if module is None:
        return None
    lessons = await list_published_lessons(db, module_id)
    return [LessonPublic.model_validate(lesson) for lesson in lessons]


async def get_published_lesson_for_learner(
    db: AsyncSession, lesson_id: UUID
) -> LessonPublic | None:
    lesson = await get_published_lesson_by_id(db, lesson_id)
    return None if lesson is None else LessonPublic.model_validate(lesson)


async def list_visible_lesson_resources_for_learner(
    db: AsyncSession, lesson_id: UUID
) -> list[LessonResourcePublic] | None:
    """Visible resources for a published lesson; ``None`` (→ 404) otherwise."""
    lesson = await get_published_lesson_by_id(db, lesson_id)
    if lesson is None:
        return None
    resources = await list_visible_lesson_resources(db, lesson_id)
    return [LessonResourcePublic.model_validate(r) for r in resources]


async def get_lesson_resource_download_url(
    db: AsyncSession, resource_id: UUID
) -> ResourceDownloadUrl | None:
    """Mint a presigned GET URL for a student-visible resource (or ``None``).

    Returns ``None`` when the resource is missing, soft-deleted, has
    ``visible_to_students=FALSE``, OR sits under an unpublished lesson /
    module / course. The router maps ``None`` to HTTP 404 so existence
    is never leaked. Returns ``None`` also when no ``storage_object`` is
    attached (the resource has no downloadable bytes).
    """
    target = await get_visible_resource_storage_target(db, resource_id)
    if target is None:
        return None
    bucket, object_key = target
    url, expires_at = await create_stream_url(_StorageTarget(bucket=bucket, object_key=object_key))
    return ResourceDownloadUrl(url=url, expires_at=expires_at)


__all__ = [
    "ResourceDownloadUrl",
    "get_lesson_resource_download_url",
    "get_published_course_content_for_learner",
    "get_published_course_detail",
    "get_published_lesson_for_learner",
    "get_published_module_for_learner",
    "list_enrolled_courses_for_user",
    "list_published_course_outcomes_for_learner",
    "list_published_course_tags_for_learner",
    "list_published_courses_for_user",
    "list_published_lessons_for_module",
    "list_published_modules_for_course",
    "list_visible_lesson_resources_for_learner",
    "list_visible_module_items_for_learner",
    "resolve_user_primary_organization_id",
]
