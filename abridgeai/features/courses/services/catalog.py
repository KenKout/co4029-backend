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
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.courses.queries import (
    CursorPage,
    build_outcome_code_map,
    get_course_instructor,
    get_published_course_by_id,
    get_published_course_by_slug,
    get_published_course_content,
    get_published_course_thumbnail_storage_target,
    get_published_lesson_by_id,
    get_published_module_by_id,
    get_user_primary_organization_id,
    get_visible_lesson_resource,
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
from abridgeai.features.courses.queries import assignment as assignment_queries
from abridgeai.features.courses.schemas import (
    CourseCareerPlacementPublic,
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
from abridgeai.features.courses.services.assignment import list_teachers_with_emails
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


async def user_can_access_org_resource(
    db: AsyncSession,
    *,
    user_id: UUID,
    resource_org_id: UUID | None,
) -> bool:
    """True iff the caller may read a resource owned by ``resource_org_id``.

    Tenancy gate for the by-id learner catalog reads. Organizations do NOT
    share courses or quizzes: a learner may only reach content owned by an
    organization they actively belong to. ``resource_org_id is None`` (resource
    missing / soft-deleted) is treated as no-access so the router 404s
    identically to a genuinely absent id — the caller cannot distinguish
    "wrong tenant" from "does not exist", so ids are not an existence oracle
    across tenants.

    A platform admin (``scope_kind='global'``) has no primary org membership
    and would resolve to ``None``; such callers reach content through the admin
    surfaces, not the learner catalog, so they are intentionally NOT special-
    cased here (the learner routes are for enrolled learners).
    """
    if resource_org_id is None:
        return False
    from abridgeai.features.access_control.api.public import (  # noqa: PLC0415
        is_user_member_of_org,
    )

    return await is_user_member_of_org(db, user_id=user_id, org_id=resource_org_id)


async def organization_id_for_course_resource(
    db: AsyncSession, *, kind: str, resource_id: UUID
) -> UUID | None:
    """Resolve any learner-addressable resource id to its owning org.

    ``kind`` is one of ``course`` / ``module`` / ``lesson`` / ``resource``.
    Returns ``None`` when the row is missing or soft-deleted.
    """
    from abridgeai.features.courses.queries import resolution  # noqa: PLC0415

    resolvers = {
        "course": resolution.organization_id_for_course,
        "module": resolution.organization_id_for_module,
        "lesson": resolution.organization_id_for_lesson,
        "resource": resolution.organization_id_for_resource,
    }
    resolver = resolvers.get(kind)
    if resolver is None:  # pragma: no cover - programming error
        raise ValueError(f"unknown resource kind {kind!r}")
    return await resolver(db, resource_id)


async def course_id_for_course_resource(
    db: AsyncSession, *, kind: str, resource_id: UUID
) -> UUID | None:
    """Resolve any learner-addressable resource id to its owning course.

    Companion to :func:`organization_id_for_course_resource` for the
    enrollment gate: the router needs the COURSE id to require an active
    enrollment. Returns ``None`` when the row is missing / soft-deleted.
    """
    from abridgeai.features.courses.queries import resolution  # noqa: PLC0415

    resolvers = {
        "course": resolution.course_id_for_course,
        "module": resolution.course_id_for_module,
        "lesson": resolution.course_id_for_lesson,
        "resource": resolution.course_id_for_resource,
    }
    resolver = resolvers.get(kind)
    if resolver is None:  # pragma: no cover - programming error
        raise ValueError(f"unknown kind {kind!r}")
    return await resolver(db, resource_id)


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
    items = []
    for course in page.items:
        dto = CoursePublic.model_validate(course)
        dto.thumbnail_url = await _mint_course_thumbnail_url(db, dto.id)
        items.append(dto)
    return CursorPage(
        items=items,
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
    items = []
    for course in page.items:
        dto = CoursePublic.model_validate(course)
        dto.thumbnail_url = await _mint_course_thumbnail_url(db, dto.id)
        items.append(dto)
    return CursorPage(
        items=items,
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
        # Mint a presigned avatar_url from the instructor's storage object
        # (bucket/key), if they have an avatar uploaded. A storage blip must
        # never break the course detail page, so fall back to None.
        bucket = instructor_data.pop("avatar_bucket", None)
        object_key = instructor_data.pop("avatar_object_key", None)
        avatar_url: str | None = None
        if bucket and object_key:
            try:
                url, _ = await create_stream_url(
                    _StorageTarget(bucket=bucket, object_key=object_key)
                )
                avatar_url = url
            except Exception:  # noqa: BLE001 — storage blip must not break detail
                avatar_url = None
        instructor_data["avatar_url"] = avatar_url
        public = public.model_copy(
            update={"instructor": InstructorRead.model_validate(instructor_data)}
        )
    public.thumbnail_url = await _mint_course_thumbnail_url(db, public.id)
    # Hydrate the full teaching team, Course Instructor first then TAs, each
    # with its course_role so the learner page can label CI vs TA. Empty (and
    # `instructor` null) when the course has no assigned teachers.
    teacher_rows = await list_teachers_with_emails(db, public.id)
    if teacher_rows:
        instructors = [
            InstructorRead.model_validate(
                {
                    "user_id": row["user_id"],
                    "display_name": row["display_name"] or row["primary_email"],
                    "avatar_url": row.get("avatar_url"),
                    "headline": None,
                    "course_role": row.get("course_role"),
                }
            )
            for row in teacher_rows
        ]
        public = public.model_copy(update={"instructors": instructors})
    # Career-path placements -> the DERIVED level label (\u201cStage N \u2014 title\u201d).
    placements = await assignment_queries.list_career_paths_containing_course(db, public.id)
    if placements:
        public = public.model_copy(
            update={
                "career_paths": [
                    CourseCareerPlacementPublic.model_validate(row)
                    for row in placements
                ]
            }
        )
    return public


async def _mint_course_thumbnail_url(db: AsyncSession, course_id: UUID) -> str | None:
    """Mint a short-TTL presigned GET URL for a published course's thumbnail.

    Returns ``None`` when the course has no thumbnail set, or a storage blip
    occurs (a blip must never break a course read — the SPA falls back to the
    gradient banner).
    """
    target = await get_published_course_thumbnail_storage_target(db, course_id)
    if target is None:
        return None
    bucket, object_key = target
    try:
        url, _ = await create_stream_url(
            _StorageTarget(bucket=bucket, object_key=object_key)
        )
        return url
    except Exception:  # noqa: BLE001 — a storage blip must not break the course read
        return None


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
    # without touching the lazy ORM relationship. Each item maps its
    # type-specific row (lesson / quiz / interview config) into
    # ``target`` so the polymorphic field is populated for all rows.
    items_by_module: dict[str, list[dict[str, Any]]] = {}
    for item in tree.get("items", []):
        mid = str(item["module_id"])
        if item["item_type"] == "quiz":
            target = item.get("quiz")
        elif item["item_type"] == "interview":
            target = item.get("interview")
        else:
            # Lesson targets are slimmed: the tree must not carry the
            # lesson body past the FR-4.5 unlock gate.
            lesson_row = item.get("lesson")
            target = _slim_lesson_public(lesson_row) if lesson_row is not None else None
        items_by_module.setdefault(mid, []).append(
            {
                "id": item["id"],
                "module_id": item["module_id"],
                "item_type": item["item_type"],
                "position": item["position"],
                "target": target,
            }
        )

    modules_data = []
    for m in tree["modules"]:
        modules_data.append(
            {
                "id": m.id,
                "course_id": m.course_id,
                "title": m.title,
                "position": m.position,
                "items": items_by_module.get(str(m.id), []),
            }
        )

    course_public = await _course_with_instructor(db, tree["course"])
    return CourseContentPublic.model_validate(
        {
            "course": course_public,
            "modules": modules_data,
        }
    )


async def list_published_modules_for_course(
    db: AsyncSession, course_id: UUID
) -> list[ModulePublic] | None:
    """Modules under a published course; ``None`` if the course is not visible.

    Validates from plain dicts, NOT the raw ORM ``Module`` rows: passing an
    ORM instance straight to ``ModulePublic.model_validate`` triggers a
    lazy load of ``Module.items`` outside its async session context
    (MissingGreenlet), because ``ModulePublic.items`` defaults to ``[]``
    but pydantic still probes the attribute. This endpoint only returns
    the module list (id/title/position) — items are fetched separately via
    ``/modules/{id}/items`` — so ``items`` is always ``[]`` here, same
    mitigation as :func:`get_published_course_content_for_learner`.
    """
    course = await get_published_course_by_id(db, course_id)
    if course is None:
        return None
    modules = await list_published_modules(db, course_id)
    return [
        ModulePublic.model_validate(
            {
                "id": m.id,
                "course_id": m.course_id,
                "title": m.title,
                "position": m.position,
                "items": [],
            }
        )
        for m in modules
    ]


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
    # ``code``/``depth`` are projection-only, derived from the parent chain.
    # Without stamping them the learner view falls back to raw ``position``,
    # which is per-parent and therefore ambiguous across branches.
    code_map = build_outcome_code_map(outcomes)
    dtos: list[CourseLearningOutcomePublic] = []
    for o in outcomes:
        dto = CourseLearningOutcomePublic.model_validate(o)
        dto.code, dto.depth = code_map.get(o.id, (str(o.position), 0))
        dtos.append(dto)
    # Tree order: dotted code compared segment-wise so 1.2 sorts before 1.10.
    return sorted(
        dtos,
        key=lambda d: [int(p) for p in (d.code or "").split(".") if p.isdigit()],
    )


async def get_published_module_for_learner(
    db: AsyncSession, module_id: UUID
) -> ModulePublic | None:
    module = await get_published_module_by_id(db, module_id)
    if module is None:
        return None
    # Plain-dict validation, same MissingGreenlet mitigation as
    # :func:`list_published_modules_for_course`: validating the raw ORM
    # row makes pydantic probe the lazy ``items`` relationship outside
    # the async session. Items are served by ``/modules/{id}/items``.
    return ModulePublic.model_validate(
        {
            "id": module.id,
            "course_id": module.course_id,
            "title": module.title,
            "position": module.position,
            "items": [],
        }
    )


def _slim_lesson_public(lesson: object) -> LessonPublic:
    """Serialize a lesson WITHOUT its body fields.

    List/tree payloads are navigational: ``notes_markdown`` (the lesson
    body), ``summary`` and ``primary_material_id`` are only served by the
    unlock-gated ``GET /lessons/{lesson_id}`` (FR-4.5). Including them in
    ungated list/tree responses would let locked lesson content leak.
    """
    public = LessonPublic.model_validate(lesson)
    return public.model_copy(
        update={"summary": None, "notes_markdown": None, "primary_material_id": None}
    )


async def list_visible_module_items_for_learner(
    db: AsyncSession, module_id: UUID
) -> list[ModuleItemPublic] | None:
    """Visible items under a published module; ``None`` if the module is unpublished.

    Items come back as dicts with ``target`` already hydrated to the
    matching ``Lesson`` / ``Quiz`` row (see
    :func:`list_visible_module_items`); lesson targets are slimmed via
    :func:`_slim_lesson_public` before serialization.
    """
    module = await get_published_module_by_id(db, module_id)
    if module is None:
        return None
    items = await list_visible_module_items(db, module_id)
    slimmed = [
        {**item, "target": _slim_lesson_public(item["target"])}
        if item["item_type"] == "lesson" and item["target"] is not None
        else item
        for item in items
    ]
    return [ModuleItemPublic.model_validate(item) for item in slimmed]


async def list_published_lessons_for_module(
    db: AsyncSession, module_id: UUID
) -> list[LessonPublic] | None:
    """Published lessons under a published module; ``None`` (→ 404) otherwise.

    Body fields are stripped (:func:`_slim_lesson_public`) — the gated
    lesson-detail endpoint is the only body source.
    """
    module = await get_published_module_by_id(db, module_id)
    if module is None:
        return None
    lessons = await list_published_lessons(db, module_id)
    return [_slim_lesson_public(lesson) for lesson in lessons]


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


async def get_visible_resource_lesson_id(db: AsyncSession, resource_id: UUID) -> UUID | None:
    """Lesson id owning a student-visible resource, or ``None`` when the
    resource is missing / hidden / under unpublished content.

    Used by the learner router to apply the lesson unlock gate (FR-4.5)
    to resource downloads without leaking resource existence.
    """
    resource = await get_visible_lesson_resource(db, resource_id)
    return resource.lesson_id if resource else None


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
