from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.features.career_paths.models import (
    CareerPath,
    CareerPathCourse,
    CareerPathStage,
    CareerPathVersion,
)
from abridgeai.features.career_paths.queries import authoring as authoring_queries
from abridgeai.features.career_paths.queries.published import (
    get_user_primary_organization_id,
)
from abridgeai.features.career_paths.schemas import (
    CareerPathAuthoring,
    CareerPathCourseAuthoring,
    CareerPathCourseCandidate,
    CareerPathCreate,
    CareerPathImpactRead,
    CareerPathImpactStage,
    CareerPathStageAuthoring,
    CareerPathStageCreate,
    CareerPathStageReorderResult,
    CareerPathStageUpdate,
    CareerPathUpdate,
    CareerPathVersionRead,
    StageReorderWarning,
)
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.enrollments.api import public as enrollments_api

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser

_OFFSET = 100_000
# "cap of 1 but this many required courses in one stage" → publish warning.
_CAP_ONE_MANY_REQUIRED = 4


register_conflict_mappings(
    {
        "career_paths_organization_id_slug_key": "career_path_slug_taken: a career path with this slug already exists in this organization",  # noqa: E501
        "uq_career_paths_org_slug": "career_path_slug_taken: a career path with this slug already exists in this organization",  # noqa: E501
        "career_path_courses_career_path_id_position_key": "career_path_course_position_taken: another course already occupies this position in the path",  # noqa: E501
        "career_path_courses_career_path_id_course_id_key": "career_path_course_already_attached: this course is already attached to the path",  # noqa: E501
    }
)


def _to_authoring(
    path: CareerPath, *, stage_count: int = 0, course_count: int = 0
) -> CareerPathAuthoring:
    return CareerPathAuthoring(
        id=path.id,
        organization_id=path.organization_id,
        org_unit_id=path.org_unit_id,
        slug=path.slug,
        name=path.name,
        description=path.description,
        status=path.status,
        stage_count=stage_count,
        course_count=course_count,
        created_at=path.created_at,
        updated_at=path.updated_at,
        created_by=path.created_by,
        updated_by=path.updated_by,
        deleted_at=path.deleted_at,
        deleted_by=path.deleted_by,
    )


async def _require_path(db: AsyncSession, career_path_id: UUID) -> CareerPath:
    path = await authoring_queries.get_career_path_for_authoring(db, career_path_id)
    if path is None or path.deleted_at is not None:
        raise NotFoundError(f"CareerPath {career_path_id} not found")
    return path


async def _require_draft_version(db: AsyncSession, version_id: UUID) -> CareerPathVersion:
    """A stage/item mutation may only touch a DRAFT version.

    Gap 3 (D1b pinned): a published version is frozen — its stages and
    course items are the route students are walking, so mutating them
    would change the promise mid-walk. Managers edit the draft (created by
    the explicit fork, D2a) and publish that.
    """
    version = await authoring_queries.get_version(db, version_id)
    if version is None:
        raise NotFoundError(f"CareerPathVersion {version_id} not found")
    if version.status == "published":
        raise ConflictError(
            "career_path_version_published: this version is published and frozen — "
            "create a new version to edit the path (Gap 3 versioning)"
        )
    return version


async def _require_authoring_version(
    db: AsyncSession, career_path_id: UUID
) -> CareerPathVersion:
    """The version authoring reads/writes for this path (latest draft, else
    the latest published pre-fork)."""
    version = await authoring_queries.get_current_authoring_version(db, career_path_id)
    if version is None:
        raise NotFoundError(f"No version found for career path {career_path_id}")
    return version


async def list_career_paths_for_org(
    db: AsyncSession, organization_id: UUID, *, include_archived: bool = False
) -> list[CareerPathAuthoring]:
    rows = await authoring_queries.list_career_paths_for_org(
        db, organization_id, include_archived=include_archived
    )
    path_ids = [row.id for row in rows]
    stage_counts = await authoring_queries.list_path_stage_counts(db, path_ids)
    course_counts = await authoring_queries.list_path_course_counts(db, path_ids)
    return [
        _to_authoring(
            row,
            stage_count=stage_counts.get(row.id, 0),
            course_count=course_counts.get(row.id, 0),
        )
        for row in rows
    ]


async def get_career_path(db: AsyncSession, career_path_id: UUID) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    stage_counts = await authoring_queries.list_path_stage_counts(db, [path.id])
    course_counts = await authoring_queries.list_path_course_counts(db, [path.id])
    return _to_authoring(
        path,
        stage_count=stage_counts.get(path.id, 0),
        course_count=course_counts.get(path.id, 0),
    )


async def get_path_impact(db: AsyncSession, career_path_id: UUID) -> CareerPathImpactRead:
    """Blast radius of editing a path (Gap 3 §2.1).

    Served to the manager BEFORE a mutation on a published path so the edit
    is informed instead of silent. Meaningless on draft paths (no students
    can be enrolled), but harmless — the counts are all zero.
    """
    await _require_path(db, career_path_id)
    active, rows = await authoring_queries.get_path_impact(db, career_path_id)
    return CareerPathImpactRead(
        career_path_id=career_path_id,
        active_enrollments=active,
        stages=[
            CareerPathImpactStage(
                stage_id=stage_id,
                position=position,
                title=title,
                students_in_stage=in_stage,
                students_not_completed=not_completed,
            )
            for stage_id, position, title, in_stage, not_completed in rows
        ],
    )


# Gap 3 §2.2: enforcement loosen/tighten ordering (advisory < soft < hard).
_ENFORCEMENT_RANK = {"advisory": 0, "soft": 1, "hard": 2}


def classify_path_edit(
    *,
    mutation: str,
    active_enrollments: int,
    stage_students_not_completed: int | None = None,
    is_required: bool | None = None,
    is_required_before: bool | None = None,
    min_optional_before: int | None = None,
    min_optional_after: int | None = None,
    enforcement_before: str | None = None,
    enforcement_after: str | None = None,
    max_student_stage_position: int | None = None,
    new_stage_position: int | None = None,
) -> str:
    """Classify a path edit as ``"safe"`` or ``"breaking"`` (Gap 3 §2.2).

    The Gap-3 taxonomy: a change is BREAKING when an in-flight student is
    worse off — work added to a stage they have not finished, the goal
    moved (quota raised / path lengthened), or a stage they could enter
    locks. Safe edits never make anyone worse off.

    * ``mutation`` — one of ``add_course``, ``update_course``,
      ``update_stage``, ``create_stage``, ``delete_stage``,
      ``reorder_stages``.
    * ``active_enrollments`` — path-level count of walking students; 0 ⇒
      nothing can be breaking.
    * ``stage_students_not_completed`` — impact count for the TARGET stage
      (from :func:`get_path_impact`).
    * ``max_student_stage_position`` — highest stage position any active
      enrollment has reached (the last ``students_in_stage > 0`` position;
      ``None`` when no student has reached any stage).

    This is the advisory half of the plan; chunk 5 (versioning) will turn
    breaking edits on a published path into a hard fork-or-409 gate. The
    classification must stay in sync with that gate's rules.
    """
    if active_enrollments <= 0:
        return "safe"

    if mutation == "add_course":
        # Optional adds are extra work the student may ignore; required adds
        # to a stage they still must pass are imposed work.
        return "breaking" if is_required and (stage_students_not_completed or 0) > 0 else "safe"

    if mutation == "update_course":
        flipped_to_required = is_required is True and is_required_before is False
        return (
            "breaking"
            if flipped_to_required and (stage_students_not_completed or 0) > 0
            else "safe"
        )

    if mutation == "update_stage":
        quota_raised = (
            min_optional_before is not None
            and min_optional_after is not None
            and min_optional_after > min_optional_before
        )
        tightened = (
            enforcement_before is not None
            and enforcement_after is not None
            and _ENFORCEMENT_RANK[enforcement_after] > _ENFORCEMENT_RANK[enforcement_before]
        )
        # Title/description edits and any loosening are safe.
        return (
            "breaking"
            if (quota_raised or tightened) and (stage_students_not_completed or 0) > 0
            else "safe"
        )

    if mutation == "create_stage":
        # Appending past every student's current position does not move their
        # goal; inserting at/before it does.
        return (
            "breaking"
            if max_student_stage_position is not None
            and new_stage_position is not None
            and new_stage_position <= max_student_stage_position
            else "safe"
        )

    if mutation in ("delete_stage", "reorder_stages"):
        # Deleting or reordering always rewrites the sequence under students
        # already walking it (delete is separately guarded by latched
        # progress; the classification is about the fairness signal).
        return "breaking"

    return "safe"


async def list_path_versions(
    db: AsyncSession, career_path_id: UUID
) -> list[CareerPathVersionRead]:
    """All versions of a path, newest first (Gap 3 manager surface)."""
    await _require_path(db, career_path_id)
    versions = await authoring_queries.list_versions(db, career_path_id)
    from abridgeai.features.identity.api import public as identity_api

    users = await identity_api.get_users_by_ids(
        db, [v.updated_by for v in versions if v.published_at is not None and v.updated_by]
    )
    return [
        CareerPathVersionRead.model_validate(
            {
                **version.__dict__,
                "published_by": version.updated_by
                if version.published_at is not None
                else None,
                "published_by_name": users[version.updated_by].display_name
                if version.updated_by in users
                else None,
            }
        )
        for version in versions
    ]


async def list_career_path_courses(
    db: AsyncSession, career_path_id: UUID, *, version_id: UUID | None = None
) -> list[CareerPathCourseAuthoring]:
    await _require_path(db, career_path_id)
    if version_id is not None:
        version = await authoring_queries.get_version(db, version_id)
        if version is None or version.career_path_id != career_path_id:
            raise NotFoundError("career_path_version_not_found")
    rows = await authoring_queries.list_authoring_career_path_courses(
        db, career_path_id, version_id=version_id
    )
    return [CareerPathCourseAuthoring.model_validate(row) for row in rows]


async def create_career_path(
    db: AsyncSession, payload: CareerPathCreate, actor: CurrentUser
) -> CareerPathAuthoring:
    """Create a career path in the actor's primary organization.

    ``organization_id`` is server-derived from the bearer token to match
    the contract used by ``POST /teacher/courses``: a manager in Org A
    cannot create a path in Org B by sending a forged payload.
    """
    org_id = await get_user_primary_organization_id(db, actor.user_id)
    if org_id is None:
        raise AppError(
            f"User {actor.user_id} has no primary organization; cannot create a career path."
        )
    path = CareerPath(
        organization_id=org_id,
        org_unit_id=payload.org_unit_id,
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
        status="draft",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(path)
    await flush_or_conflict(db)
    await db.refresh(path)
    # Gap 3 (0074): a new path starts with a draft v1 — stages/items always
    # hang off a version, and publishing freezes it.
    version = CareerPathVersion(
        career_path_id=path.id,
        version_no=1,
        status="draft",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(version)
    await flush_or_conflict(db)
    return _to_authoring(path)


async def list_course_candidates(
    db: AsyncSession, career_path_id: UUID
) -> list[CareerPathCourseCandidate]:
    """Full org course catalogue (ANY status) for the attach-to-path picker.

    The learner ``/courses`` endpoint returns only published courses, but a
    draft path may hold draft/archived courses — the publish gate
    (``validate_path_for_publish``) re-checks every link when the path goes
    live. So the picker shows the path's whole organization, letting the
    manager build the skeleton before courses are published.
    """
    path = await _require_path(db, career_path_id)
    courses = await courses_api.list_courses_by_org(db, path.organization_id)
    return [
        CareerPathCourseCandidate(
            id=c.id, title=c.title, slug=c.slug, status=c.status
        )
        for c in courses
    ]


async def update_career_path(
    db: AsyncSession,
    career_path_id: UUID,
    payload: CareerPathUpdate,
    actor: CurrentUser,
) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(path, key, value)
    path.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(path)
    return _to_authoring(path)


async def add_course_to_path(
    db: AsyncSession,
    career_path_id: UUID,
    course_id: UUID,
    *,
    stage_id: UUID,
    position: int | None,
    is_required: bool,
    satisfied_by: str = "completion",
    actor: CurrentUser,
) -> CareerPathCourseAuthoring:
    """Attach a course to ONE stage of the path.

    No enrollee backfill: under Pattern B (lazy enrollment) adding a required
    course to a path must not silently create course enrollments for everyone
    already on it. Students pick it up via the Start endpoint when its stage
    is unlocked for them. The eager fan-out that used to live here was
    removed with ``_autoenroll_required_courses``.

    The published-course requirement applies ONLY to published paths: a draft
    path has no enrolled students, so a draft/archived course cannot break
    anything there — it lets the manager build the skeleton and slot draft
    courses into stages before the path goes live. The publish gate
    (``stage_course_not_published``) re-checks every link when the path is
    published, so no integrity is lost on a live path.
    """
    del actor
    path = await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    stage = await authoring_queries.get_stage(db, stage_id)
    if stage is None or stage.version_id != version.id:
        raise NotFoundError(f"Stage {stage_id} not found in career path {career_path_id}")
    if not await authoring_queries.course_belongs_to_org(db, course_id, path.organization_id):
        raise AppError(
            f"Course {course_id} does not belong to organization {path.organization_id} — "
            "only courses of this organization can be attached"
        )
    if path.status == "published" and not await authoring_queries.course_is_published_in_org(
        db, course_id, path.organization_id
    ):
        # Name the course and its actual status: a raw uuid plus "is not
        # published" left the manager guessing WHICH pick was wrong, and
        # the sentence read as if the PATH were the unpublished thing.
        course = await courses_api.get_course_by_id(db, course_id)
        title = course.title if course is not None else str(course_id)
        state = course.status if course is not None else "missing"
        raise AppError(
            f"course_not_published: {title!r} is a {state} course. "
            f"{path.name!r} is already published, so only published courses can be "
            "added to it. Publish the course first, or add it to a new draft version "
            "of the path."
        )
    existing = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if existing is not None:
        raise AppError(f"Course {course_id} already attached to career path {career_path_id}")
    target_position = position
    if target_position is None:
        target_position = await authoring_queries.next_stage_course_position(db, stage_id)
    else:
        await _make_room_for_position(db, stage_id, target_position)

    link = CareerPathCourse(
        version_id=version.id,
        course_id=course_id,
        stage_id=stage_id,
        position=target_position,
        is_required=is_required,
        satisfied_by=satisfied_by,
    )
    db.add(link)
    await flush_or_conflict(db)
    await _validate_stage_integrity(db, stage_id)

    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    target = next(row for row in rows if row["course_id"] == course_id)
    return CareerPathCourseAuthoring.model_validate(target)


async def _make_room_for_position(db: AsyncSession, stage_id: UUID, target_position: int) -> None:
    """Shift items at/after ``target_position`` down by one, within ONE stage.

    Two-phase through ``_OFFSET`` because ``(stage_id, position)`` is UNIQUE:
    a naive sequential increment collides with the row it is about to move.
    """
    links = await authoring_queries.list_stage_course_links(db, stage_id)
    affected = [link for link in links if link.position >= target_position]
    if not affected:
        return
    for idx, link in enumerate(affected):
        link.position = _OFFSET + idx
    await flush_or_conflict(db)
    for idx, link in enumerate(affected, start=1):
        link.position = target_position + idx
    await flush_or_conflict(db)


async def remove_course_from_path(
    db: AsyncSession,
    career_path_id: UUID,
    course_id: UUID,
    actor: CurrentUser,
) -> None:
    del actor
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    link = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if link is None:
        raise NotFoundError(f"Course {course_id} not attached to career path {career_path_id}")
    await db.delete(link)
    await db.flush()


async def reorder_courses_in_path(
    db: AsyncSession,
    career_path_id: UUID,
    course_ids: list[UUID],
    actor: CurrentUser,
) -> list[CareerPathCourseAuthoring]:
    """Reorder courses WITHIN their stages.

    ``course_ids`` is the full set of the path's courses; each keeps its own
    ``stage_id`` and is renumbered 1..n inside that stage. Two-phase via
    ``_OFFSET`` because ``(stage_id, position)`` is UNIQUE.

    To move a course to a DIFFERENT stage use :func:`move_course_to_stage` —
    that mutates two position sequences and needs both offset.
    """
    del actor
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    existing_links = await authoring_queries.list_path_course_links(db, career_path_id)
    existing_by_course = {link.course_id: link for link in existing_links}
    if set(course_ids) != set(existing_by_course):
        raise AppError(f"reorder course_ids must match existing path courses for {career_path_id}")

    for idx, course_id in enumerate(course_ids):
        existing_by_course[course_id].position = _OFFSET + idx
    await flush_or_conflict(db)

    # Renumber per stage, preserving the caller's relative order.
    per_stage: dict[UUID, int] = {}
    for course_id in course_ids:
        link = existing_by_course[course_id]
        per_stage[link.stage_id] = per_stage.get(link.stage_id, 0) + 1
        link.position = per_stage[link.stage_id]
    await flush_or_conflict(db)

    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    return [CareerPathCourseAuthoring.model_validate(row) for row in rows]


async def move_course_to_stage(
    db: AsyncSession,
    career_path_id: UUID,
    course_id: UUID,
    *,
    stage_id: UUID,
    position: int | None,
) -> list[CareerPathCourseAuthoring]:
    """Move one course from its current stage to ``stage_id``.

    **This mutates TWO ``(stage_id, position)`` sequences**, which is why it
    is not a drop-in for ``reorder_courses_in_path``'s single-sequence
    ``_OFFSET`` dance. Offsetting only the target leaves a hole in the source;
    offsetting only the source collides in the target. So: park BOTH stages'
    items in the offset band, then reindex both, inside one transaction.

    A same-stage call is a plain reposition and is handled by the same code.
    """
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    stage = await authoring_queries.get_stage(db, stage_id)
    if stage is None or stage.version_id != version.id:
        raise NotFoundError(f"Stage {stage_id} not found in career path {career_path_id}")
    link = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if link is None:
        raise NotFoundError(f"Course {course_id} not attached to career path {career_path_id}")

    source_stage_id = link.stage_id
    source = await authoring_queries.list_stage_course_links(db, source_stage_id)
    target = (
        source
        if source_stage_id == stage_id
        else await authoring_queries.list_stage_course_links(db, stage_id)
    )

    # Phase 1: park every row of BOTH sequences in the offset band so no
    # intermediate assignment can collide with a live position.
    parked = {id(row): row for row in [*source, *target]}.values()
    for idx, row in enumerate(parked):
        row.position = _OFFSET + idx
    await flush_or_conflict(db)

    # Phase 2: reindex the source without the moved row, then the target with
    # it inserted at the requested slot.
    remaining = [row for row in source if row.course_id != course_id]
    if source_stage_id != stage_id:
        for idx, row in enumerate(remaining, start=1):
            row.position = idx
        target_rows = [row for row in target if row.course_id != course_id]
    else:
        target_rows = remaining

    slot = len(target_rows) + 1 if position is None else max(1, min(position, len(target_rows) + 1))
    target_rows.insert(slot - 1, link)
    link.stage_id = stage_id
    for idx, row in enumerate(target_rows, start=1):
        row.position = idx
    await flush_or_conflict(db)

    await _validate_stage_integrity(db, source_stage_id)
    await _validate_stage_integrity(db, stage_id)

    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    return [CareerPathCourseAuthoring.model_validate(row) for row in rows]


async def update_path_course(
    db: AsyncSession,
    career_path_id: UUID,
    course_id: UUID,
    *,
    is_required: bool | None = None,
    satisfied_by: str | None = None,
) -> list[CareerPathCourseAuthoring]:
    """Patch the policy flags on an attached course."""
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    link = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if link is None:
        raise NotFoundError(f"Course {course_id} not attached to career path {career_path_id}")

    if is_required is not None:
        link.is_required = is_required
    if satisfied_by is not None:
        link.satisfied_by = satisfied_by
    await flush_or_conflict(db)

    # Must run AFTER the flush: optional -> required shrinks optional_count.
    await _validate_stage_integrity(db, link.stage_id)

    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    return [CareerPathCourseAuthoring.model_validate(row) for row in rows]


# --- stage CRUD -------------------------------------------------------


async def list_path_stages(
    db: AsyncSession, career_path_id: UUID, *, version_id: UUID | None = None
) -> list[CareerPathStageAuthoring]:
    await _require_path(db, career_path_id)
    if version_id is not None:
        version = await authoring_queries.get_version(db, version_id)
        if version is None or version.career_path_id != career_path_id:
            raise NotFoundError("career_path_version_not_found")
    stages = await authoring_queries.list_path_stages(
        db, career_path_id, version_id=version_id
    )
    return [await _to_stage_authoring(db, stage) for stage in stages]


async def _to_stage_authoring(db: AsyncSession, stage: CareerPathStage) -> CareerPathStageAuthoring:
    # career_path_id is identity-level, living on the version — resolved via
    # the version row (the stage relationship is not eager-loaded in async).
    version = await db.get(CareerPathVersion, stage.version_id)
    return CareerPathStageAuthoring(
        id=stage.id,
        career_path_id=version.career_path_id if version is not None else stage.version_id,
        position=stage.position,
        title=stage.title,
        description=stage.description,
        min_optional_to_complete=stage.min_optional_to_complete,
        unlock_policy=stage.unlock_policy,
        enforcement=stage.enforcement,
        course_count=await authoring_queries.count_stage_courses(db, stage.id),
    )


async def create_stage(
    db: AsyncSession,
    career_path_id: UUID,
    payload: CareerPathStageCreate,
    actor: CurrentUser,
) -> CareerPathStageAuthoring:
    """Create a stage — EMPTY is valid, including on a published path.

    Deliberately does not enforce "a stage has >= 1 course": the authoring
    flow is create-then-fill, so that check belongs on the publish gate. With
    it here you could never add a second stage to a published path.
    """
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    target_position = payload.position
    if target_position is None:
        target_position = await authoring_queries.next_stage_position(db, career_path_id)
    else:
        await _make_room_for_stage_position(db, career_path_id, target_position)

    stage = CareerPathStage(
        version_id=version.id,
        position=target_position,
        title=payload.title,
        description=payload.description,
        min_optional_to_complete=payload.min_optional_to_complete,
        unlock_policy=payload.unlock_policy,
        enforcement=payload.enforcement,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(stage)
    await flush_or_conflict(db)
    await db.refresh(stage)
    return await _to_stage_authoring(db, stage)


async def _make_room_for_stage_position(
    db: AsyncSession, career_path_id: UUID, target_position: int
) -> None:
    stages = await authoring_queries.list_path_stages(db, career_path_id)
    affected = [s for s in stages if s.position >= target_position]
    if not affected:
        return
    for idx, stage in enumerate(affected):
        stage.position = _OFFSET + idx
    await flush_or_conflict(db)
    for idx, stage in enumerate(affected, start=1):
        stage.position = target_position + idx
    await flush_or_conflict(db)


async def _require_stage(db: AsyncSession, career_path_id: UUID, stage_id: UUID) -> CareerPathStage:
    stage = await authoring_queries.get_stage(db, stage_id)
    if stage is None:
        raise NotFoundError(f"Stage {stage_id} not found in career path {career_path_id}")
    version = await _require_authoring_version(db, career_path_id)
    if stage.version_id != version.id:
        raise NotFoundError(f"Stage {stage_id} not found in career path {career_path_id}")
    return stage


async def update_stage(
    db: AsyncSession,
    career_path_id: UUID,
    stage_id: UUID,
    payload: CareerPathStageUpdate,
    actor: CurrentUser,
) -> CareerPathStageAuthoring:
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    stage = await _require_stage(db, career_path_id, stage_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(stage, key, value)
    stage.updated_by = actor.user_id
    await flush_or_conflict(db)
    # Integrity class: a quota above the stage's optional count makes the
    # stage — and therefore the path — unfinishable. Checked on EVERY
    # mutation, unlike the completeness checks at the publish gate.
    await _validate_stage_integrity(db, stage_id)
    await db.refresh(stage)
    return await _to_stage_authoring(db, stage)


async def delete_stage(
    db: AsyncSession,
    career_path_id: UUID,
    stage_id: UUID,
    actor: CurrentUser,
) -> None:
    """Soft-delete a stage — blocked when it still holds courses OR latched progress.

    Blocking on courses alone is not enough: a manager could move every
    course out and then delete a stage students had already LATCHED, which
    orphans the latch rows and silently changes the progress denominator, so
    a student's bar jumps or slides backward without them doing anything.
    """
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    stage = await _require_stage(db, career_path_id, stage_id)

    if await authoring_queries.count_stage_courses(db, stage_id) > 0:
        raise ConflictError(
            "stage_in_use: this stage still contains courses — move or remove them first"
        )
    if await authoring_queries.has_latched_stage_progress(db, stage_id):
        raise ConflictError(
            "stage_in_use: at least one student has already completed this stage; "
            "deleting it would change their recorded progress"
        )

    stage.deleted_at = datetime.now(tz=UTC)
    stage.deleted_by = actor.user_id
    stage.updated_by = actor.user_id
    await flush_or_conflict(db)
    await _renumber_stages(db, career_path_id)


async def _renumber_stages(db: AsyncSession, career_path_id: UUID) -> None:
    """Close gaps left by a deletion so positions stay 1..n contiguous."""
    stages = await authoring_queries.list_path_stages(db, career_path_id)
    if not stages:
        return
    for idx, stage in enumerate(stages):
        stage.position = _OFFSET + idx
    await flush_or_conflict(db)
    for idx, stage in enumerate(stages, start=1):
        stage.position = idx
    await flush_or_conflict(db)


async def reorder_stages(
    db: AsyncSession,
    career_path_id: UUID,
    stage_ids: list[UUID],
    actor: CurrentUser,
) -> CareerPathStageReorderResult:
    """Reorder stages, WARNING rather than rewriting unlock policy.

    Reorder deliberately does not normalise ``unlock_policy``. Rewriting it
    would silently edit manager intent and could not be undone by moving the
    stage back. Instead it reports what the reorder changes in *effective*
    unlock for students with active enrollments:

    * a non-``always`` stage moved INTO position 1 becomes unconditionally
      unlocked (position 1 is an implicit override);
    * the stage moved OUT of position 1 starts obeying its stored policy,
      which can re-lock a stage students are working in right now.
    """
    del actor
    await _require_path(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    await _require_draft_version(db, version.id)
    stages = await authoring_queries.list_path_stages(db, career_path_id)
    by_id = {stage.id: stage for stage in stages}
    if set(stage_ids) != set(by_id):
        raise AppError(f"reorder stage_ids must match existing stages of path {career_path_id}")

    old_first = next((s.id for s in stages if s.position == 1), None)
    warnings: list[StageReorderWarning] = []
    new_first_id = stage_ids[0]
    if old_first != new_first_id:
        new_first = by_id[new_first_id]
        if new_first.unlock_policy != "always":
            warnings.append(
                StageReorderWarning(
                    stage_id=new_first_id,
                    code="stage_becomes_implicitly_unlocked",
                    message=(
                        f"Stage moved to position 1 keeps unlock_policy "
                        f"'{new_first.unlock_policy}', but position 1 is always "
                        "unlocked — students will be able to start it immediately."
                    ),
                )
            )
        if old_first is not None:
            previous_first = by_id[old_first]
            if previous_first.unlock_policy != "always":
                warnings.append(
                    StageReorderWarning(
                        stage_id=old_first,
                        code="stage_may_become_locked",
                        message=(
                            f"Stage moved out of position 1 will now enforce "
                            f"unlock_policy '{previous_first.unlock_policy}', which may "
                            "re-lock it for students currently working in it."
                        ),
                    )
                )

    for idx, stage_id in enumerate(stage_ids):
        by_id[stage_id].position = _OFFSET + idx
    await flush_or_conflict(db)
    for idx, stage_id in enumerate(stage_ids, start=1):
        by_id[stage_id].position = idx
    await flush_or_conflict(db)

    ordered = await authoring_queries.list_path_stages(db, career_path_id)
    return CareerPathStageReorderResult(
        stages=[await _to_stage_authoring(db, stage) for stage in ordered],
        warnings=warnings,
    )


# --- validation: two classes ------------------------------------------


async def _validate_stage_integrity(db: AsyncSession, stage_id: UUID) -> None:
    """INTEGRITY check — runs on every mutation.

    Only catches states that make the path **unfinishable**: a
    ``min_optional_to_complete`` above the number of optional courses
    actually in the stage can never be satisfied, so the stage can never
    complete and every later stage stays locked forever.
    """
    stage = await authoring_queries.get_stage(db, stage_id)
    if stage is None:
        return
    links = await authoring_queries.list_stage_course_links(db, stage_id)
    optional_count = sum(1 for link in links if not link.is_required)
    if stage.min_optional_to_complete > optional_count:
        raise AppError(
            f"stage_min_optional_exceeds_optional_count: stage requires "
            f"{stage.min_optional_to_complete} optional course(s) but only "
            f"{optional_count} optional course(s) are attached — the stage "
            "could never be completed"
        )


async def validate_path_for_publish(db: AsyncSession, career_path_id: UUID) -> list[str]:
    """COMPLETENESS checks — publish gate only. Returns warning strings.

    Hard failures raise; soft findings are returned so the UI can surface
    them without blocking a deliberate publish.
    """
    stages = await authoring_queries.list_path_stages(db, career_path_id)
    if not stages:
        raise AppError("path_has_no_stages: add at least one stage before publishing")

    warnings: list[str] = []
    path = await _require_path(db, career_path_id)
    for stage in stages:
        links = await authoring_queries.list_stage_course_links(db, stage.id)
        if not links:
            raise AppError(
                f"stage_has_no_courses: stage at position {stage.position} is empty — "
                "every stage must contain at least one course before publishing"
            )
        required = [link for link in links if link.is_required]
        for link in links:
            if not await authoring_queries.course_is_published_in_org(
                db, link.course_id, path.organization_id
            ):
                raise AppError(
                    f"stage_course_not_published: course {link.course_id} in stage "
                    f"position {stage.position} is not a published course of this organization"
                )
            # A published course with no gradeable unit can never be completed:
            # the D2 writer refuses to promote an empty course, so `satisfied`
            # stays false forever. As a REQUIRED course that locks the stage and
            # every stage behind it permanently; as an optional one it can still
            # be counted toward `min_optional_to_complete` and make the quota
            # unreachable. Publishing is the last point where a manager can be
            # told before a student is stuck, so both are hard failures.
            units = await enrollments_api.count_course_gradeable_units(db, course_id=link.course_id)
            if units == 0:
                raise AppError(
                    f"stage_course_has_no_gradeable_units: course {link.course_id} in "
                    f"stage position {stage.position} has no published lessons, quizzes "
                    "or interviews, so no student could ever complete it"
                )
        if not required and stage.min_optional_to_complete == 0:
            warnings.append(
                f"stage_completes_immediately: stage at position {stage.position} has no "
                "required courses and no optional quota, so it completes with no work done"
            )
        if path.max_concurrent == 1 and len(required) >= _CAP_ONE_MANY_REQUIRED:
            warnings.append(
                f"cap_one_with_many_required: stage at position {stage.position} has "
                f"{len(required)} required courses while the path caps concurrency at 1"
            )
    return warnings


async def publish_path(
    db: AsyncSession, career_path_id: UUID, actor: CurrentUser
) -> CareerPathAuthoring:
    """Publish a path after the COMPLETENESS gate passes.

    This is where "every path has >= 1 stage" and "every stage has >= 1
    course" are enforced — not on the mutation path, where they would make
    create-then-fill authoring impossible.

    Gap 3: publishing freezes the path's authoring VERSION (status →
    published, published_at = now). Existing enrollments stay pinned to
    whatever version they started on; new enrollments pin to this one.
    """
    path = await _require_path(db, career_path_id)
    if path.status == "archived":
        raise AppError(f"Cannot publish archived career path {career_path_id}")
    await validate_path_for_publish(db, career_path_id)
    version = await _require_authoring_version(db, career_path_id)
    if version.status != "published":
        version.status = "published"
        version.published_at = datetime.now(tz=UTC)
        version.updated_by = actor.user_id
    path.status = "published"
    path.updated_by = actor.user_id
    await db.flush()
    await db.refresh(path)
    return _to_authoring(path)


async def create_path_version(
    db: AsyncSession, career_path_id: UUID, actor: CurrentUser
) -> CareerPathVersion:
    """Copy-on-write fork (Gap 3 D2a explicit): clone the latest PUBLISHED
    version's stages + course items into a new DRAFT version.

    Only one draft may be in flight per path — a second fork while one
    exists is a 409 (editing two drafts at once is a merge problem nobody
    wants). New version_no = max + 1.
    """
    await _require_path(db, career_path_id)
    source = await authoring_queries.get_published_version(db, career_path_id)
    if source is None:
        raise ConflictError(
            "career_path_version_no_source: this path has no published version to fork — "
            "publish it first"
        )
    versions = await authoring_queries.list_versions(db, career_path_id)
    if any(v.status == "draft" for v in versions):
        raise ConflictError(
            "career_path_version_draft_exists: a draft version is already being edited — "
            "publish or discard it before forking again"
        )

    new_version = CareerPathVersion(
        career_path_id=career_path_id,
        version_no=await authoring_queries.next_version_no(db, career_path_id),
        status="draft",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(new_version)
    await db.flush()

    source_stages = await authoring_queries.list_stages_for_version(db, source.id)
    stage_id_map: dict[UUID, UUID] = {}
    for source_stage in source_stages:
        clone_stage = CareerPathStage(
            version_id=new_version.id,
            position=source_stage.position,
            title=source_stage.title,
            description=source_stage.description,
            min_optional_to_complete=source_stage.min_optional_to_complete,
            unlock_policy=source_stage.unlock_policy,
            enforcement=source_stage.enforcement,
            created_by=actor.user_id,
            updated_by=actor.user_id,
        )
        db.add(clone_stage)
        await db.flush()
        stage_id_map[source_stage.id] = clone_stage.id

    source_items = await authoring_queries.list_items_for_version(db, source.id)
    for item in source_items:
        if item.stage_id not in stage_id_map:
            # Item points at a stage deleted from the source version; it
            # cannot be cloned (defensive — published versions normally
            # cannot lose stages while holding items).
            continue
        db.add(
            CareerPathCourse(
                version_id=new_version.id,
                course_id=item.course_id,
                stage_id=stage_id_map[item.stage_id],
                position=item.position,
                is_required=item.is_required,
                satisfied_by=item.satisfied_by,
            )
        )
    await db.flush()
    await db.refresh(new_version)
    return new_version


async def archive_path(
    db: AsyncSession, career_path_id: UUID, actor: CurrentUser
) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    if path.status == "archived":
        raise AppError(f"CareerPath {career_path_id} is already archived")
    path.status = "archived"
    path.updated_by = actor.user_id
    await db.flush()
    await db.refresh(path)
    return _to_authoring(path)


async def soft_delete_path(db: AsyncSession, career_path_id: UUID, actor: CurrentUser) -> None:
    path = await _require_path(db, career_path_id)
    if path.status != "draft":
        raise AppError(
            "published_or_archived_career_path_cannot_be_deleted: archive it instead"
        )
    await soft_delete_cascade(db, path, actor_id=actor.user_id)


__all__ = [
    "add_course_to_path",
    "archive_path",
    "create_career_path",
    "create_path_version",
    "create_stage",
    "delete_stage",
    "get_career_path",
    "list_career_path_courses",
    "list_career_paths_for_org",
    "list_course_candidates",
    "list_path_stages",
    "move_course_to_stage",
    "publish_path",
    "remove_course_from_path",
    "reorder_courses_in_path",
    "reorder_stages",
    "soft_delete_path",
    "update_career_path",
    "update_stage",
    "validate_path_for_publish",
]
