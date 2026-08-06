"""Courses authoring router -- teacher CRUD on courses + sub-resources (T3.7).

Mounted at ``/api/v1/teacher`` by T3.10 integration. Every endpoint enforces
either a global permission (``course.create``) or a course-scoped permission
chain that walks UP from the path-param sub-resource to the owning course
(``require_*_authoring_access`` from :mod:`._deps`).

**FIX-SEC-1 invariant** (Reconciliation §A9 + §E4) -- the legacy
``backend/app/routes/teacher/courses_router.py`` authenticated sub-resource
endpoints (modules, lessons, module_items, resources) with the bare
identity dep, so any authenticated user could PATCH or DELETE another
teacher's sub-resources. Every endpoint in this file uses the corresponding
wrapper from :mod:`abridgeai.features.courses.routers._deps`, so the only
path that can mutate course state is one whose principal also holds
course-scoped permissions on the resource's owning course (or owns the
course outright).

Scope deviation -- T3.7 plan body lists ~25 endpoints. Only the ~13
endpoints in this file have matching public helpers in
:mod:`features.courses.services.authoring` (T3.5 frozen for T3.7). The
remaining read endpoints (GET ``/teacher/courses``, GET
``/teacher/courses/{id}``, GET ``/teacher/courses/{id}/content``, etc.)
require new authoring read services that T3.5 did not ship, and adding
those is explicitly out of scope per T3.7's "Do NOT touch T3.5 services
source" guardrail. They are deferred to a follow-up task; the security
invariant (FIX-SEC-1) is satisfied for every WRITE endpoint, which is the
gap the audit identified.

Architectural rules honoured:

* Routers ↔ services only (no ``queries.*`` imports here -- import-linter
  contract #2). Resolver SQL lives in :mod:`._deps`, NOT here.
* Services flush; the router commits after a successful write.
* Domain exceptions (``NotFoundError``, ``AppError``) are mapped to HTTP
  errors locally -- services stay HTTP-agnostic.
"""

from __future__ import annotations

from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
    require_permission,
)
from abridgeai.features.access_control.queries.permissions import load_course_permissions
from abridgeai.features.courses.routers._deps import (
    require_lesson_authoring_access,
    require_module_authoring_access,
    require_module_item_authoring_access,
    require_outcome_authoring_access,
    require_resource_authoring_access,
)
from abridgeai.features.courses.schemas import (
    CourseAuthoring,
    CourseContentAuthoring,
    CourseCreate,
    CourseLearningOutcomeAuthoring,
    CourseLearningOutcomeCreate,
    CourseLearningOutcomeUpdate,
    CourseRosterRead,
    CourseUpdate,
    LessonAuthoring,
    LessonCreate,
    LessonOutline,
    LessonResourceAuthoring,
    LessonResourceCreate,
    LessonUpdate,
    ModuleAuthoring,
    ModuleCreate,
    ModuleItemAuthoring,
    ModuleItemReorder,
    ModuleItemUpdate,
    ModulePrerequisiteSet,
    ModuleReorder,
    ModuleUpdate,
    OutlineSection,
    RosterStudentRead,
    SlugAvailability,
    StreamUrlResponse,
    TeacherDashboardStats,
)
from abridgeai.features.courses.services import authoring as authoring_service
from abridgeai.features.quizzes.ai.outline import build_lesson_outline

# Whitelist of ``content_role`` values surfaced by ``OutlineSection``.
# Anything else returned by the chunk metadata is coerced to "body" so
# the response stays inside the literal type defined by the schema.
_ALLOWED_OUTLINE_ROLES: frozenset[str] = frozenset({"body", "summary", "review", "front_matter"})

router = APIRouter(prefix="/teacher", tags=["courses-authoring"])

_REQUIRE_CREATE = require_permission("course.create")
_REQUIRE_AUTHORING_LIST = require_any_permission("course.read.draft", "course.create")
_REQUIRE_COURSE_UPDATE = require_course_permission("course_id", "course.update")
_REQUIRE_COURSE_PUBLISH = require_course_permission("course_id", "course.publish")

# Fields on CourseUpdate a TEACHER (course.update) may patch: the course
# description, the study-time estimate, and their own contact details.
# User decision 2026-08-06.
_TEACHER_PATCHABLE_COURSE_FIELDS: frozenset[str] = frozenset(
    {
        "description",
        "estimated_minutes",
        "contact_email",
        "contact_phone",
        "contact_website_url",
        "contact_social_url",
    }
)

# Everything else on CourseUpdate is manager-owned (needs course.delete):
# title, slug, status, level, org_unit_id, thumbnail_object_id,
# expected_completion_days, enrollment_cap.
#
# DERIVED from the schema rather than hand-listed, so a field added to
# CourseUpdate later defaults to manager-only instead of silently becoming
# teacher-writable — fail closed, not open.
_MANAGER_ONLY_COURSE_FIELDS: frozenset[str] = (
    frozenset(CourseUpdate.model_fields) - _TEACHER_PATCHABLE_COURSE_FIELDS
)
# Course deletion is manager-owned. ``allow_owner=False`` kills the ownership
# short-circuit so a teacher who owns the course still cannot delete it —
# ownership grants authoring access (course.update), NOT lifecycle control.
_REQUIRE_COURSE_DELETE = require_course_permission("course_id", "course.delete", allow_owner=False)
# Learning outcomes are manager-owned (§LO split): gate on learning_outcome.manage
# and disable the owner short-circuit so a course-owning teacher (who holds
# course.update but NOT learning_outcome.manage) cannot author LOs.
_REQUIRE_OUTCOME_CREATE = require_course_permission(
    "course_id", "learning_outcome.manage", allow_owner=False
)
_REQUIRE_MODULE = require_module_authoring_access()
_REQUIRE_MODULE_ITEM = require_module_item_authoring_access()
_REQUIRE_LESSON = require_lesson_authoring_access()
_REQUIRE_RESOURCE = require_resource_authoring_access()
_REQUIRE_OUTCOME = require_outcome_authoring_access()


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": detail},
    )


def _conflict(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": detail},
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "permission_denied", "message": detail},
    )


@router.post(
    "/courses",
    response_model=CourseAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_course(
    payload: CourseCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    """Create a new course owned by the requesting principal.

    Global permission -- a teacher anywhere on the platform can create a
    course; ownership / scope is enforced on subsequent edits via
    :func:`require_course_permission`.
    """
    try:
        course = await authoring_service.create_course(db, payload, current_user)
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return course


@router.get("/courses/check-slug", response_model=SlugAvailability)
async def check_course_slug(
    slug: Annotated[str, Query(min_length=1, max_length=100)],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SlugAvailability:
    """Pre-flight check for the new-course form.

    Returns ``{"available": true}`` when ``slug`` is free in the caller's
    primary organization, ``false`` otherwise. Same auth as ``POST
    /teacher/courses`` so a 200 here implies the create attempt would not
    be rejected for permission reasons.
    """
    try:
        available = await authoring_service.check_course_slug_available(
            db, slug=slug, owner=current_user
        )
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    return SlugAvailability(available=available)


@router.get("/courses", response_model=list[CourseAuthoring])
async def list_authoring_courses(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_AUTHORING_LIST)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = False,
) -> list[CourseAuthoring]:
    """Courses the caller can author (owned + scope=course teacher assignments).

    Drafts and archived rows are visible to the author. Permission is
    intentionally lax (``course.read.draft`` OR ``course.create``) — the
    visibility filter happens in the service via owner/assignment match,
    not via permission gating, so a teacher seeing nothing is a UX
    problem rather than a 403.
    """
    return await authoring_service.list_authoring_courses_for_user(
        db, user=current_user, include_archived=include_archived
    )


@router.get("/dashboard/stats", response_model=TeacherDashboardStats)
async def get_teacher_dashboard_stats(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_AUTHORING_LIST)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeacherDashboardStats:
    """Actionable counts for the teacher dashboard's clickable widgets.

    Scoped to the caller's authorable courses (owned + assignments):
    draft courses, ungraded quiz attempts, and interview sessions awaiting
    evaluation. Same lax permission as the courses list — visibility is
    enforced in the service via owner/assignment match.
    """
    return await authoring_service.get_teacher_dashboard_stats(db, user=current_user)


@router.get("/courses/{course_id}", response_model=CourseAuthoring)
async def get_authoring_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    del current_user
    try:
        return await authoring_service.get_authoring_course(db, course_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.get(
    "/courses/{course_id}/content",
    response_model=CourseContentAuthoring,
)
async def get_authoring_course_content(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = False,
) -> CourseContentAuthoring:
    """Authoring content tree (drafts included) for ``course_id``."""
    del current_user
    try:
        tree = await authoring_service.get_authoring_content(
            db, course_id, include_archived=include_archived
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return CourseContentAuthoring.model_validate(tree)


@router.get(
    "/courses/{course_id}/roster",
    response_model=CourseRosterRead,
)
async def get_authoring_course_roster(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseRosterRead:
    """Roster of enrolled students for the teacher's "Students" page.

    Unlike ``GET /dept/courses/{id}/roster`` (HOD-scope, thin
    ``RosterEntry`` shape), this endpoint returns the envelope +
    progress/risk fields the SPA's ``RosterStudent`` type actually
    expects (the SPA was previously calling this same path and silently
    rendering an empty roster because the shapes didn't match — see
    ``queries/sql/roster_with_progress.sql``).
    """
    del current_user
    rows = await authoring_service.list_course_roster_with_progress(db, course_id)
    return CourseRosterRead(
        course_id=course_id,
        students=[RosterStudentRead.model_validate(row) for row in rows],
    )


@router.patch("/courses/{course_id}", response_model=CourseAuthoring)
async def update_course(
    course_id: UUID,
    payload: CourseUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    # Field ownership (user decision 2026-08-06). `course.update` is the
    # CONTENT permission a teacher holds on a course assigned to them; it must
    # not carry course identity, lifecycle, or delivery policy with it.
    #
    # Teacher may patch only: description, estimated_minutes and the four
    # contact_* fields (their own contact details).
    #
    # Everything else needs `course.delete` (manager/admin). `status` in
    # particular: without it a teacher could PATCH {"status": "published"} and
    # publish their own course, bypassing the manager publish gate entirely —
    # the POST /publish ROUTE is gated on `course.publish`, but this PATCH was
    # not, so the gate had a hole straight through it.
    #
    # Checked before the patch so a mixed payload either fully applies or
    # fully rejects.
    manager_only = _MANAGER_ONLY_COURSE_FIELDS & payload.model_fields_set
    if manager_only:
        course_perms = await load_course_permissions(db, current_user.user_id, course_id)
        if "course.delete" not in course_perms:
            raise _forbidden(
                "Only managers may change "
                + ", ".join(sorted(manager_only))
                + " on a course."
            )
    try:
        course = await authoring_service.update_course(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return course


@router.put("/courses/{course_id}/thumbnail", response_model=CourseAuthoring)
async def upload_course_thumbnail(
    course_id: UUID,
    request: Request,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    """Upload a course thumbnail image (JPEG/PNG/WebP/GIF, ≤ 5 MiB).

    The raw image bytes are sent as the request body with the image's MIME
    type in the ``Content-Type`` header (no multipart wrapper — matches the
    avatar upload pattern). Stores the image in object storage and points the
    course at it. Requires ``course.update`` on the course.
    """
    data = await request.body()
    content_type = request.headers.get("content-type", "application/octet-stream")
    content_type = content_type.split(";", 1)[0].strip().lower()
    try:
        course = await authoring_service.upload_course_thumbnail(
            db,
            course_id,
            data=data,
            content_type=content_type,
            uploaded_by=current_user.user_id,
        )
    except authoring_service.ThumbnailUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return course


@router.post("/courses/{course_id}/publish", response_model=CourseAuthoring)
async def publish_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_PUBLISH)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    try:
        course = await authoring_service.publish_course(db, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    # ConflictError subclasses AppError, so it MUST be caught first or the
    # gradeable-unit refusal degrades from 409 to a generic 400.
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return course


@router.post("/courses/{course_id}/archive", response_model=CourseAuthoring)
async def archive_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    try:
        course = await authoring_service.archive_course(db, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return course


@router.delete("/courses/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a course the caller can delete (reversible tombstone).

    Cascades to the course's modules/lessons/items via
    ``soft_delete_cascade``. Requires ``course.delete`` on the course.
    Returns 204 on success; 404 when the course is missing or already
    soft-deleted.
    """
    try:
        await authoring_service.delete_course(db, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@router.post(
    "/courses/{course_id}/modules",
    response_model=ModuleAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_module(
    course_id: UUID,
    payload: ModuleCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    try:
        module = await authoring_service.add_module(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return module


@router.patch("/modules/{module_id}", response_model=ModuleAuthoring)
async def update_module(
    module_id: UUID,
    payload: ModuleUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    try:
        module = await authoring_service.update_module(db, module_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return module


@router.post(
    "/modules/{module_id}/duplicate",
    response_model=ModuleAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def duplicate_module(
    module_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    """Deep-clone a whole module: the module + every item + every target.

    Creates a new ``status='draft'`` module at the end of the course, with each
    item's lesson/quiz/interview target deep-cloned into it (all unpublished /
    pending). The copy is fully independent — no rows shared with the source.
    Module prerequisites are not carried over.
    """
    try:
        module = await authoring_service.duplicate_module(db, module_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return module


# ---------------------------------------------------------------------------
# Course learning outcomes (§LO-1/2) — teacher CRUD. Positions are
# server-managed (append on create, contiguous re-index on delete); the
# ``(L.O.x)`` code is derived from position at display time.
# ---------------------------------------------------------------------------
@router.get(
    "/courses/{course_id}/outcomes",
    response_model=list[CourseLearningOutcomeAuthoring],
)
async def list_course_outcomes(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CourseLearningOutcomeAuthoring]:
    """Teacher view of a course's learning outcomes, ordered by position."""
    del current_user
    try:
        return await authoring_service.list_course_outcomes(db, course_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.post(
    "/courses/{course_id}/outcomes",
    response_model=CourseLearningOutcomeAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_course_outcome(
    course_id: UUID,
    payload: CourseLearningOutcomeCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_OUTCOME_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseLearningOutcomeAuthoring:
    """Append a learning outcome to a course (§LO-1)."""
    try:
        outcome = await authoring_service.add_course_outcome(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return outcome


@router.patch(
    "/courses/{course_id}/outcomes/{outcome_id}",
    response_model=CourseLearningOutcomeAuthoring,
)
async def update_course_outcome(
    course_id: UUID,
    outcome_id: UUID,
    payload: CourseLearningOutcomeUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_OUTCOME)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseLearningOutcomeAuthoring:
    """Edit an outcome's text (§LO-2)."""
    try:
        outcome = await authoring_service.update_course_outcome(
            db, course_id, outcome_id, payload, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return outcome


@router.delete(
    "/courses/{course_id}/outcomes/{outcome_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_course_outcome(
    course_id: UUID,
    outcome_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_OUTCOME)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete an outcome and compact positions to 1..N (§LO-2)."""
    try:
        await authoring_service.delete_course_outcome(db, course_id, outcome_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()


@router.put("/modules/{module_id}/prerequisites", response_model=ModuleAuthoring)
async def set_module_prerequisites(
    module_id: UUID,
    payload: ModulePrerequisiteSet,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleAuthoring:
    try:
        module = await authoring_service.set_module_prerequisites(
            db, module_id, payload.prerequisite_module_ids, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return module


@router.put(
    "/modules/{module_id}/items/reorder",
    response_model=list[ModuleItemAuthoring],
)
async def reorder_module_items(
    module_id: UUID,
    payload: ModuleItemReorder,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModuleItemAuthoring]:
    """Reorder ``ModuleItem`` rows under ``module_id`` (Reconciliation §A6).

    The service uses the ``_OFFSET=100_000`` two-phase swap to escape
    the ``uq_module_items_position`` unique constraint mid-update.
    """
    try:
        items = await authoring_service.reorder_module_items(
            db, module_id, payload.new_order, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return items


@router.put(
    "/courses/{course_id}/modules/reorder",
    response_model=list[ModuleAuthoring],
)
async def reorder_modules(
    course_id: UUID,
    payload: ModuleReorder,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_UPDATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModuleAuthoring]:
    """Reorder ``Module`` rows under ``course_id``.

    The service uses the ``_OFFSET=100_000`` two-phase swap to escape the
    ``uq_modules_course_position`` unique constraint mid-update (mirrors
    the module-items reorder endpoint above).
    """
    try:
        modules = await authoring_service.reorder_modules(
            db, course_id, payload.new_order, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return modules


@router.patch(
    "/module-items/{module_item_id}",
    response_model=ModuleItemAuthoring,
)
async def update_module_item(
    module_item_id: UUID,
    payload: ModuleItemUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE_ITEM)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleItemAuthoring:
    """Patch an item's ``unlock_rule_json`` (only mutable field).

    Position changes go through ``PUT /modules/{id}/items/reorder``;
    identity (lesson_id / quiz_id / interview_config_id) is immutable.
    """
    try:
        item = await authoring_service.update_module_item(db, module_item_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return item


@router.delete(
    "/module-items/{module_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_module_item(
    module_item_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE_ITEM)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a module item (un-pins from the order; target survives)."""
    try:
        await authoring_service.delete_module_item(db, module_item_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@router.post(
    "/module-items/{module_item_id}/duplicate",
    response_model=ModuleItemAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def duplicate_module_item(
    module_item_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE_ITEM)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModuleItemAuthoring:
    """Deep-clone a module item (lesson / quiz / interview) into its own module.

    The polymorphic target is fully copied as an independent draft and the new
    pin is appended at the end of the module. Duplicated content is always
    unpublished (``status='draft'``, questions ``review_status='pending'``).
    """
    try:
        item = await authoring_service.duplicate_module_item(db, module_item_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise _bad_request(str(exc)) from exc
    await db.commit()
    return item


@router.get(
    "/modules/{module_id}/lessons",
    response_model=list[LessonAuthoring],
)
async def list_module_lessons_for_authoring(
    module_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonAuthoring]:
    """List all lessons under ``module_id`` (drafts included) for authoring.

    Authoring sibling of the learner-facing ``GET /modules/{id}/lessons``
    in :mod:`courses.routers.learner`: that endpoint filters to
    published-only, which hides drafts from teachers building a quiz on
    a yet-unpublished module. The FR-5 quiz generation panel needs the
    full list, so this authoring variant skips the publish filter.
    """
    try:
        return await authoring_service.list_authoring_lessons(db, module_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.post(
    "/modules/{module_id}/lessons",
    response_model=LessonAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_lesson(
    module_id: UUID,
    payload: LessonCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MODULE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonAuthoring:
    """Create a lesson AND auto-attach a ``ModuleItem`` (Reconciliation §A5).

    The service emits both INSERTs in a single flush; the router commits
    once on success so the lesson + module_item land atomically.
    """
    try:
        lesson = await authoring_service.add_lesson(db, module_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return lesson


@router.get("/lessons/{lesson_id}", response_model=LessonAuthoring)
async def get_authoring_lesson(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonAuthoring:
    """Authoring detail for a single lesson (drafts included)."""
    del current_user
    try:
        return await authoring_service.get_authoring_lesson(db, lesson_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.get("/lessons/{lesson_id}/outline", response_model=LessonOutline)
async def get_lesson_outline(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
    slides_per_section: Annotated[int, Query(ge=1, le=20)] = 4,
    section_grouping: Annotated[Literal["auto", "fixed"], Query()] = "auto",
) -> LessonOutline:
    """Authoring outline preview (drafts visible).

    Surfaces under the teacher router (rather than the learner one) so
    the auth boundary matches the SPA's ``useLessonOutline`` consumer
    pages, and so drafts surface during course assembly.

    Phase 3 of the FR-5 schema port (T5.14): now invokes the real
    :func:`abridgeai.features.quizzes.ai.outline.build_lesson_outline`
    against the lesson's ``document_chunks``. Falls back to a single
    synthetic ``body`` section sourced from the lesson summary when no
    chunks have been ingested yet — keeps the SPA renderable for
    lessons that don't have material attached.

    ``suggested_question_count`` mirrors the legacy heuristic: 1
    question per eligible body section, capped to a 1..50 band.
    ``min_for_full_coverage`` reports the same number so the SPA can
    surface "you need at least N questions for coverage mode" copy.
    """
    del current_user
    try:
        lesson = await authoring_service.get_authoring_lesson(db, lesson_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc

    # Pure SQL + Python — safe to invoke synchronously inside the route.
    # No LLM calls, no embedding lookups, just chunk metadata grouping.
    outlines = await build_lesson_outline(
        db,
        [lesson.id],
        slides_per_section=slides_per_section,
        force_bundle=(section_grouping == "fixed"),
    )
    if outlines and outlines[0].sections:
        outline = outlines[0]
        body_sections = sum(1 for s in outline.sections if s.content_role == "body")
        # Cap suggestion to the legacy 1..50 band — the schema enforces
        # the same on ``QuizGenerationRequest.question_count``, so the
        # SPA can pre-fill the field without an extra clamp on the
        # frontend.
        suggested = max(1, min(50, body_sections or len(outline.sections)))
        # Narrow ``content_role`` (free-form string from chunk metadata)
        # down to the OutlineSection literal — anything outside the
        # whitelist falls back to "body" so the API contract stays tight.
        return LessonOutline(
            lesson_id=lesson.id,
            lesson_title=lesson.title,
            sections=[
                OutlineSection(
                    id=s.id,
                    title=s.title,
                    depth=s.depth,
                    chunk_count=len(s.chunk_ids),
                    char_count=s.char_count,
                    page_range=s.page_range,
                    content_role=cast(
                        Literal["body", "summary", "review", "front_matter"],
                        s.content_role if s.content_role in _ALLOWED_OUTLINE_ROLES else "body",
                    ),
                    preview=s.preview,
                )
                for s in outline.sections
            ],
            suggested_question_count=suggested,
            min_for_full_coverage=max(body_sections, 1),
        )

    # Fallback: lesson has no chunks yet (material not ingested).
    # Surface a single synthetic body section so the panel still
    # renders something coherent and the teacher can pick topic mode.
    return LessonOutline(
        lesson_id=lesson.id,
        lesson_title=lesson.title,
        sections=[
            OutlineSection(
                id=f"sec_{str(lesson.id)[:8]}_lesson_0",
                title=lesson.title,
                depth=0,
                chunk_count=0,
                char_count=0,
                page_range=(0, 0),
                content_role="body",
                preview=(lesson.summary or "")[:200],
            )
        ],
        suggested_question_count=0,
        min_for_full_coverage=0,
    )


@router.get(
    "/lessons/{lesson_id}/resources",
    response_model=list[LessonResourceAuthoring],
)
async def list_authoring_lesson_resources(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonResourceAuthoring]:
    """All resources attached to ``lesson_id`` (drafts + hidden included).

    The learner-side ``/lessons/{id}/resources`` endpoint applies a
    ``visible_to_students=TRUE`` filter; this authoring sibling does
    not, so the teacher can see and reorder hidden / draft resources.
    """
    del current_user
    try:
        return await authoring_service.list_authoring_lesson_resources(db, lesson_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.patch("/lessons/{lesson_id}", response_model=LessonAuthoring)
async def update_lesson(
    lesson_id: UUID,
    payload: LessonUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonAuthoring:
    try:
        lesson = await authoring_service.update_lesson(db, lesson_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return lesson


@router.post(
    "/lessons/{lesson_id}/resources",
    response_model=LessonResourceAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_lesson_resource(
    lesson_id: UUID,
    payload: LessonResourceCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonResourceAuthoring:
    try:
        resource = await authoring_service.add_lesson_resource(db, lesson_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return resource


@router.delete(
    "/lesson-resources/{resource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_lesson_resource(
    resource_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_RESOURCE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a lesson resource via :func:`soft_delete_cascade` (T0.15)."""
    try:
        await authoring_service.delete_lesson_resource(db, resource_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@router.get(
    "/lesson-resources/{resource_id}/download-url",
    response_model=StreamUrlResponse,
)
async def get_authoring_resource_download_url(
    resource_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_RESOURCE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamUrlResponse:
    """Mint a presigned GET URL for a teacher-visible resource.

    Same auth as the DELETE sibling — caller must have authoring access
    on the owning course (owner OR scope=course teacher assignment).
    Unlike the learner-side ``/lesson-resources/{id}/download-url`` this
    surfaces hidden / draft resources too.
    """
    del current_user
    try:
        url, expires_at = await authoring_service.get_authoring_resource_download_url(
            db, resource_id
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return StreamUrlResponse(stream_url=url, expires_at=expires_at)


__all__ = ["router"]
