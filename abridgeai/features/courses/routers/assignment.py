"""Courses assignment router -- HOD/Manager teacher-staffing surface (T3.8).

Mounted at ``/api/v1/dept`` by T3.10. Returns ``*Authoring`` schemas
(drafts visible -- HOD/Manager need to oversee draft work).

Permission convergence per plan §4455: every endpoint composes
``require_any_permission("course.assign_teacher", "user.role_assign",
"system.administer")`` so HOD / Manager / Admin all reach the surface.
Course-scoped sub-resource endpoints layer
:func:`require_course_permission` on top so an HOD cannot operate on a
course outside their dept (plan §4467 -- scope must match).

Scope auto-derivation (plan §4464) for ``GET /dept/courses``:

* ``role=admin, scope_kind=global``         -> no filter (all courses)
* ``role=manager, scope_kind=organization`` -> filter by ``organization_id``
* ``role=hod, scope_kind=org_unit``         -> filter by ``org_unit_id``

Most-restrictive active assignment wins (org_unit > organization >
global) so a user holding multiple roles still gets the narrowest
scope.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ConflictError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
)
from abridgeai.features.courses.schemas import (
    AssignableTeacher,
    AssignTeacherRequest,
    CourseAuthoring,
    CourseCloneRequest,
    CourseReadiness,
    CourseTeacherRoleRequest,
    CourseUpdate,
    RosterEntry,
    TeacherAssignmentCreated,
    TeacherAssignmentRead,
)
from abridgeai.features.courses.services import assignment as assignment_service
from abridgeai.features.courses.services.authoring import (
    clone_course as clone_course_service,
)
from abridgeai.features.courses.services.authoring import (
    delete_course as delete_course_service,
)
from abridgeai.features.courses.services.authoring import (
    update_course as update_course_service,
)

router = APIRouter(prefix="/dept", tags=["courses-assignment"])


async def get_arq_pool() -> object | None:
    """ARQ Redis pool dependency (email dispatch).

    Returns ``None`` until the app factory overrides it with a real
    ``ArqRedis`` pool; the notification path accepts ``None`` and simply skips
    the email enqueue (in-app notification is still written). Mirrors the
    identical dependency in the materials / quizzes / interviews routers.
    """
    return None


_REQUIRE_STAFFING = require_any_permission(
    "course.assign_teacher", "user.role_assign", "system.administer"
)
_REQUIRE_COURSE_STAFFING = require_course_permission(
    "course_id", "course.assign_teacher", "user.role_assign", "system.administer"
)
_REQUIRE_FACULTY_STAFFING = require_org_unit_permission(
    "faculty_id", "course.assign_teacher", "user.role_assign", "system.administer"
)
# Course deletion is manager-owned: only an explicit ``course.delete`` grant
# passes. ``allow_owner=False`` kills the ownership short-circuit so a
# teacher-owner cannot delete their own course through the dept surface.
_REQUIRE_COURSE_DELETE = require_course_permission("course_id", "course.delete", allow_owner=False)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


def _conflict(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": detail},
    )


_SCOPE_PRIORITY: dict[str, int] = {
    "course": 1,
    "org_unit": 2,
    "organization": 3,
    "global": 4,
}


async def _resolve_caller_scope(
    db: AsyncSession, user_id: UUID
) -> tuple[str, UUID | None, UUID | None]:
    assignments = await access_control_api.get_role_assignments_for_user(db, user_id)
    if not assignments:
        return ("global", None, None)
    most_specific = min(assignments, key=lambda a: _SCOPE_PRIORITY.get(a.scope_kind, 5))
    return (most_specific.scope_kind, most_specific.organization_id, most_specific.org_unit_id)


@router.get("/courses", response_model=list[CourseAuthoring])
async def list_dept_courses(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CourseAuthoring]:
    """List courses in the caller's auto-derived staffing scope.

    HOD (``scope_kind=org_unit``) -> dept courses; Manager
    (``scope_kind=organization``) -> org courses; Admin
    (``scope_kind=global``) -> all courses.
    """
    scope_kind, organization_id, org_unit_id = await _resolve_caller_scope(db, current_user.user_id)
    if scope_kind == "org_unit" and org_unit_id is not None:
        assignments = await access_control_api.get_role_assignments_for_user(
            db, current_user.user_id
        )
        faculty_ids = list(
            {
                UUID(str(row.org_unit_id))
                for row in assignments
                if row.scope_kind == "org_unit" and row.org_unit_id is not None
            }
        )
        return await assignment_service.list_courses_in_faculties(db, faculty_ids)

    org_filter = (
        UUID(str(organization_id)) if scope_kind == "organization" and organization_id else None
    )
    return await assignment_service.list_courses_for_organization(db, org_filter)


@router.get("/courses/{course_id}/teachers", response_model=list[TeacherAssignmentRead])
async def list_course_teachers(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TeacherAssignmentRead]:
    """Active teachers (``active_until`` IS NULL or future) for a course."""
    rows = await assignment_service.list_teachers_with_emails(db, course_id)
    return [TeacherAssignmentRead.model_validate(row) for row in rows]


@router.post(
    "/courses/{course_id}/teachers",
    response_model=TeacherAssignmentCreated,
    status_code=status.HTTP_201_CREATED,
)
async def assign_teacher(
    course_id: UUID,
    payload: AssignTeacherRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> TeacherAssignmentCreated:
    """Create (or no-op return) a ``role=teacher, scope=course`` assignment.

    The course-scoped permission dep ensures HOD scope auto-matches the
    course's org_unit; an HOD on Dept-X cannot assign teachers to a
    course in Dept-Y (plan §4467).

    When the course is already published, the teacher is notified with a
    deep-link to the course (see ``assignment_service.assign_teacher_to_course``).

    The assignee must be a member of the course's organization — enforced
    server-side, so the org restriction does not depend on the client only
    offering in-org users to pick from. 403 otherwise.
    """
    try:
        result = await assignment_service.assign_teacher_to_course(
            db,
            course_id,
            payload.user_id,
            current_user,
            is_instructor=payload.is_instructor,
            is_assistant=payload.is_assistant,
            arq_pool=arq_pool,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": str(exc)},
        ) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return TeacherAssignmentCreated.model_validate(result)


@router.get("/assignable-teachers", response_model=list[AssignableTeacher])
async def list_assignable_teachers_for_new_course(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
    faculty_id: UUID | None = None,
) -> list[AssignableTeacher]:
    """Teachers for a course that does not exist yet — the create wizard's picker.

    Same list as the per-course endpoint, but the organization comes from the
    CALLER's token instead of the course, because the wizard staffs the course
    in the same form that creates it. That is the same org ``create_course``
    stamps on the new row, so the picker cannot offer someone the follow-up
    assignment would reject.

    Guarded by the GLOBAL staffing dependency, not the per-course one: there is
    no ``course_id`` path param to scope against, and asking for one would make
    the policy layer 500 with ``policy_misconfigured``.

    Declared BEFORE ``/courses/{course_id}/...`` deliberately — a literal path
    segment must not be shadowed by a parameterised route.
    """
    try:
        rows = await assignment_service.list_assignable_teachers_for_creator(
            db, current_user, faculty_id=faculty_id
        )
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": str(exc)},
        ) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "message": str(exc)},
        ) from exc
    return [AssignableTeacher.model_validate(row) for row in rows]


@router.get(
    "/courses/{course_id}/assignable-teachers",
    response_model=list[AssignableTeacher],
)
async def list_assignable_teachers(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[AssignableTeacher]:
    """Teachers this course can be assigned to — same organization, teacher role.

    Backs the manager's teacher picker so assignment stops being "paste a
    user UUID". The organization comes from the COURSE, not from a query
    parameter: the restriction has to hold server-side, or it is only a UI
    convention. POST /teachers re-checks membership for the same reason.

    ``already_assigned`` flags current teachers so the picker can render them
    as chosen instead of offering a no-op assignment.
    """
    try:
        rows = await assignment_service.list_assignable_teachers(db, course_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return [AssignableTeacher.model_validate(row) for row in rows]


@router.get("/courses/{course_id}/readiness", response_model=CourseReadiness)
async def get_course_readiness(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseReadiness:
    """Is this course actually deliverable? Asked before publish, not after.

    Three checks: an assigned teacher, at least one gradeable unit, and the
    course's own status — plus learning outcomes, which are also a publish
    gate. Career-path placements ride along as informational data (the
    course detail's Career Paths tab). `can_publish` mirrors the publish
    gate's condition exactly, so the checklist cannot promise a publish the
    gate then refuses with a 409.
    """
    try:
        data = await assignment_service.get_course_readiness(db, course_id)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    return CourseReadiness.model_validate(data)


@router.delete(
    "/courses/{course_id}/teachers/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_teacher(
    course_id: UUID,
    user_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-revoke -- sets ``active_until = NOW()``, preserves audit trail."""
    try:
        await assignment_service.remove_teacher_from_course(db, course_id, user_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()


@router.put(
    "/courses/{course_id}/teachers/{user_id}/role",
    response_model=TeacherAssignmentRead,
)
async def set_teacher_titles(
    course_id: UUID,
    user_id: UUID,
    payload: CourseTeacherRoleRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeacherAssignmentRead:
    """Set a teacher's course-scoped title flags (Course Instructor / TA).

    Both flags may be true — one teacher holding both titles is legal (user
    decision 2026-08-30). Rejected (409): clearing both flags, and turning
    off the LAST Course Instructor while the course still has teachers.
    Available to the manager staffing surface only.
    """
    try:
        await assignment_service.set_teacher_titles(
            db,
            course_id=course_id,
            user_id=user_id,
            is_instructor=payload.is_instructor,
            is_assistant=payload.is_assistant,
            actor=current_user,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "message": str(exc)},
        ) from exc
    await db.commit()
    rows = await assignment_service.list_teachers_with_emails(db, course_id)
    row = next((r for r in rows if r["user_id"] == user_id), None)
    if row is None:
        raise _not_found("teacher assignment not found after role update")
    return TeacherAssignmentRead.model_validate(row)


@router.get("/courses/{course_id}/roster", response_model=list[RosterEntry])
async def get_course_roster(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RosterEntry]:
    """Enrolled students for HOD/Manager oversight (read-only)."""
    rows = await assignment_service.list_course_roster(db, course_id)
    return [RosterEntry.model_validate(row) for row in rows]


@router.delete(
    "/courses/{course_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_dept_course(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Manager-facing soft-delete of a course (reversible tombstone).

    Manager-owned (``course.delete``): a manager can delete a course in
    their org; HOD (``course.assign_teacher`` only) and teachers — even
    the course owner — get 403. Cascades to the course's children via
    ``soft_delete_cascade`` (same semantics as the admin delete).
    """
    try:
        await delete_course_service(db, course_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@router.post(
    "/courses/{course_id}/clone",
    response_model=CourseAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def clone_dept_course(
    course_id: UUID,
    payload: CourseCloneRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)] = None,
) -> CourseAuthoring:
    """Manager-only course clone with selectable depth (user request).

    Cloning creates a new course from an existing one — a lifecycle/identity
    operation, so it is gated on ``course.delete`` (the same manager-only
    gate as delete/archive) with the ownership short-circuit disabled: a
    teacher who owns the course cannot clone it through the dept surface.

    Depth (REQUIRED, no default):

    * ``shell``     — course + learning outcomes only.
    * ``structure`` — + module skeleton (modules + module prerequisites).
    * ``full``      — complete deep clone: modules + items + lessons +
      quizzes + interviews + resources, every cross-reference re-wired to
      the copy. All content lands as drafts; runtime data is never copied.

    The clone gets a fresh org-unique slug (``{slug}-copy``), a
    ``" (Copy)"`` title suffix, and is owned by the requesting manager.
    """
    try:
        course = await clone_course_service(
            db,
            source_course_id=course_id,
            depth=payload.depth,
            actor=current_user,
            arq_pool=arq_pool,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "message": str(exc)},
        ) from exc
    await db.commit()
    return course


@router.patch("/courses/{course_id}", response_model=CourseAuthoring)
async def update_dept_course(
    course_id: UUID,
    payload: CourseUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_DELETE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    """Manager-facing course update (title/slug/description/…).

    Gated on ``course.delete`` (manager-owned, same gate as the delete
    route) so identity edits — title, slug — live on the dept surface and
    are manager-only, while the teacher surface keeps content authoring.
    """
    try:
        course = await update_course_service(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(str(exc)) from exc
    await db.commit()
    return course


@router.get("/faculties/{faculty_id}/courses", response_model=list[CourseAuthoring])
async def list_faculty_courses(
    faculty_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_FACULTY_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CourseAuthoring]:
    """All courses owned by ``faculty_id``.

    HOD can only pass their own org_unit (the org-unit-scoped permission
    factory walks the unit's ancestor chain and rejects cross-dept
    requests with 403). Manager (``scope_kind=organization``) and Admin
    (``scope_kind=global``) pass for any org_unit.
    """
    return await assignment_service.list_courses_in_faculty(db, faculty_id)


__all__ = ["router"]
