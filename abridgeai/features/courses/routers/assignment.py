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
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
)
from abridgeai.features.courses.schemas import (
    AssignTeacherRequest,
    CourseAuthoring,
    RosterEntry,
    TeacherAssignmentCreated,
    TeacherAssignmentRead,
)
from abridgeai.features.courses.services import assignment as assignment_service

router = APIRouter(prefix="/dept", tags=["courses-assignment"])

_REQUIRE_STAFFING = require_any_permission(
    "course.assign_teacher", "user.role_assign", "system.administer"
)
_REQUIRE_COURSE_STAFFING = require_course_permission(
    "course_id", "course.assign_teacher", "user.role_assign", "system.administer"
)
_REQUIRE_ORG_UNIT_STAFFING = require_org_unit_permission(
    "org_unit_id", "course.assign_teacher", "user.role_assign", "system.administer"
)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
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
        return await assignment_service.list_courses_in_dept(db, UUID(str(org_unit_id)))

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
) -> TeacherAssignmentCreated:
    """Create (or no-op return) a ``role=teacher, scope=course`` assignment.

    The course-scoped permission dep ensures HOD scope auto-matches the
    course's org_unit; an HOD on Dept-X cannot assign teachers to a
    course in Dept-Y (plan §4467).
    """
    try:
        result = await assignment_service.assign_teacher_to_course(
            db, course_id, payload.user_id, current_user
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return TeacherAssignmentCreated.model_validate(result)


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
    await db.commit()


@router.get("/courses/{course_id}/roster", response_model=list[RosterEntry])
async def get_course_roster(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RosterEntry]:
    """Enrolled students for HOD/Manager oversight (read-only)."""
    rows = await assignment_service.list_course_roster(db, course_id)
    return [RosterEntry.model_validate(row) for row in rows]


@router.get("/org-units/{org_unit_id}/courses", response_model=list[CourseAuthoring])
async def list_org_unit_courses(
    org_unit_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_ORG_UNIT_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CourseAuthoring]:
    """All courses in ``org_unit_id``.

    HOD can only pass their own org_unit (the org-unit-scoped permission
    factory walks the unit's ancestor chain and rejects cross-dept
    requests with 403). Manager (``scope_kind=organization``) and Admin
    (``scope_kind=global``) pass for any org_unit.
    """
    return await assignment_service.list_courses_in_dept(db, org_unit_id)


__all__ = ["router"]
