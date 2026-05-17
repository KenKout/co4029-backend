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

Roster endpoint deferred-ORM strategy (plan §4461): the enrollments
aggregate is owned by Phase 7. T3.8 reads ``course_enrollments`` via
raw ``sa.text()``. When Phase 7 lands, replace the inline SQL with the
canonical enrollments query helper.

FIX-CRIT-4 + FIX-SEC-1 perimeter: every endpoint uses a ``require_*``
dependency -- never bare :func:`get_current_user`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
)
from abridgeai.features.courses.models import Course
from abridgeai.features.courses.schemas import CourseAuthoring
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


class TeacherAssignmentRead(BaseModel):
    """Authoring DTO for a teacher-on-course assignment.

    ``primary_email`` / ``display_name`` are joined from ``users`` +
    ``user_profiles``; ``active_until`` is non-null for soft-revoked
    rows (audit trail).
    """

    user_id: UUID
    display_name: str
    primary_email: str
    assignment_id: UUID | None = None
    active_from: datetime | None = None
    active_until: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AssignTeacherRequest(BaseModel):
    user_id: UUID

    model_config = ConfigDict(extra="forbid")


class TeacherAssignmentCreated(BaseModel):
    """Response payload for POST -- the new (or pre-existing) assignment row."""

    id: UUID
    course_id: UUID
    user_id: UUID
    role_code: str
    scope_kind: str
    organization_id: UUID
    granted_by: UUID

    model_config = ConfigDict(from_attributes=True)


class RosterEntry(BaseModel):
    """Single row in the ``GET /dept/courses/{id}/roster`` response.

    Phase 7 will replace this with the canonical ``EnrollmentRead``
    schema once the enrollments aggregate lands; the raw-SQL read here
    is the documented bridge per plan §4461.
    """

    enrollment_id: UUID
    student_id: UUID
    display_name: str | None
    primary_email: str
    status: str
    enrolled_at: datetime
    completed_at: datetime | None = None
    dropped_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


_RESOLVE_CALLER_SCOPE_SQL = text(
    """
    SELECT ura.scope_kind,
           ura.organization_id,
           ura.org_unit_id
    FROM user_role_assignments ura
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.active_from <= NOW()
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
    ORDER BY
        CASE ura.scope_kind
            WHEN 'course' THEN 1
            WHEN 'org_unit' THEN 2
            WHEN 'organization' THEN 3
            WHEN 'global' THEN 4
            ELSE 5
        END
    """
)


async def _resolve_caller_scope(
    db: AsyncSession, user_id: UUID
) -> tuple[str, UUID | None, UUID | None]:
    result = await db.execute(_RESOLVE_CALLER_SCOPE_SQL, {"user_id": user_id})
    row = result.first()
    if row is None:
        return ("global", None, None)
    return (str(row.scope_kind), row.organization_id, row.org_unit_id)


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

    stmt = select(Course).order_by(Course.created_at.desc())
    if scope_kind == "organization" and organization_id is not None:
        stmt = stmt.where(Course.organization_id == UUID(str(organization_id)))

    courses = list((await db.execute(stmt)).scalars().all())
    return [CourseAuthoring.model_validate(course) for course in courses]


_LIST_TEACHERS_WITH_EMAILS_SQL = text(
    """
    SELECT
        u.id            AS user_id,
        u.primary_email AS primary_email,
        COALESCE(up.display_name, u.primary_email) AS display_name,
        ura.id          AS assignment_id,
        ura.active_from AS active_from,
        ura.active_until AS active_until
    FROM user_role_assignments ura
    JOIN roles r ON r.id = ura.role_id
    JOIN users u ON u.id = ura.user_id
    LEFT JOIN user_profiles up ON up.user_id = u.id
    WHERE ura.course_id = :course_id
      AND ura.scope_kind = 'course'
      AND r.code = 'teacher'
      AND ura.deleted_at IS NULL
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
    ORDER BY ura.active_from
    """
)


async def _list_teachers_with_emails(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_LIST_TEACHERS_WITH_EMAILS_SQL, {"course_id": course_id})).mappings()
    return [dict(row) for row in rows]


@router.get("/courses/{course_id}/teachers", response_model=list[TeacherAssignmentRead])
async def list_course_teachers(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TeacherAssignmentRead]:
    """Active teachers (``active_until`` IS NULL or future) for a course."""
    rows = await _list_teachers_with_emails(db, course_id)
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


_LIST_ROSTER_SQL = text(
    """
    SELECT
        ce.id           AS enrollment_id,
        ce.student_id   AS student_id,
        u.primary_email AS primary_email,
        up.display_name AS display_name,
        ce.status       AS status,
        ce.enrolled_at  AS enrolled_at,
        ce.completed_at AS completed_at,
        ce.dropped_at   AS dropped_at
    FROM course_enrollments ce
    JOIN users u ON u.id = ce.student_id
    LEFT JOIN user_profiles up ON up.user_id = u.id
    WHERE ce.course_id = :course_id
    ORDER BY ce.enrolled_at DESC
    """
)


@router.get("/courses/{course_id}/roster", response_model=list[RosterEntry])
async def get_course_roster(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_STAFFING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RosterEntry]:
    """Enrolled students for HOD/Manager oversight (read-only).

    The enrollments aggregate is owned by Phase 7; this endpoint reads
    ``course_enrollments`` via raw SQL until the canonical
    :class:`EnrollmentRead` schema lands. Plan §4461 documents the
    deferred-ORM strategy.
    """
    rows = (await db.execute(_LIST_ROSTER_SQL, {"course_id": course_id})).mappings()
    return [RosterEntry.model_validate(dict(row)) for row in rows]


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
