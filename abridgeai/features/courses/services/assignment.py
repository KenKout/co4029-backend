"""HOD/Manager-side teacher assignment service for courses (plan §4209).

Bridges the courses feature to the access-control feature for the narrow
case of "assign teacher X to course Y". Per Reconciliation §A1 + the
import-linter contract, the cross-feature reach is read-and-write but
intentionally narrow: this module composes raw-SQL helpers from
:mod:`features.courses.queries.assignment` + the seeded ``role_code='teacher'``
catalog row from T1.12.

Soft-revoke semantics — :func:`remove_teacher_from_course` sets
``active_until = NOW()`` rather than DELETE-ing the assignment row; the
legacy ``backend/app/routes/courses/`` flow had no remove endpoint, so
the locked decision per plan §4211 is "match revoke pattern from
T1.10 admin", which is the soft-revoke.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.queries import (
    assignment as assignment_queries,
)
from abridgeai.features.courses.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.courses.schemas import CourseAuthoring, InstructorRead

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def assign_teacher_to_course(
    db: AsyncSession,
    course_id: UUID,
    user_id: UUID,
    actor: CurrentUser,
) -> dict[str, Any]:
    """Create (or no-op return) a ``role=teacher, scope=course`` assignment.

    If an active assignment already exists for ``(course_id, user_id)`` the
    existing row is returned unchanged. Otherwise a new row is INSERT-ed
    with ``role_id`` resolved from the seeded T1.12 catalog,
    ``scope_kind='course'``, and ``granted_by=actor.user_id``.
    """
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")

    existing = await assignment_queries.find_active_teacher_assignment(
        db, course_id=course_id, user_id=user_id
    )
    if existing is not None:
        return {
            "id": existing,
            "course_id": course_id,
            "user_id": user_id,
            "role_code": "teacher",
            "scope_kind": "course",
            "organization_id": course.organization_id,
            "granted_by": actor.user_id,
        }

    role_id = await assignment_queries.get_teacher_role_id(db)
    new_id = uuid4()
    await assignment_queries.insert_teacher_assignment(
        db,
        assignment_id=new_id,
        user_id=user_id,
        role_id=role_id,
        organization_id=course.organization_id,
        course_id=course_id,
        granted_by=actor.user_id,
    )
    return {
        "id": new_id,
        "course_id": course_id,
        "user_id": user_id,
        "role_code": "teacher",
        "scope_kind": "course",
        "organization_id": course.organization_id,
        "granted_by": actor.user_id,
    }


async def remove_teacher_from_course(
    db: AsyncSession,
    course_id: UUID,
    user_id: UUID,
    actor: CurrentUser,
) -> None:
    """Soft-revoke the active teacher assignment for ``(course_id, user_id)``.

    Sets ``active_until = NOW()`` on the assignment row rather than
    deleting it, preserving the audit trail (legacy parity with the
    T1.10 admin revoke flow). 404 when no active assignment exists.
    """
    del actor
    assignment_id = await assignment_queries.find_active_teacher_assignment(
        db, course_id=course_id, user_id=user_id
    )
    if assignment_id is None:
        raise NotFoundError(f"No active teacher assignment for course={course_id} user={user_id}")
    await assignment_queries.revoke_teacher_assignment(db, assignment_id)


async def list_teachers_for_course(db: AsyncSession, course_id: UUID) -> list[InstructorRead]:
    """Return ``InstructorRead`` rows for the active teachers of ``course_id``."""
    rows = await assignment_queries.list_teachers_for_course(db, course_id)
    return [
        InstructorRead.model_validate(
            {
                "user_id": row["user_id"],
                "display_name": row["display_name"] or row["primary_email"],
                "avatar_url": None,
                "headline": None,
            }
        )
        for row in rows
    ]


async def list_courses_in_dept(db: AsyncSession, org_unit_id: UUID) -> list[CourseAuthoring]:
    """HOD overview — all courses scoped to ``org_unit_id``."""
    courses = await authoring_queries.list_courses_in_org_unit(db, org_unit_id)
    return [CourseAuthoring.model_validate(course) for course in courses]


__all__ = [
    "assign_teacher_to_course",
    "list_courses_in_dept",
    "list_teachers_for_course",
    "remove_teacher_from_course",
]
