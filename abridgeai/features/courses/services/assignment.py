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

from abridgeai.core.exceptions import ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.queries import (
    assignment as assignment_queries,
)
from abridgeai.features.courses.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.courses.schemas import CourseAuthoring, InstructorRead
from abridgeai.features.courses.services import notify

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def assign_teacher_to_course(
    db: AsyncSession,
    course_id: UUID,
    user_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> dict[str, Any]:
    """Create (or no-op return) a ``role=teacher, scope=course`` assignment.

    If an active assignment already exists for ``(course_id, user_id)`` the
    existing row is returned unchanged. Otherwise a new row is INSERT-ed
    with ``role_id`` resolved from the seeded T1.12 catalog,
    ``scope_kind='course'``, and ``granted_by=actor.user_id``.

    When the course is already **published**, the newly-assigned teacher is
    notified (in-app + email) with a deep-link to the course. A no-op re-assign
    does not re-notify. Draft courses do not notify here — the teacher is told
    when the course publishes (see :func:`publish_course`).
    """
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")

    # The assignee MUST belong to the course's organization. Enforced here,
    # server-side, rather than by whatever list the UI happened to render: the
    # request carries a bare user_id, so a client could otherwise name any user
    # in the system and grant them course.update on another org's course. The
    # course-scoped permission dep upstream checks the ACTOR's reach, not the
    # assignee's membership — different question.
    if not await _user_is_in_org(db, user_id=user_id, org_id=course.organization_id):
        raise ForbiddenError(
            f"teacher_not_in_course_org: user {user_id} is not a member of the "
            f"organization that owns course {course_id}"
        )

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

    # Notify on assignment for draft AND published alike. The manager flow is
    # create (draft) -> assign teacher -> teacher edits content -> publish, so
    # at assignment time the course is ALWAYS a draft: gating this on
    # `status == "published"` meant the notification never fired in the real
    # flow and the teacher was handed work nobody told them about.
    #
    # The premise the old guard rested on — "a teacher can't act on a draft
    # they can't yet see" — is false. `list_courses_assigned_to_teacher`
    # applies only `_archived_filter` and has no status filter, so assigned
    # teachers do see drafts; that is what makes the "teacher edits content"
    # step work at all.
    #
    # Archived is the one status with nothing left to act on.
    # Never let a notification failure roll back the assignment.
    if course.status != "archived":
        await notify.notify_teacher_assigned(
            db,
            teacher_user_id=user_id,
            course_id=course_id,
            course_title=course.title,
            arq_pool=arq_pool,
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


async def list_assignable_teachers(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Teachers a manager may assign to ``course_id``: same org, teacher role.

    The organization is derived from the COURSE, never from a client
    parameter — "belongs to that org" has to be a server-side fact, otherwise
    it is only a UI convention that a crafted request walks straight past.
    """
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")
    return await assignment_queries.list_assignable_teachers(
        db, organization_id=course.organization_id, course_id=course_id
    )


async def _user_is_in_org(db: AsyncSession, *, user_id: UUID, org_id: UUID) -> bool:
    """Whether ``user_id`` has an active membership in ``org_id``.

    Lazy import keeps the courses -> access_control edge out of module import
    time, matching the pattern in ``courses.services.catalog``.
    """
    from abridgeai.features.access_control.api import public as access_api  # noqa: PLC0415

    return await access_api.is_user_member_of_org(db, user_id=user_id, org_id=org_id)


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


async def list_courses_for_organization(
    db: AsyncSession, organization_id: UUID | None = None
) -> list[CourseAuthoring]:
    """Manager/Admin overview — courses optionally filtered by organization."""
    courses = await assignment_queries.list_courses_by_organization(db, organization_id)
    return [CourseAuthoring.model_validate(course) for course in courses]


async def list_teachers_with_emails(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Active teachers for a course with email + display_name."""
    return await assignment_queries.list_teachers_for_course(db, course_id)


async def list_course_roster(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Enrolled students for a course (HOD/Manager view)."""
    return await authoring_queries.list_course_roster(db, course_id)


__all__ = [
    "assign_teacher_to_course",
    "list_course_roster",
    "list_courses_for_organization",
    "list_courses_in_dept",
    "list_teachers_for_course",
    "list_teachers_with_emails",
    "remove_teacher_from_course",
]
