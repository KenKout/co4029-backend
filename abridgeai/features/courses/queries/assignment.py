"""HOD/Manager-side teacher-assignment queries.

Cross-feature joins between ``user_role_assignments`` (owned by
``features.access_control``) and ``user_profiles`` / ``users`` (owned by
``features.identity``) needed by the courses-feature assignment service.

Per AGENTS.md, cross-feature ORM imports are allowed in ``queries/``
when the JOIN is unavoidable.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.models import (
    CareerPath,
    OrganizationMembership,
    Role,
    UserFacultyAssignment,
    UserRoleAssignment,
)
from abridgeai.features.career_paths.models import (
    CareerPathCourse,
    CareerPathStage,
    CareerPathVersion,
)
from abridgeai.features.courses.models import Course
from abridgeai.features.identity.models import StorageObject, User, UserProfile


async def list_career_paths_containing_course(
    db: AsyncSession, course_id: UUID
) -> list[dict[str, Any]]:
    """Career paths this course sits on, with the stage it occupies.

    Backs the readiness checklist's "on a career path" row. A course nobody put
    on a path is invisible to students — the paths are how they reach it — so a
    manager finishing a course needs to know that before calling it done.

    ``is_required`` matters for the warning's severity: a REQUIRED course with
    no gradeable unit locks its stage and everything behind it, whereas an
    optional one merely cannot be completed.
    """
    stmt = (
        select(
            CareerPath.id.label("career_path_id"),
            CareerPath.name.label("career_path_name"),
            CareerPath.status.label("career_path_status"),
            CareerPathStage.id.label("stage_id"),
            CareerPathStage.title.label("stage_title"),
            CareerPathStage.position.label("stage_position"),
            CareerPathCourse.is_required,
        )
        # Gap 3 (0074): items hang off a VERSION; join through it to the path.
        .join(CareerPathVersion, CareerPathVersion.id == CareerPathCourse.version_id)
        .join(CareerPath, CareerPath.id == CareerPathVersion.career_path_id)
        .join(CareerPathStage, CareerPathStage.id == CareerPathCourse.stage_id)
        .where(
            CareerPathCourse.course_id == course_id,
            CareerPath.deleted_at.is_(None),
            CareerPathVersion.deleted_at.is_(None),
            CareerPathStage.deleted_at.is_(None),
        )
        .order_by(CareerPath.name, CareerPathStage.position)
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(row) for row in rows]


async def list_assignable_teachers(
    db: AsyncSession,
    *,
    organization_id: UUID,
    course_id: UUID | None = None,
    faculty_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """Users holding the ``teacher`` role who are members of ``organization_id``.

    Backs the manager's teacher picker. Two filters matter and both are
    server-side:

    * **role** — the user must hold ``teacher`` at some scope. Filtering on
      the role alone is not enough, which is why the org filter is a JOIN
      here rather than a query parameter the client passes: a teacher of
      another organization also "holds the teacher role".
    * **organization membership** — an active, non-deleted membership row.

    ``already_assigned`` marks users who already teach this course so the
    picker can show them as chosen instead of offering a no-op. ``course_id``
    is optional because the create-course wizard picks teachers BEFORE the
    course exists; with no course there is nothing to be already assigned to,
    so the flag is uniformly false.
    """
    assigned_subq = (
        select(UserRoleAssignment.user_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.course_id == course_id,
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
        .scalar_subquery()
    )
    already_assigned = (
        User.id.in_(assigned_subq) if course_id is not None else literal(False)  # noqa: FBT003
    )
    stmt = (
        select(
            User.id.label("user_id"),
            User.primary_email,
            UserProfile.display_name,
            already_assigned.label("already_assigned"),
        )
        .join(OrganizationMembership, OrganizationMembership.user_id == User.id)
        .join(UserRoleAssignment, UserRoleAssignment.user_id == User.id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(
            OrganizationMembership.organization_id == organization_id,
            OrganizationMembership.status == "active",
            OrganizationMembership.deleted_at.is_(None),
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
            # `users` has no deleted_at (0002 skip-list); `status` is the
            # lifecycle column, so an inactive/suspended account is excluded
            # here rather than by a soft-delete filter.
            User.status == "active",
        )
        .distinct()
        .order_by(UserProfile.display_name, User.primary_email)
    )
    if faculty_id is not None:
        stmt = stmt.join(
            UserFacultyAssignment,
            (UserFacultyAssignment.user_id == User.id)
            & (UserFacultyAssignment.organization_id == organization_id),
        ).where(
            UserFacultyAssignment.faculty_id == faculty_id,
            UserFacultyAssignment.status == "active",
            UserFacultyAssignment.deleted_at.is_(None),
            (UserFacultyAssignment.active_until.is_(None))
            | (UserFacultyAssignment.active_until > func.now()),
        )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(row) for row in rows]


async def list_teachers_for_course(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Active teachers for a course with profile info.

    Projects the avatar object's ``bucket``/``object_key`` (never a URL) —
    the service layer mints the short-TTL presigned ``avatar_url`` from them,
    the same split used by ``authoring.list_instructors_for_courses`` and the
    roster query. Without it the staffing tables drew the same teacher as
    initials while the worklist one click away drew their photo.
    """
    stmt = (
        select(
            User.id.label("user_id"),
            User.primary_email,
            UserProfile.display_name,
            StorageObject.bucket.label("avatar_bucket"),
            StorageObject.object_key.label("avatar_object_key"),
            UserRoleAssignment.id.label("assignment_id"),
            UserRoleAssignment.is_instructor,
            UserRoleAssignment.is_assistant,
            UserRoleAssignment.active_from,
            UserRoleAssignment.active_until,
        )
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .join(User, User.id == UserRoleAssignment.user_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .outerjoin(StorageObject, StorageObject.id == UserProfile.avatar_object_id)
        .where(
            UserRoleAssignment.course_id == course_id,
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
        .order_by(UserRoleAssignment.active_from)
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(row) for row in rows]


async def count_active_course_teachers(db: AsyncSession, course_id: UUID) -> int:
    """Number of active course-scoped teacher rows on ``course_id``."""
    stmt = (
        select(func.count())
        .select_from(UserRoleAssignment)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.course_id == course_id,
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
    )
    return int((await db.execute(stmt)).scalar_one())


_ACTIVE_TEACHER_WHERE: tuple = (
    UserRoleAssignment.scope_kind == "course",
    Role.code == "teacher",
    UserRoleAssignment.deleted_at.is_(None),
    (UserRoleAssignment.active_until.is_(None)) | (UserRoleAssignment.active_until > func.now()),
)


async def get_active_teacher_assignment_row(
    db: AsyncSession, *, course_id: UUID, user_id: UUID
) -> UserRoleAssignment | None:
    """The active course-scoped teacher row for ``(course_id, user_id)``."""
    stmt = (
        select(UserRoleAssignment)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.course_id == course_id,
            UserRoleAssignment.user_id == user_id,
            *_ACTIVE_TEACHER_WHERE,
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def count_course_instructors(db: AsyncSession, course_id: UUID) -> int:
    """Number of active Course-Instructor flags on ``course_id`` (0..N).

    Multiple instructors are legal (user decision 2026-08-30), so this is a
    COUNT, not an existence probe — callers that used to ask "who is the one
    instructor" now ask "how many are there".
    """
    stmt = (
        select(func.count())
        .select_from(UserRoleAssignment)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.course_id == course_id,
            UserRoleAssignment.is_instructor.is_(True),
            *_ACTIVE_TEACHER_WHERE,
        )
    )
    return int((await db.execute(stmt)).scalar_one())


async def find_active_teacher_assignment(
    db: AsyncSession, *, course_id: UUID, user_id: UUID
) -> UUID | None:
    """Return id of an active ``role=teacher`` assignment for ``user_id`` on
    ``course_id`` (None if none).
    """
    stmt = (
        select(UserRoleAssignment.id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.course_id == course_id,
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
        .limit(1)
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    return row if row is None else UUID(str(row))


async def revoke_teacher_assignment(db: AsyncSession, assignment_id: UUID) -> None:
    """Soft-revoke by setting ``active_until = NOW()`` (legacy behaviour)."""
    stmt = select(UserRoleAssignment).where(UserRoleAssignment.id == assignment_id)
    assignment = (await db.execute(stmt)).scalar_one_or_none()
    if assignment is not None:
        assignment.active_until = func.now()
        await db.flush()


async def insert_teacher_assignment(
    db: AsyncSession,
    *,
    assignment_id: UUID,
    user_id: UUID,
    role_id: UUID,
    organization_id: UUID,
    course_id: UUID,
    granted_by: UUID,
    is_instructor: bool = False,
    is_assistant: bool = True,
) -> None:
    """INSERT a ``role=teacher, scope=course`` row into ``user_role_assignments``."""
    assignment = UserRoleAssignment(
        id=assignment_id,
        user_id=user_id,
        role_id=role_id,
        scope_kind="course",
        organization_id=organization_id,
        course_id=course_id,
        granted_by=granted_by,
        is_instructor=is_instructor,
        is_assistant=is_assistant,
    )
    db.add(assignment)
    await db.flush()


async def get_teacher_role_id(db: AsyncSession) -> UUID:
    """Resolve the seeded ``role_code='teacher'`` UUID via T1.12 catalog."""
    stmt = select(Role.id).where(Role.code == "teacher", Role.deleted_at.is_(None))
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise RuntimeError(
            "teacher role not seeded; expected migration 0004_seed_permission_catalog"
        )
    return UUID(str(row))


async def list_courses_by_organization(
    db: AsyncSession, organization_id: UUID | None = None
) -> list[Course]:
    """List courses optionally filtered by organization, newest first."""
    stmt = select(Course).order_by(Course.created_at.desc())
    if organization_id is not None:
        stmt = stmt.where(Course.organization_id == organization_id)
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "count_active_course_teachers",
    "count_course_instructors",
    "find_active_teacher_assignment",
    "get_active_teacher_assignment_row",
    "get_teacher_role_id",
    "insert_teacher_assignment",
    "list_courses_by_organization",
    "list_teachers_for_course",
    "revoke_teacher_assignment",
]
