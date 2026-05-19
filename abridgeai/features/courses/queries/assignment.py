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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.models import Role, UserRoleAssignment
from abridgeai.features.courses.models import Course
from abridgeai.features.identity.models import User, UserProfile


async def list_teachers_for_course(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Active teachers for a course with profile info."""
    stmt = (
        select(
            User.id.label("user_id"),
            User.primary_email,
            UserProfile.display_name,
            UserRoleAssignment.id.label("assignment_id"),
            UserRoleAssignment.active_from,
            UserRoleAssignment.active_until,
        )
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .join(User, User.id == UserRoleAssignment.user_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
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
    "find_active_teacher_assignment",
    "get_teacher_role_id",
    "insert_teacher_assignment",
    "list_courses_by_organization",
    "list_teachers_for_course",
    "revoke_teacher_assignment",
]
