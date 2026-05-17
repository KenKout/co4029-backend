"""HOD/Manager-side teacher-assignment queries.

Cross-feature joins between ``user_role_assignments`` (owned by
``features.access_control``) and ``user_profiles`` / ``users`` (owned by
``features.identity``) needed by the courses-feature assignment service.
The joins are encoded as raw SQL so this module does not need ORM
imports of the foreign features (the import-linter
"Features are independent" contract stays green for queries — only the
courses-feature service consumes these helpers).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_LIST_TEACHERS_FOR_COURSE_SQL = text(
    """
    SELECT
        u.id           AS user_id,
        u.primary_email AS primary_email,
        up.display_name AS display_name,
        ura.id          AS assignment_id,
        ura.active_from AS active_from,
        ura.active_until AS active_until
    FROM user_role_assignments ura
    JOIN roles r          ON r.id = ura.role_id
    JOIN users u          ON u.id = ura.user_id
    LEFT JOIN user_profiles up ON up.user_id = u.id
    WHERE ura.course_id = :course_id
      AND ura.scope_kind = 'course'
      AND r.code = 'teacher'
      AND ura.deleted_at IS NULL
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
    ORDER BY ura.active_from
    """
)


async def list_teachers_for_course(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_LIST_TEACHERS_FOR_COURSE_SQL, {"course_id": course_id})).mappings()
    return [dict(row) for row in rows]


_FIND_TEACHER_ASSIGNMENT_SQL = text(
    """
    SELECT ura.id
    FROM user_role_assignments ura
    JOIN roles r ON r.id = ura.role_id
    WHERE ura.course_id = :course_id
      AND ura.user_id = :user_id
      AND ura.scope_kind = 'course'
      AND r.code = 'teacher'
      AND ura.deleted_at IS NULL
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
    LIMIT 1
    """
)


async def find_active_teacher_assignment(
    db: AsyncSession, *, course_id: UUID, user_id: UUID
) -> UUID | None:
    """Return id of an active ``role=teacher`` assignment for ``user_id`` on
    ``course_id`` (None if none).
    """
    result = await db.execute(
        _FIND_TEACHER_ASSIGNMENT_SQL,
        {"course_id": course_id, "user_id": user_id},
    )
    row = result.scalar_one_or_none()
    return row if row is None else UUID(str(row))


_REVOKE_TEACHER_SQL = text(
    """
    UPDATE user_role_assignments
       SET active_until = NOW()
     WHERE id = :assignment_id
    """
)


async def revoke_teacher_assignment(db: AsyncSession, assignment_id: UUID) -> None:
    """Soft-revoke by setting ``active_until = NOW()`` (legacy behaviour)."""
    await db.execute(_REVOKE_TEACHER_SQL, {"assignment_id": assignment_id})
    await db.flush()


_INSERT_TEACHER_ASSIGNMENT_SQL = text(
    """
    INSERT INTO user_role_assignments
        (id, user_id, role_id, scope_kind,
         organization_id, course_id, granted_by)
    VALUES (:id, :user_id, :role_id, 'course',
            :organization_id, :course_id, :granted_by)
    """
)


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
    await db.execute(
        _INSERT_TEACHER_ASSIGNMENT_SQL,
        {
            "id": assignment_id,
            "user_id": user_id,
            "role_id": role_id,
            "organization_id": organization_id,
            "course_id": course_id,
            "granted_by": granted_by,
        },
    )
    await db.flush()


_GET_TEACHER_ROLE_ID_SQL = text(
    "SELECT id FROM roles WHERE code = 'teacher' AND deleted_at IS NULL"
)


async def get_teacher_role_id(db: AsyncSession) -> UUID:
    """Resolve the seeded ``role_code='teacher'`` UUID via T1.12 catalog."""
    result = await db.execute(_GET_TEACHER_ROLE_ID_SQL)
    row = result.scalar_one_or_none()
    if row is None:
        raise RuntimeError(
            "teacher role not seeded; expected migration 0004_seed_permission_catalog"
        )
    return UUID(str(row))


__all__ = [
    "find_active_teacher_assignment",
    "get_teacher_role_id",
    "insert_teacher_assignment",
    "list_teachers_for_course",
    "revoke_teacher_assignment",
]
