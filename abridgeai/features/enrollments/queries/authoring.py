from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import and_, select, text

from abridgeai.features.enrollments.models import Enrollment, InvitationCode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_enrollments_for_course(
    db: AsyncSession, course_id: UUID
) -> list[Enrollment]:
    result = await db.execute(
        select(Enrollment)
        .where(Enrollment.course_id == course_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return list(result.scalars().all())


async def list_enrollments_for_user(db: AsyncSession, user_id: UUID) -> list[Enrollment]:
    result = await db.execute(
        select(Enrollment)
        .where(Enrollment.student_id == user_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return list(result.scalars().all())


async def find_enrollment(
    db: AsyncSession, course_id: UUID, student_id: UUID
) -> Enrollment | None:
    result = await db.execute(
        select(Enrollment).where(
            and_(
                Enrollment.course_id == course_id,
                Enrollment.student_id == student_id,
            )
        )
    )
    return result.scalar_one_or_none()


async def find_invitation_code_by_string(
    db: AsyncSession, code: str
) -> InvitationCode | None:
    result = await db.execute(
        select(InvitationCode).where(
            and_(
                InvitationCode.code == code,
                InvitationCode.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_invitation_code(db: AsyncSession, code_id: UUID) -> InvitationCode | None:
    result = await db.execute(
        select(InvitationCode).where(
            and_(
                InvitationCode.id == code_id,
                InvitationCode.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def list_invitation_codes_for_course(
    db: AsyncSession, course_id: UUID
) -> list[InvitationCode]:
    result = await db.execute(
        select(InvitationCode)
        .where(
            and_(
                InvitationCode.course_id == course_id,
                InvitationCode.deleted_at.is_(None),
            )
        )
        .order_by(InvitationCode.created_at.desc())
    )
    return list(result.scalars().all())


_LOOKUP_USERS_BY_EMAIL_SQL = text(
    """
    SELECT id, primary_email
    FROM users
    WHERE primary_email = ANY(:emails)
      AND status != 'archived'
    """
)


async def lookup_users_by_email(
    db: AsyncSession, emails: list[str]
) -> list[dict[str, Any]]:
    if not emails:
        return []
    rows = (await db.execute(_LOOKUP_USERS_BY_EMAIL_SQL, {"emails": emails})).mappings()
    return [dict(row) for row in rows]


_GET_COURSE_ORG_SQL = text(
    "SELECT organization_id FROM courses WHERE id = :course_id AND deleted_at IS NULL"
)


async def get_course_organization_id(
    db: AsyncSession, course_id: UUID
) -> UUID | None:
    result = await db.execute(_GET_COURSE_ORG_SQL, {"course_id": course_id})
    row = result.scalar_one_or_none()
    return None if row is None else UUID(str(row))


_INSERT_USER_SQL = text(
    """
    INSERT INTO users (id, primary_email, status, created_at, updated_at)
    VALUES (:id, :primary_email, 'active', NOW(), NOW())
    """
)

_INSERT_USER_PROFILE_SQL = text(
    """
    INSERT INTO user_profiles
        (user_id, given_name, family_name, display_name)
    VALUES (:user_id, :given_name, :family_name, :display_name)
    """
)


async def insert_user_with_profile(
    db: AsyncSession,
    *,
    user_id: UUID,
    primary_email: str,
    given_name: str | None,
    family_name: str | None,
    display_name: str | None,
) -> None:
    await db.execute(
        _INSERT_USER_SQL,
        {"id": user_id, "primary_email": primary_email},
    )
    await db.execute(
        _INSERT_USER_PROFILE_SQL,
        {
            "user_id": user_id,
            "given_name": given_name,
            "family_name": family_name,
            "display_name": display_name or primary_email,
        },
    )
    await db.flush()


__all__ = [
    "find_enrollment",
    "find_invitation_code_by_string",
    "get_course_organization_id",
    "get_invitation_code",
    "insert_user_with_profile",
    "list_enrollments_for_course",
    "list_enrollments_for_user",
    "list_invitation_codes_for_course",
    "lookup_users_by_email",
]
