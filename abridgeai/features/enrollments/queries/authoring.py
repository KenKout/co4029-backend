from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import and_, select, text

from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.enrollments.models import Enrollment, InvitationCode
from abridgeai.features.identity.models import User, UserProfile

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_enrollments_for_course(db: AsyncSession, course_id: UUID) -> list[Enrollment]:
    result = await db.execute(
        select(Enrollment)
        .where(Enrollment.course_id == course_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return list(result.scalars().all())


async def list_enrollments_for_course_with_identity(
    db: AsyncSession, course_id: UUID
) -> list[dict[str, Any]]:
    """Same rows as :func:`list_enrollments_for_course`, plus the
    enrolled user's ``primary_email`` / ``display_name``.

    The Manager enrollments-roster tab was rendering a raw
    ``student_id`` UUID with no name because the plain ORM read never
    joined identity. This mirrors the join
    ``courses.queries.authoring.list_course_roster`` already does for
    the sibling teacher-roster endpoint.
    """
    result = await db.execute(
        select(
            Enrollment.id,
            Enrollment.course_id,
            Enrollment.student_id,
            Enrollment.status,
            Enrollment.source,
            Enrollment.invitation_code_id,
            Enrollment.enrolled_at,
            Enrollment.completed_at,
            Enrollment.dropped_at,
            Enrollment.created_at,
            Enrollment.updated_at,
            Enrollment.created_by,
            Enrollment.updated_by,
            User.primary_email,
            UserProfile.display_name,
        )
        .join(User, User.id == Enrollment.student_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(Enrollment.course_id == course_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return [dict(row) for row in result.mappings().all()]


async def list_enrollments_for_user(db: AsyncSession, user_id: UUID) -> list[Enrollment]:
    result = await db.execute(
        select(Enrollment)
        .where(Enrollment.student_id == user_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return list(result.scalars().all())


async def find_enrollment(db: AsyncSession, course_id: UUID, student_id: UUID) -> Enrollment | None:
    result = await db.execute(
        select(Enrollment).where(
            and_(
                Enrollment.course_id == course_id,
                Enrollment.student_id == student_id,
            )
        )
    )
    return result.scalar_one_or_none()


async def find_invitation_code_by_string(db: AsyncSession, code: str) -> InvitationCode | None:
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


async def lookup_users_by_email(db: AsyncSession, emails: list[str]) -> list[dict[str, Any]]:
    if not emails:
        return []
    rows = (await db.execute(_LOOKUP_USERS_BY_EMAIL_SQL, {"emails": emails})).mappings()
    return [dict(row) for row in rows]


async def get_course_organization_id(db: AsyncSession, course_id: UUID) -> UUID | None:
    course = await courses_api.get_course_by_id(db, course_id)
    return course.organization_id if course is not None else None


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
