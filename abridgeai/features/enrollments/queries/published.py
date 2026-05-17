from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import and_, select

from abridgeai.features.enrollments.models import Enrollment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_user_enrollment_for_course(
    db: AsyncSession, user_id: UUID, course_id: UUID
) -> Enrollment | None:
    result = await db.execute(
        select(Enrollment).where(
            and_(
                Enrollment.student_id == user_id,
                Enrollment.course_id == course_id,
            )
        )
    )
    return result.scalar_one_or_none()


async def list_my_enrollments(db: AsyncSession, user_id: UUID) -> list[Enrollment]:
    result = await db.execute(
        select(Enrollment)
        .where(Enrollment.student_id == user_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    return list(result.scalars().all())


__all__ = ["get_user_enrollment_for_course", "list_my_enrollments"]
