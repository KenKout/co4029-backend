from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.enrollments.queries import published as published_queries
from abridgeai.features.enrollments.schemas import EnrollmentRead

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.enrollments.models import Enrollment


def _to_read(enrollment: Enrollment) -> EnrollmentRead:
    return EnrollmentRead.model_validate(
        {
            "course_id": enrollment.course_id,
            "status": enrollment.status,
            "enrolled_at": enrollment.enrolled_at,
            "completed_at": enrollment.completed_at,
        }
    )


async def list_my_enrollments(db: AsyncSession, user_id: UUID) -> list[EnrollmentRead]:
    rows = await published_queries.list_my_enrollments(db, user_id)
    return [_to_read(row) for row in rows]


async def get_my_enrollment_status(
    db: AsyncSession, user_id: UUID, course_id: UUID
) -> EnrollmentRead | None:
    row = await published_queries.get_user_enrollment_for_course(db, user_id, course_id)
    return None if row is None else _to_read(row)


__all__ = ["get_my_enrollment_status", "list_my_enrollments"]
