"""Public, typed cross-feature read API for the enrollments feature.

Sibling features (SR cohort queries, admin dashboards) MUST import from
this module rather than reaching into ``models``/``queries``/``services``
directly. Reads return :class:`EnrollmentDTO`; ``is_user_enrolled`` is a
boolean shortcut on top of the same query (one row per
``(course_id, student_id)`` pair, enforced by the DB unique constraint).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.enrollments.models import Enrollment

from ._dto import EnrollmentDTO


async def get_course_enrollment(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
) -> EnrollmentDTO | None:
    """Return the single enrollment row for ``(student_id, course_id)``.

    Returns ``None`` if the student has never enrolled (no row). A
    ``status='dropped'`` row is still returned -- callers that care
    about active-only membership should use :func:`is_user_enrolled`
    or filter on the returned ``status`` themselves.
    """
    stmt = select(Enrollment).where(
        Enrollment.student_id == student_id,
        Enrollment.course_id == course_id,
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    return EnrollmentDTO.model_validate(row) if row is not None else None


async def is_user_enrolled(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
) -> bool:
    """Return ``True`` iff the student has an ``active`` enrollment.

    "Enrolled" here means ``status='active'``; ``dropped``,
    ``completed``, and ``waitlisted`` rows return ``False``. Callers
    needing finer state must use :func:`get_course_enrollment`.
    """
    stmt = select(Enrollment.id).where(
        Enrollment.student_id == student_id,
        Enrollment.course_id == course_id,
        Enrollment.status == "active",
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None


__all__ = [
    "EnrollmentDTO",
    "get_course_enrollment",
    "is_user_enrolled",
]
