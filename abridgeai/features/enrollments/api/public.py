"""Public, typed cross-feature API for the enrollments feature.

Sibling features (SR cohort queries, admin dashboards, career-path
auto-enroll) MUST import from this module rather than reaching into
``models``/``queries``/``services`` directly. Reads return
:class:`EnrollmentDTO`; ``is_user_enrolled`` is a boolean shortcut on top
of the same query. The one write, :func:`ensure_course_enrollment`, is an
idempotent "make it so" that career paths use to grant course access.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.enrollments.models import Enrollment
from abridgeai.features.enrollments.services import manager as manager_service

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


async def ensure_course_enrollment(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
    actor_id: UUID,
    source: str = "manager_bulk",
) -> None:
    """Idempotently ensure ``student`` has an active enrollment in ``course``.

    Cross-feature write surface for career-path auto-enroll: no-op if the
    student is already enrolled, reactivates a dropped enrollment, else
    creates one. Never raises ``already_enrolled``. Caller owns the
    transaction (no commit here).
    """
    await manager_service.ensure_enrollment(
        db,
        course_id=course_id,
        student_id=student_id,
        actor_id=actor_id,
        source=source,
    )


__all__ = [
    "EnrollmentDTO",
    "ensure_course_enrollment",
    "get_course_enrollment",
    "is_user_enrolled",
]
