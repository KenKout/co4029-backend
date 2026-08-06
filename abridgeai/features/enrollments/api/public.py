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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.enrollments.models import Enrollment
from abridgeai.features.enrollments.queries import completion_units as completion_unit_queries
from abridgeai.features.enrollments.services import completion as completion_service
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


async def has_active_or_completed_enrollment(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
) -> bool:
    """Return ``True`` iff the student's enrollment is ``active`` or ``completed``.

    The learner content gate (``courses.api.public.can_view_course_content``)
    treats a completed course as still accessible — the student was enrolled
    and keeps read access for review / re-reading — while ``dropped`` and
    ``waitlisted`` rows deny access, matching the BR that an unenrolled
    student must not reach course items.
    """
    stmt = select(Enrollment.id).where(
        Enrollment.student_id == student_id,
        Enrollment.course_id == course_id,
        Enrollment.status.in_(("active", "completed")),
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


async def list_active_student_ids(db: AsyncSession, *, course_id: UUID) -> list[UUID]:
    """Return the user ids of all students with an ``active`` enrollment.

    Cross-feature read used by the courses publish flow to notify everyone
    currently enrolled when a course goes live. ``dropped`` / ``completed`` /
    ``waitlisted`` rows are excluded.
    """
    stmt = select(Enrollment.student_id).where(
        Enrollment.course_id == course_id,
        Enrollment.status == "active",
    )
    return list((await db.execute(stmt)).scalars().all())


async def sync_course_completion_for_lesson(
    db: AsyncSession,
    *,
    lesson_id: UUID,
    student_id: UUID,
) -> str | None:
    """Recompute ``course_enrollments.status`` for the lesson's course.

    The D2 writer, exposed cross-feature so ``progress.services.tracking``
    can fire it **synchronously** on every lesson-progress write. Returns
    the new status when it changed, else ``None``. Promotes at 100% and
    demotes below it; never touches the append-only stage latch. Caller owns
    the transaction.
    """
    return await completion_service.sync_completion_for_lesson(
        db, lesson_id=lesson_id, student_id=student_id
    )


async def sync_course_completion(
    db: AsyncSession,
    *,
    course_id: UUID,
    student_id: UUID,
) -> str | None:
    """:func:`sync_course_completion_for_lesson` for a known course id.

    The entry point for the quiz and interview write sites, which already know
    the course. Completion counts every gradeable curriculum unit — lessons,
    quizzes and interviews — so a graded quiz or a passed interview must fire
    this or ``satisfied`` (and the career-path stage gate behind it) goes
    stale until the nightly drift sweep.
    """
    return await completion_service.sync_course_completion(
        db, course_id=course_id, student_id=student_id
    )


async def count_course_gradeable_units(db: AsyncSession, *, course_id: UUID) -> int:
    """Gradeable curriculum units in a course (lessons + quizzes + interviews).

    Counts only what a student can actually see and satisfy: a live
    ``module_items`` row in a non-deleted module pointing at a published,
    non-deleted target. Zero means no student can ever complete the course, so
    the career-path publish gate rejects it — the D2 writer refuses to promote
    an empty course, which would otherwise lock every stage behind it forever.
    """
    counts = await completion_unit_queries.count_course_units(db, course_id=course_id)
    return counts.total


async def resync_stale_course_completions(db: AsyncSession) -> tuple[int, int]:
    """Drift backstop for the D2 writer. Returns ``(scanned, fixed)``.

    Exposed cross-feature for the nightly worker. Every synchronous call site
    swallows its own failures, so without this sweep a lost write leaves
    ``satisfied`` wrong permanently.
    """
    return await completion_service.resync_stale_course_completions(db)


async def count_active_enrollments_in_courses(
    db: AsyncSession, *, student_id: UUID, course_ids: list[UUID]
) -> int:
    """How many of ``course_ids`` the student currently has ``active``.

    Backs the career-path attention cap (``career_paths.max_concurrent``),
    which is counted **path-wide** — the caller passes every course in the
    path, not one stage's worth.
    """
    if not course_ids:
        return 0
    stmt = (
        select(func.count())
        .select_from(Enrollment)
        .where(
            Enrollment.student_id == student_id,
            Enrollment.course_id.in_(course_ids),
            Enrollment.status == "active",
        )
    )
    return int((await db.execute(stmt)).scalar_one())


__all__ = [
    "EnrollmentDTO",
    "count_active_enrollments_in_courses",
    "count_course_gradeable_units",
    "ensure_course_enrollment",
    "get_course_enrollment",
    "has_active_or_completed_enrollment",
    "is_user_enrolled",
    "list_active_student_ids",
    "resync_stale_course_completions",
    "sync_course_completion",
    "sync_course_completion_for_lesson",
]
