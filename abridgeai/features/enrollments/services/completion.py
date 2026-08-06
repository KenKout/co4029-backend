"""Course completion writer — the ``course_enrollments.status`` maintainer.

Nothing wrote ``course_enrollments.status = 'completed'`` before this module.
The career-path feature *reads* that column as its definition of "satisfied"
(D2)::

    satisfied(course) ⟺ course_enrollments.status = 'completed'

so without a writer the column was permanently ``'active'`` and no stage
could ever unlock. (``career_paths.services.enrollment.
sync_enrollment_completion`` looks like the writer from its name, but it
flips the *career* enrollment row, not the course one.)

Why synchronous
---------------
Firing this only from a lazy read plus a nightly cron means a student who
finishes their last required course at 9pm sees stage 2 stay locked until
tomorrow. So the primary trigger is synchronous, on every lesson-progress
write point in :mod:`progress.services.tracking`. The lazy read and the
nightly job remain as a **drift backstop only** — they can repair a missed
write, they are no longer how completion normally happens.

Demotion
--------
Completion can move backward: ``unmark_lesson_complete`` recomputes lesson
status from engagement, so a student can un-tick a lesson and drop a course
below 100%. This module demotes ``'completed' → 'active'`` in that case,
which keeps ``satisfied`` honest.

It deliberately does **not** touch the stage latch. ``student_stage_progress``
is append-only, so a stage that ever completed stays complete. The resulting
asymmetry — course un-satisfied, stage still complete — is intentional: it is
the same protection that shields a student from a manager editing the path
under them, and it applies to self-inflicted un-completion too.

``waitlisted`` and ``dropped`` rows are never touched: promotion out of
either would silently grant a place (waitlist) or resurrect an enrollment the
student left (dropped).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.features.enrollments.queries import published as published_queries
from abridgeai.features.enrollments.queries.authoring import find_enrollment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_COMPLETE_AT_PERCENT = 100.0


async def sync_course_completion(
    db: AsyncSession,
    *,
    course_id: UUID,
    student_id: UUID,
) -> str | None:
    """Recompute ``course_enrollments.status`` from lesson progress.

    Promotes ``'active' → 'completed'`` at 100% and demotes
    ``'completed' → 'active'`` below it. Returns the new status when it
    changed, else ``None`` (so a caller can decide whether to commit).
    Idempotent. Caller owns the transaction — no commit here.

    No row ⇒ nothing to do: under Pattern B (lazy enrollment) a student can
    have progress on a course they were never enrolled in, and inventing an
    enrollment here would be an implicit eager enroll through the back door.

    A course with zero published lessons is never promoted — averaging over
    an empty set yields 0%, and an empty course is not an achievement.
    """
    enrollment = await find_enrollment(db, course_id, student_id)
    if enrollment is None:
        return None
    # Only the active/completed pair participates. Promoting a waitlisted row
    # would grant a place; promoting a dropped row would resurrect it.
    if enrollment.status not in {"active", "completed"}:
        return None

    lesson_count, percent = await published_queries.get_course_completion_percent(
        db, course_id=course_id, student_id=student_id
    )
    should_be_complete = lesson_count > 0 and percent >= _COMPLETE_AT_PERCENT

    if should_be_complete and enrollment.status != "completed":
        enrollment.status = "completed"
        enrollment.completed_at = datetime.now(tz=UTC)
        await flush_or_conflict(db)
        return "completed"

    if not should_be_complete and enrollment.status == "completed":
        # Demotion: keeps `satisfied` honest when a lesson is un-marked.
        # The stage latch is deliberately NOT unwound (see module docstring).
        enrollment.status = "active"
        enrollment.completed_at = None
        await flush_or_conflict(db)
        return "active"

    return None


async def sync_completion_for_lesson(
    db: AsyncSession,
    *,
    lesson_id: UUID,
    student_id: UUID,
) -> str | None:
    """:func:`sync_course_completion` for whichever course owns ``lesson_id``.

    The lesson-progress call sites know a lesson, not a course, so they need
    this resolving wrapper. A lesson with no resolvable course (deleted
    mid-flight) is a no-op rather than an error: failing a student's
    mark-complete because a completion side-effect could not resolve would
    trade a real capability for a bookkeeping one.
    """
    from abridgeai.features.courses.api import public as courses_api  # noqa: PLC0415

    lesson = await courses_api.get_lesson_by_id(db, lesson_id)
    if lesson is None:
        return None
    module = await courses_api.get_module_by_id(db, lesson.module_id)
    if module is None:
        return None
    return await sync_course_completion(db, course_id=module.course_id, student_id=student_id)


__all__ = ["sync_completion_for_lesson", "sync_course_completion"]
