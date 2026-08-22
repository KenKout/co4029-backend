"""Course completion writer — the ``course_enrollments.status`` maintainer.

Nothing wrote ``course_enrollments.status = 'completed'`` before this module.
The career-path feature *reads* that column as its definition of "satisfied"
(D2)::

    satisfied(course) ⟺ course_enrollments.status = 'completed'

so without a writer the column was permanently ``'active'`` and no stage
could ever unlock. (``career_paths.services.enrollment.
sync_enrollment_completion`` looks like the writer from its name, but it
flips the *career* enrollment row, not the course one.)

What counts as complete
-----------------------
Every gradeable curriculum UNIT must be done — lessons, quizzes and
interviews. See :mod:`abridgeai.features.enrollments.queries.completion_units`
for the per-kind rules; they are the same rules the curriculum screen renders,
deliberately, so what a student sees ticked is what unlocks their next stage.

This replaced an average over published lessons that ignored quiz and
interview items entirely: measured on this database, all four courses carrying
quizzes were completable without answering one, and a stage gated on such a
course unlocked on lessons alone.

Why synchronous
---------------
Firing this only from a lazy read plus a nightly cron means a student who
finishes their last required course at 9pm sees stage 2 stay locked until
tomorrow. So the primary trigger is synchronous, from every write point that
can change a unit's state:

* lesson  — :mod:`progress.services.tracking` (mark/unmark/engagement)
* quiz    — ``quizzes.services.gradebook.recompute_final_grade``
* interview — ``interviews.services.evaluation`` once ``pass_verdict`` lands

Drift backstop
--------------
:func:`resync_stale_course_completions` is that backstop, run nightly by
``enrollments.workers.completion_drift``. Every synchronous call site swallows
its own failures (a student's mark-complete must not fail because a
side-effect could not be computed), so without a sweeper a single missed write
left ``satisfied`` wrong forever. Earlier revisions of this docstring claimed
the nightly readiness cron covered it — it does not: that job calls
``sync_enrollment_completion``, which flips the *career* row and never touches
this one.

Demotion
--------
Completion can move backward: ``unmark_lesson_complete`` recomputes lesson
status from engagement, a regrade can drop a quiz below its milestone. This
module demotes ``'completed' → 'active'`` in those cases, which keeps
``satisfied`` honest.

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
from abridgeai.core.observability import get_logger
from abridgeai.features.enrollments.queries import completion_units as completion_unit_queries
from abridgeai.features.enrollments.queries.authoring import find_enrollment
from abridgeai.features.enrollments.queries.completion_units import get_course_unit_tally

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


async def sync_course_completion(
    db: AsyncSession,
    *,
    course_id: UUID,
    student_id: UUID,
) -> str | None:
    """Recompute ``course_enrollments.status`` from curriculum UNIT progress.

    A course is complete when every gradeable unit it contains is done —
    lessons, quizzes AND interviews (see
    ``queries.completion_units`` for the per-kind rules, which are the same
    ones the curriculum screen renders). It used to average lesson progress
    alone, which let a course carrying quizzes complete without a single
    question answered.

    Promotes ``'active' → 'completed'`` when all units are done and demotes
    ``'completed' → 'active'`` when they are not. Returns the new status when
    it changed, else ``None`` (so a caller can decide whether to commit).
    Idempotent. Caller owns the transaction — no commit here.

    No row ⇒ nothing to do: under Pattern B (lazy enrollment) a student can
    have progress on a course they were never enrolled in, and inventing an
    enrollment here would be an implicit eager enroll through the back door.

    A course with **zero** gradeable units is never promoted — an empty course
    is not an achievement, and promoting one would hand out a stage unlock for
    no work. The career-path publish gate rejects such a course precisely
    because this writer refuses it.

    Per-student quiz overrides (``quiz_overrides``, resolved by
    ``quizzes.services.overrides``) are not applied by the aggregate query: it
    reads the quiz's own ``allow_retakes``/``max_attempts``. That only affects
    the "failed and exhausted" branch — a PASSED quiz counts identically
    either way — so an override that grants extra attempts can leave this
    writer treating a quiz as terminal slightly early. Tracked as a known
    narrowing rather than duplicating the override resolver in SQL; the
    authoritative per-item read stays ``learner_progress``.
    """
    enrollment = await find_enrollment(db, course_id, student_id)
    if enrollment is None:
        return None
    # Only the active/completed pair participates. Promoting a waitlisted row
    # would grant a place; promoting a dropped row would resurrect it.
    if enrollment.status not in {"active", "completed"}:
        return None

    tally = await get_course_unit_tally(db, course_id=course_id, student_id=student_id)
    should_be_complete = tally.is_complete()

    if should_be_complete and enrollment.status != "completed":
        enrollment.status = "completed"
        enrollment.completed_at = datetime.now(tz=UTC)
        await flush_or_conflict(db)
        from abridgeai.features.learning_programs.api import public as programs_api

        await programs_api.ensure_completion_award(
            db,
            student_id=student_id,
            course_id=course_id,
            source_enrollment_id=enrollment.id,
        )
        return "completed"

    if not should_be_complete and enrollment.status == "completed":
        # Learning progress may demote, but the academic completion award is
        # intentionally immutable and remains available to future paths.
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


async def resync_stale_course_completions(db: AsyncSession) -> tuple[int, int]:
    """Drift backstop: recompute every eligible enrollment. ``(scanned, fixed)``.

    The real backstop the module docstring promises. Every synchronous call
    site deliberately swallows its own exceptions so a student action can never
    fail because a completion side-effect broke — which means a lost write is
    invisible and permanent without a sweeper. ``satisfied`` being wrong does
    not degrade gracefully: it either hands out a career-path stage unlock
    nobody earned or withholds one that was.

    Caller owns the transaction. Each pair runs in its own SAVEPOINT so one
    bad row cannot poison the session and lose the whole nightly batch —
    the same shape ``career_paths.services.readiness.
    snapshot_all_active_enrollments`` uses.
    """
    pairs = await completion_unit_queries.list_completion_candidate_pairs(db)
    fixed = 0
    for course_id, student_id in pairs:
        try:
            async with db.begin_nested():
                changed = await sync_course_completion(
                    db, course_id=course_id, student_id=student_id
                )
            if changed is not None:
                fixed += 1
                logger.info(
                    "enrollments.completion_drift_repaired",
                    course_id=str(course_id),
                    student_id=str(student_id),
                    new_status=changed,
                )
        except Exception:
            logger.exception(
                "enrollments.completion_drift_failed",
                course_id=str(course_id),
                student_id=str(student_id),
            )
    return len(pairs), fixed


__all__ = [
    "resync_stale_course_completions",
    "sync_completion_for_lesson",
    "sync_course_completion",
]
