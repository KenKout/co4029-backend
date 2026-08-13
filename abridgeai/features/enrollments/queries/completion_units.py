"""Course completion measured in graded UNITS, not lessons alone.

Why this file exists
--------------------
Course completion used to be ``AVG(lesson_progress.completion_percent)`` over
published lessons, and nothing else. A curriculum is built from three item
kinds (``module_items.item_type IN ('lesson','quiz','interview')``), so that
metric silently ignored two thirds of the vocabulary: on this database every
course carrying quizzes was completable without answering a single one, and
``course_enrollments.status`` — which career-path stage unlock reads as
``satisfied`` (D2) — flipped to ``'completed'`` on lessons alone.

Completion is therefore counted over units, one per curriculum item:

=========  ==============================================================
lesson     ``lesson_progress.status = 'completed'``
quiz       passed the teacher's milestone, OR failed with every allowed
           attempt consumed and none still in flight
interview  at least one non-practice attempt with ``pass_verdict = TRUE``
=========  ==============================================================

The quiz and interview rules are NOT invented here. They are the rules the
curriculum screen already renders (``quizzes.services.learner_progress``,
``interviews.services.learner_progress``), restated in SQL so one aggregate
query can answer "is this course done?" without a per-item Python round trip.
Divergence between what the student sees ticked and what unlocks their next
stage is the failure this shape exists to prevent — so the two docstrings
cross-reference each other, and the parity tests assert they agree.

Deliberate asymmetry, kept from the source rules
------------------------------------------------
A quiz completes when it is terminal (passed OR exhausted); an interview
completes only when PASSED. Failing every interview attempt leaves the unit
pending, because the curriculum tag there means *đạt/passed*, not merely
*finished* (user decision 2026-08-06). Practice runs never count.

Populations must match the curriculum exactly
---------------------------------------------
Every unit is gated on the same visibility rules ``courses.queries.published``
applies: a live ``module_items`` row, a non-deleted module, and a published,
non-deleted target. A unit the student cannot see is a unit they cannot
satisfy, and counting one locks the course — and every career-path stage
behind it — forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class CourseUnitTally(NamedTuple):
    """Unit counts for one (course, student) pair.

    ``total`` is 0 for a course with no gradeable curriculum at all. Callers
    must treat that as "not completable" rather than as 100% — see
    :meth:`is_complete`.
    """

    lessons_total: int
    lessons_done: int
    quizzes_total: int
    quizzes_done: int
    interviews_total: int
    interviews_done: int

    @property
    def total(self) -> int:
        return self.lessons_total + self.quizzes_total + self.interviews_total

    @property
    def done(self) -> int:
        return self.lessons_done + self.quizzes_done + self.interviews_done

    @property
    def percent(self) -> float:
        """Unit completion as 0–100. A course with no units reports 0.0.

        0.0 rather than 100.0 on an empty curriculum: an empty course is not an
        achievement, and the D2 writer must never promote one.
        """
        if self.total == 0:
            return 0.0
        return round(self.done * 100.0 / self.total, 2)

    def is_complete(self) -> bool:
        """True only when there is real work and all of it is done."""
        return self.total > 0 and self.done == self.total


# One row of counts for a (course, student) pair.
#
# Each CTE is its own population; the FULL/CROSS joins at the bottom would
# multiply rows, so every branch aggregates to a single row of scalars first
# and the SELECT just reads them. That keeps the query flat and lets a course
# with (say) no interviews contribute 0 instead of dropping the row entirely.
#
# ``effective_max_attempts`` reproduces the quiz start-attempt gate:
# ``allow_retakes = FALSE`` clamps the ceiling to 1 whatever ``max_attempts``
# says. Per-student overrides (``quiz_overrides``) are resolved in Python by
# ``quizzes.services.overrides`` and are NOT visible here — see the caller
# note in ``services/completion.py``: this query is the aggregate fast path,
# and the authoritative per-item read stays the learner_progress services.
_UNIT_TALLY_SQL = text(
    """
WITH lesson_units AS (
    -- Lessons are reached through modules, NOT through module_items.
    --
    -- module_items is the curriculum ORDERING table, and a published lesson
    -- can legitimately have no row in it: measured on this database, one
    -- course has 4 published lessons but only 3 module_items rows. Joining
    -- through module_items dropped that lesson from the denominator, so the
    -- course completed while a real lesson was still unfinished — the exact
    -- class of bug this whole change exists to remove.
    --
    -- Quizzes and interviews are different: they have no other link to a
    -- course's curriculum, so module_items IS their membership signal and
    -- they must join through it.
    SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE lp.status = 'completed') AS done
    FROM modules m
    JOIN lessons l ON l.module_id = m.id
        AND l.deleted_at IS NULL
        AND l.status = 'published'
    LEFT JOIN lesson_progress lp
        ON lp.lesson_id = l.id AND lp.user_id = :student_id
    WHERE m.course_id = :course_id
      AND m.deleted_at IS NULL
),
quiz_pop AS (
    SELECT
        q.id AS quiz_id,
        CASE WHEN q.allow_retakes THEN q.max_attempts ELSE 1 END AS eff_max
    FROM module_items mi
    JOIN modules m ON m.id = mi.module_id AND m.deleted_at IS NULL
    JOIN quizzes q ON q.id = mi.quiz_id
        AND q.deleted_at IS NULL
        AND q.status = 'published'
    WHERE m.course_id = :course_id
      AND mi.item_type = 'quiz'
      AND mi.deleted_at IS NULL
),
quiz_units AS (
    SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (
            WHERE COALESCE(g.passed, FALSE)
               OR (
                    qp.eff_max IS NOT NULL
                    AND COALESCE(a.used, 0) >= qp.eff_max
                    AND COALESCE(a.in_flight, 0) = 0
                  )
        ) AS done
    FROM quiz_pop qp
    LEFT JOIN (
        SELECT quiz_id, passed
        FROM quiz_grades
        WHERE student_id = :student_id AND grade_item_id IS NULL
    ) g ON g.quiz_id = qp.quiz_id
    LEFT JOIN (
        SELECT quiz_id,
               COUNT(*) AS used,
               COUNT(*) FILTER (WHERE status = 'in_progress') AS in_flight
        FROM quiz_attempts
        WHERE student_id = :student_id
        GROUP BY quiz_id
    ) a ON a.quiz_id = qp.quiz_id
),
interview_units AS (
    SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE COALESCE(s.passed, FALSE)) AS done
    FROM module_items mi
    JOIN modules m ON m.id = mi.module_id AND m.deleted_at IS NULL
    JOIN interview_configs ic ON ic.id = mi.interview_config_id
        AND ic.deleted_at IS NULL
        AND ic.status = 'published'
    LEFT JOIN (
        SELECT interview_config_id,
               BOOL_OR(pass_verdict IS TRUE) AS passed
        FROM interview_sessions
        WHERE student_id = :student_id
        GROUP BY interview_config_id
    ) s ON s.interview_config_id = ic.id
    WHERE m.course_id = :course_id
      AND mi.item_type = 'interview'
      AND mi.deleted_at IS NULL
)
SELECT
    lu.total AS lessons_total,     lu.done AS lessons_done,
    qu.total AS quizzes_total,     qu.done AS quizzes_done,
    iu.total AS interviews_total,  iu.done AS interviews_done
FROM lesson_units lu, quiz_units qu, interview_units iu
"""
)


async def get_course_unit_tally(
    db: AsyncSession, *, course_id: UUID, student_id: UUID
) -> CourseUnitTally:
    """Unit tally for one (course, student) pair.

    Every count is scoped to the curriculum the student can actually see, so a
    draft quiz or a soft-deleted interview never becomes an unsatisfiable unit.
    """
    row = (
        await db.execute(_UNIT_TALLY_SQL, {"course_id": course_id, "student_id": student_id})
    ).one()
    return CourseUnitTally(
        lessons_total=int(row.lessons_total),
        lessons_done=int(row.lessons_done),
        quizzes_total=int(row.quizzes_total),
        quizzes_done=int(row.quizzes_done),
        interviews_total=int(row.interviews_total),
        interviews_done=int(row.interviews_done),
    )


# Gradeable-unit count for a course, independent of any student. Used by the
# career-path publish gate to reject a course that no student could ever
# finish (0 units ⇒ the D2 writer never promotes ⇒ the stage never unlocks).
_COURSE_UNIT_COUNT_SQL = text(
    """
SELECT
    (SELECT COUNT(*)
       FROM modules m
       JOIN lessons l ON l.module_id = m.id
           AND l.deleted_at IS NULL AND l.status = 'published'
      WHERE m.course_id = :course_id AND m.deleted_at IS NULL) AS lessons,
    (SELECT COUNT(*)
       FROM module_items mi
       JOIN modules m ON m.id = mi.module_id AND m.deleted_at IS NULL
       JOIN quizzes q ON q.id = mi.quiz_id
           AND q.deleted_at IS NULL AND q.status = 'published'
      WHERE m.course_id = :course_id
        AND mi.item_type = 'quiz' AND mi.deleted_at IS NULL) AS quizzes,
    (SELECT COUNT(*)
       FROM module_items mi
       JOIN modules m ON m.id = mi.module_id AND m.deleted_at IS NULL
       JOIN interview_configs ic ON ic.id = mi.interview_config_id
           AND ic.deleted_at IS NULL AND ic.status = 'published'
      WHERE m.course_id = :course_id
        AND mi.item_type = 'interview' AND mi.deleted_at IS NULL) AS interviews
"""
)


class CourseUnitCounts(NamedTuple):
    lessons: int
    quizzes: int
    interviews: int

    @property
    def total(self) -> int:
        return self.lessons + self.quizzes + self.interviews


async def count_course_units(db: AsyncSession, *, course_id: UUID) -> CourseUnitCounts:
    """Gradeable units in a course, ignoring any student's progress."""
    row = (await db.execute(_COURSE_UNIT_COUNT_SQL, {"course_id": course_id})).one()
    return CourseUnitCounts(
        lessons=int(row.lessons),
        quizzes=int(row.quizzes),
        interviews=int(row.interviews),
    )


_ACTIVE_OR_COMPLETED_PAIRS_SQL = text(
    """
SELECT ce.course_id, ce.student_id
FROM course_enrollments ce
JOIN courses c ON c.id = ce.course_id AND c.deleted_at IS NULL
WHERE ce.status IN ('active', 'completed')
ORDER BY ce.course_id, ce.student_id
"""
)


async def list_completion_candidate_pairs(db: AsyncSession) -> list[tuple[UUID, UUID]]:
    """``(course_id, student_id)`` for every enrollment the D2 writer can move.

    Only ``active``/``completed`` rows: those are the pair
    :func:`enrollments.services.completion.sync_course_completion` will act on,
    and feeding it ``waitlisted``/``dropped`` rows would just be work it
    discards. Soft-deleted courses are skipped.

    Used by the nightly drift sweeper. This is a full scan by design — the
    sweeper's job is to find rows whose synchronous write was LOST, and a
    filter derived from the same progress tables that produced the loss could
    filter out exactly the rows it needs to repair.
    """
    rows = (await db.execute(_ACTIVE_OR_COMPLETED_PAIRS_SQL)).all()
    return [(row.course_id, row.student_id) for row in rows]


__all__ = [
    "count_course_units",
    "get_course_unit_tally",
    "list_completion_candidate_pairs",
    "CourseUnitCounts",
    "CourseUnitTally",
]
