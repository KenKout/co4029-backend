"""Service: per-interview learner progress for the course-learn screen.

Answers "which interview items in this course are passed?" for the calling
student. Interviews were already graded per attempt — ``pass_verdict`` is
written by :func:`services.evaluation.evaluate_and_generate_report` — but that
verdict never surfaced on the curriculum, so an interview item stayed
"pending" forever and a module containing one could never auto-collapse.

Completion rule (user decision, 2026-08-06): an interview item is COMPLETED
when the student has **at least one non-practice attempt with
``pass_verdict = TRUE``**.

This is DELIBERATELY NOT the quiz rule. A quiz also completes on
"failed with every attempt consumed", because a failed-but-exhausted quiz is
terminal. The user chose the stricter rule here so the curriculum tag means
*passed*, not merely *finished*: a student who has failed every interview
attempt keeps the item pending. Two consequences worth stating out loud:

* ``max_attempts`` is NULL (unlimited) on every interview config in this
  deployment, so a "failed and exhausted" branch would essentially never fire
  anyway — the strict rule costs almost nothing in practice.
* Failing therefore leaves the item as the "next thing to do", which is the
  intended reading: the interview is not done until it is passed.

Layering: services -> queries. Owns its own DB reads, matching the quiz
counterpart (``quizzes/services/learner_progress.py``) and the local
precedent of ``services/taking.py``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Interview configs reachable from the curriculum, in render order.
#
# Mirrors the quiz query's population rule exactly: only configs linked from a
# live ``module_items`` row in a published, non-deleted module, and only when
# the config itself is published and not soft-deleted. That is what makes the
# returned keys line up with ``ModuleItemPublic.target.id`` on interview items,
# so the frontend can map them without translation.
#
# The config-level filter is NOT optional. ``courses.visibility.
# published_interview_clause`` and ``courses.queries.published`` both gate
# interview items on ``interview_configs.status = 'published'`` and
# ``deleted_at IS NULL``, so omitting it here reported interviews the
# curriculum never renders (measured on this database: a course showing 0
# interview items reported 3, and 5 curriculum rows point at draft or deleted
# configs). That mismatch is now load-bearing: course completion counts
# interview units, so an invisible draft interview would be an unsatisfiable
# unit that locks the course — and every career-path stage behind it — forever.
_INTERVIEW_IDS_SQL = """
SELECT mi.interview_config_id AS config_id
FROM module_items mi
JOIN modules m ON m.id = mi.module_id
JOIN interview_configs ic ON ic.id = mi.interview_config_id
WHERE m.course_id = :course_id
  AND mi.item_type = 'interview'
  AND mi.interview_config_id IS NOT NULL
  AND mi.deleted_at IS NULL
  AND m.deleted_at IS NULL
  AND ic.status = 'published'
  AND ic.deleted_at IS NULL
ORDER BY m.position, mi.position
"""

# Per-config attempt rollup for one student.
#
# ``passed`` is the whole point: a single TRUE verdict on any attempt completes
# the item. ``attempts_used`` remains the audit/history count, while
# ``attempts_awaiting_grade`` counts only terminal statuses an evaluator can
# actually process. ``abandoned`` and ``failed`` rows have no future verdict, so
# including them in the grading gap made the learner-facing badge say "Being
# marked" forever after the attempt history had already settled.
_ATTEMPT_ROLLUP_SQL = """
SELECT s.interview_config_id AS config_id,
       COUNT(*) AS attempts_used,
       COUNT(*) FILTER (WHERE s.status = 'in_progress') AS in_flight,
       BOOL_OR(s.pass_verdict IS TRUE) AS passed,
       COUNT(*) FILTER (WHERE s.pass_verdict IS NOT NULL) AS graded,
       COUNT(*) FILTER (
           WHERE s.status IN ('completed', 'timed_out')
             AND s.pass_verdict IS NULL
       ) AS awaiting_grade
FROM interview_sessions s
WHERE s.student_id = :user_id
  AND s.interview_config_id = ANY(:config_ids)
GROUP BY s.interview_config_id
"""


async def list_my_interview_progress(
    db: AsyncSession,
    *,
    course_id: uuid.UUID,
    user_id: uuid.UUID,
) -> list[dict]:
    """Per-interview completion state for ``user_id`` in ``course_id``.

    Only interview configs linked from a live ``module_items`` row in a
    published, non-deleted module are reported — the same population the
    curriculum screen renders, so the keys line up with ``target.id`` on
    interview items. Empty list when the course has no such interviews (the
    caller is expected to have already applied the org/enrollment gate).
    """
    id_rows = (await db.execute(text(_INTERVIEW_IDS_SQL), {"course_id": course_id})).all()
    config_ids = [row[0] for row in id_rows]
    if not config_ids:
        return []

    rollup = {
        row.config_id: row
        for row in (
            await db.execute(
                text(_ATTEMPT_ROLLUP_SQL),
                {"user_id": user_id, "config_ids": config_ids},
            )
        )
        .mappings()
        .all()
    }

    payload: list[dict] = []
    for config_id in config_ids:
        row = rollup.get(config_id)
        attempts_used = int(row["attempts_used"]) if row else 0
        in_flight = int(row["in_flight"]) if row else 0
        graded = int(row["graded"]) if row else 0
        awaiting_grade = int(row["awaiting_grade"]) if row else 0
        # BOOL_OR yields NULL for a group of all-NULL verdicts; treat any
        # non-TRUE as not passed rather than leaking None into `completed`.
        passed = bool(row["passed"]) if row and row["passed"] is not None else False

        payload.append(
            {
                "interview_config_id": config_id,
                "attempts_used": attempts_used,
                "attempts_in_flight": in_flight,
                "attempts_graded": graded,
                "attempts_awaiting_grade": awaiting_grade,
                "passed": passed,
                # The user-chosen rule, in one place: passing is the ONLY way an
                # interview item completes.
                "completed": passed,
            }
        )
    return payload


__all__ = ["list_my_interview_progress"]
