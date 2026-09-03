"""Service: per-quiz learner progress for the course-learn screen.

Answers "which quiz items in this course are done?" for the calling
student, using the teacher-configured milestone (``passing_score_percent``
via the materialised gradebook) plus the retake policy.

Completion rule (user-specified, 2026-08-03): a quiz is completed when the
student PASSED it (headline grade-of-record reduced per ``grading_method``)
OR failed with every allowed attempt consumed and no attempt still in
flight — a failed-but-exhausted quiz is terminal and stops being the
"next thing to do". An in-flight attempt keeps the quiz open even when the
slot count is exhausted, because the student can still resume it.

Layering: services -> queries. Owns its own DB reads (precedent:
``services/taking.py``, ``services/gradebook.py``). Policy resolution
delegates to :func:`services.overrides.resolve_policy_for_student` so the
effective ``max_attempts`` / ``allow_retakes`` match the start-attempt
gate exactly (``allow_retakes=FALSE`` clamps to 1; Phase 5 overrides win).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from sqlalchemy import select, text

from abridgeai.features.quizzes.models import Quiz
from abridgeai.features.quizzes.services.overrides import resolve_policy_for_student
from abridgeai.features.quizzes.services.review_visibility import (
    resolve_review_visibility,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_QUIZ_IDS_SQL = """
SELECT q.id
FROM quizzes q
JOIN modules m ON m.id = q.module_id
JOIN module_items mi ON mi.quiz_id = q.id
WHERE m.course_id = :course_id
  AND q.status = 'published'
  AND q.deleted_at IS NULL
  AND m.deleted_at IS NULL
  AND mi.deleted_at IS NULL
ORDER BY m.position, mi.position
"""

_ATTEMPT_COUNTS_SQL = """
SELECT quiz_id,
       COUNT(*) AS used,
       COUNT(*) FILTER (WHERE status = 'in_progress') AS in_flight,
       MAX(submitted_at) AS latest_submitted_at
FROM quiz_attempts
WHERE student_id = :user_id
  AND quiz_id = ANY(:quiz_ids)
GROUP BY quiz_id
"""

_GRADES_SQL = """
SELECT quiz_id, grade_percent, passed
FROM quiz_grades
WHERE student_id = :user_id
  AND quiz_id = ANY(:quiz_ids)
  AND grade_item_id IS NULL
"""


async def list_my_quiz_progress(
    db: AsyncSession,
    *,
    course_id: uuid.UUID,
    user_id: uuid.UUID,
) -> list[dict]:
    """Per-published-quiz completion state for ``user_id`` in ``course_id``.

    Only quizzes linked from a live ``module_items`` row in a published,
    non-deleted module are reported — the same population the curriculum
    screen renders, so the completion map keys line up with ``target.id``
    on quiz items. Empty list when the course has no such quizzes (the
    caller is expected to have already applied the org/enrollment gate).
    """
    id_rows = (
        await db.execute(text(_QUIZ_IDS_SQL), {"course_id": course_id})
    ).all()
    quiz_ids = [row[0] for row in id_rows]
    if not quiz_ids:
        return []

    quizzes_stmt = select(Quiz).where(
        Quiz.id.in_(quiz_ids),
        Quiz.status == "published",
    )
    quizzes = (await db.execute(quizzes_stmt)).scalars().all()
    by_id = {q.id: q for q in quizzes}
    ordered = [by_id[qid] for qid in quiz_ids if qid in by_id]
    if not ordered:
        return []

    counts = {
        row.quiz_id: row
        for row in (
            await db.execute(
                text(_ATTEMPT_COUNTS_SQL),
                {"user_id": user_id, "quiz_ids": quiz_ids},
            )
        ).mappings().all()
    }
    grades = {
        row.quiz_id: row
        for row in (
            await db.execute(
                text(_GRADES_SQL),
                {"user_id": user_id, "quiz_ids": quiz_ids},
            )
        ).mappings().all()
    }

    payload: list[dict] = []
    for quiz in ordered:
        policy = await resolve_policy_for_student(db, quiz, user_id)
        used_row = counts.get(quiz.id)
        used = int(used_row["used"]) if used_row else 0
        in_flight = int(used_row["in_flight"]) if used_row else 0

        grade_row = grades.get(quiz.id)
        passed = bool(grade_row["passed"]) if grade_row else None
        grade_percent = grade_row["grade_percent"] if grade_row else None
        latest_submitted_at = used_row["latest_submitted_at"] if used_row else None

        # Effective ceiling mirrors the start-attempt gate:
        # allow_retakes=FALSE clamps to 1 regardless of max_attempts.
        eff_max: int | None = 1 if not policy.allow_retakes else policy.max_attempts
        exhausted = eff_max is not None and used >= eff_max
        completed = bool(passed) or (exhausted and in_flight == 0)
        visibility = resolve_review_visibility(
            quiz,
            SimpleNamespace(submitted_at=latest_submitted_at),
            datetime.now(UTC),
        )
        public_passed = passed if visibility.show_score else None
        public_grade_percent = grade_percent if visibility.show_score else None

        payload.append(
            {
                "quiz_id": quiz.id,
                "attempts_used": used,
                "max_attempts": eff_max,
                "allow_retakes": bool(policy.allow_retakes),
                "passed": public_passed,
                "grade_percent": public_grade_percent,
                "completed": completed,
                "attempts_remaining": (
                    None if eff_max is None else max(0, eff_max - used)
                ),
            }
        )
    return payload


__all__ = ["list_my_quiz_progress"]
