"""Analytics quiz queries (cohort / outcome surface).

Plan §5504-5506. Aggregates over ``quiz_attempts`` and
``quiz_attempt_answers`` for teacher analytics dashboards.

* :func:`quiz_completion_rate` is a single-quiz aggregate via the ORM.
* :func:`top_missed_questions` cuts across an entire course and is
  pushed down to raw SQL (``sql/top_missed.sql``) for performance —
  the ``GROUP BY`` + ``HAVING`` shape doesn't translate cleanly to a
  single ORM expression on async SQLAlchemy and the rowcount can be
  large enough that the round-trip cost matters.

Soft-delete: ``quiz_attempts`` / ``quiz_attempt_answers`` are NOT in
``SOFT_DELETE_TABLES`` (per T5.1 baseline-canon), so no implicit
filter applies. Quiz / question rows ARE soft-delete eligible — the
ORM aggregate trusts the loader-criteria listener; the raw SQL
explicitly checks ``deleted_at IS NULL`` at every level.
"""

from __future__ import annotations

from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import case, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.quizzes.models import Quiz, QuizAttempt

_TOP_MISSED_SQL = text(
    resources.files("abridgeai.features.quizzes.queries.sql")
    .joinpath("top_missed.sql")
    .read_text(encoding="utf-8")
)


async def quiz_completion_rate(db: AsyncSession, quiz_id: UUID) -> dict[str, Any]:
    """Aggregate completion / pass stats for a single quiz.

    Returns a dict with:

    * ``total_attempts`` — every attempt row, regardless of status.
    * ``completed_count`` — attempts in ``submitted`` / ``graded``.
    * ``avg_score`` — mean ``score_percent`` across completed attempts
      (``None`` when there are no completed attempts).
    * ``pass_rate`` — fraction of completed attempts with
      ``passed=TRUE`` (``None`` when there are no completed attempts).
    """
    completed_statuses = ("submitted", "graded")
    stmt = select(
        func.count(QuizAttempt.id).label("total"),
        func.sum(case((QuizAttempt.status.in_(completed_statuses), 1), else_=0)).label("completed"),
        func.avg(
            case(
                (QuizAttempt.status.in_(completed_statuses), QuizAttempt.score_percent),
                else_=None,
            )
        ).label("avg_score"),
        func.avg(
            case(
                (
                    QuizAttempt.status.in_(completed_statuses),
                    case((QuizAttempt.passed.is_(True), 1.0), else_=0.0),
                ),
                else_=None,
            )
        ).label("pass_rate"),
    ).where(QuizAttempt.quiz_id == quiz_id)
    row = (await db.execute(stmt)).one()
    completed = int(row.completed or 0)
    return {
        "total_attempts": int(row.total or 0),
        "completed_count": completed,
        "avg_score": float(row.avg_score) if row.avg_score is not None else None,
        "pass_rate": float(row.pass_rate) if row.pass_rate is not None else None,
    }


async def list_attempts_for_course(db: AsyncSession, course_id: UUID) -> list[Any]:
    """Every quiz attempt (any student, any quiz) in a course, newest first.

    Powers the teacher's course-wide "Assessments" tab. Returns SQLAlchemy
    ``Row`` objects with ``.QuizAttempt`` and ``.title`` (the quiz title,
    aliased so the router doesn't need a second round-trip). Callers
    resolve student display names separately via a batched lookup —
    mirrors the pattern in ``interviews.routers.authoring.list_config_sessions``.
    """
    stmt = (
        select(QuizAttempt, Quiz.title)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .where(Quiz.course_id == course_id)
        .order_by(QuizAttempt.started_at.desc())
    )
    return list((await db.execute(stmt)).all())


async def list_attempts_for_student_in_course(
    db: AsyncSession, course_id: UUID, student_id: UUID
) -> list[Any]:
    """Every quiz attempt by one student across a course's quizzes, newest first.

    Powers the teacher's per-student profile quiz-attempts section. Same
    row shape as :func:`list_attempts_for_course`.
    """
    stmt = (
        select(QuizAttempt, Quiz.title)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .where(Quiz.course_id == course_id, QuizAttempt.student_id == student_id)
        .order_by(QuizAttempt.started_at.desc())
    )
    return list((await db.execute(stmt)).all())


async def top_missed_questions(
    db: AsyncSession,
    course_id: UUID,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Top-N questions in a course with the lowest mean correctness.

    Reads ``sql/top_missed.sql``. Filters out questions with fewer
    than 5 attempts so the ranking is statistically meaningful.
    Returns ``[{question_id, prompt, correctness_rate, attempt_count}, ...]``
    sorted by ``correctness_rate`` ascending.
    """
    rows = (
        await db.execute(
            _TOP_MISSED_SQL,
            {"course_id": course_id, "limit": limit},
        )
    ).all()
    return [
        {
            "question_id": row.question_id,
            "prompt": row.prompt,
            "correctness_rate": float(row.correctness_rate),
            "attempt_count": int(row.attempt_count),
        }
        for row in rows
    ]


__all__ = [
    "list_attempts_for_course",
    "list_attempts_for_student_in_course",
    "quiz_completion_rate",
    "top_missed_questions",
]
