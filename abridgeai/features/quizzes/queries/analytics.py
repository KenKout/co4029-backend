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


_HEADLINE_SQL_BY_METHOD: dict[str, str] = {
    # Per student, the attempt with the MAX score_percent.
    "highest": (
        "SELECT DISTINCT ON (student_id) "
        "  student_id, score_percent, passed, time_taken_seconds "
        "FROM completed "
        "ORDER BY student_id, score_percent DESC"
    ),
    # Per student, the attempt with the MAX attempt_number (most recent).
    "last": (
        "SELECT DISTINCT ON (student_id) "
        "  student_id, score_percent, passed, time_taken_seconds "
        "FROM completed "
        "ORDER BY student_id, attempt_number DESC"
    ),
    # Per student, the attempt with the MIN attempt_number (earliest).
    "first": (
        "SELECT DISTINCT ON (student_id) "
        "  student_id, score_percent, passed, time_taken_seconds "
        "FROM completed "
        "ORDER BY student_id, attempt_number ASC"
    ),
    # Per student, the mean across all completed attempts. bool_or so a
    # student who passed on any counted attempt is treated as a pass.
    "average": (
        "SELECT student_id, "
        "  AVG(score_percent) AS score_percent, "
        "  bool_or(passed) AS passed, "
        "  AVG(time_taken_seconds) AS time_taken_seconds "
        "FROM completed "
        "GROUP BY student_id"
    ),
}


def _empty_histogram() -> list[dict[str, Any]]:
    """Eleven score buckets, all zero.

    Bucket ``i`` (0..9) spans ``[10*i, 10*i+9]``; the eleventh bucket
    (``i == 10``) is the ``90..100``-inclusive top band and captures a
    perfect ``100``. Bucketing lives in Python (not SQL) so the shape is
    portable and unit-testable.
    """
    buckets: list[dict[str, Any]] = []
    for i in range(11):
        lower = i * 10
        upper = 100 if i == 10 else i * 10 + 9
        label = "90–100" if i == 10 else f"{lower}–{upper}"
        buckets.append({"label": label, "lower": lower, "upper": upper, "count": 0})
    return buckets


def _score_histogram(scores: list[float]) -> list[dict[str, Any]]:
    """Bucket ``scores`` (0..100) into the eleven bands from
    :func:`_empty_histogram`. ``sum(counts)`` equals ``len(scores)``."""
    buckets = _empty_histogram()
    for score in scores:
        idx = int(score // 10)
        idx = min(max(idx, 0), 10)
        buckets[idx]["count"] += 1
    return buckets


async def quiz_results_summary(
    db: AsyncSession, quiz_id: UUID, grading_method: str
) -> dict[str, Any]:
    """Grading-method-aware aggregate stats for a single quiz.

    Reduces COMPLETED attempts (``status IN ('submitted','graded')`` AND
    ``score_percent IS NOT NULL``) to ONE headline row per student per
    ``grading_method`` (``highest`` / ``average`` / ``first`` / ``last``),
    then aggregates over that headline set.

    ``grading_method`` is validated against the exact allowed set BEFORE
    any SQL is built — the method selects a pre-written CTE variant and is
    never string-interpolated, so an unknown value raises
    :class:`ValueError` rather than reaching the database.

    Returns a plain dict with:

    * ``total_attempts`` — COUNT over *all* completed attempts (not deduped).
    * ``unique_students`` — COUNT over the per-student headline rows.
    * ``mean_score`` — AVG ``score_percent`` over headline (``None`` if empty).
    * ``median_score`` / ``p25`` / ``p75`` — ``percentile_cont`` over headline.
    * ``pass_rate`` — fraction of headline rows with ``passed`` truthy (0..1).
    * ``mean_time_seconds`` — AVG ``time_taken_seconds`` over headline.
    * ``histogram`` — 11 score buckets over the headline set (see
      :func:`_score_histogram`); ``sum`` of counts equals ``unique_students``.

    Zero completed attempts yields zeroed counts, ``None`` stats and an
    all-zero 11-bucket histogram — never an error.
    """
    if grading_method not in _HEADLINE_SQL_BY_METHOD:
        allowed = ", ".join(sorted(_HEADLINE_SQL_BY_METHOD))
        msg = f"invalid grading_method {grading_method!r}; expected one of: {allowed}"
        raise ValueError(msg)

    # headline_sql is a checked-in fragment selected by a validated enum
    # (never user input); quiz_id is bound. Hence the S608 suppression.
    headline_sql = _HEADLINE_SQL_BY_METHOD[grading_method]
    sql = (
        "WITH completed AS ("  # noqa: S608  -- headline_sql keyed by validated enum, quiz_id bound
        "  SELECT student_id, score_percent, passed, "
        "         time_taken_seconds, attempt_number "
        "  FROM quiz_attempts "
        "  WHERE quiz_id = :quiz_id "
        "    AND status IN ('submitted', 'graded') "
        "    AND score_percent IS NOT NULL"
        "), headline AS ("
        f"  {headline_sql}"
        ") "
        "SELECT "
        "  (SELECT count(*) FROM completed) AS total_attempts, "
        "  count(*) AS unique_students, "
        "  avg(score_percent) AS mean_score, "
        "  percentile_cont(0.5) WITHIN GROUP (ORDER BY score_percent) AS median_score, "
        "  percentile_cont(0.25) WITHIN GROUP (ORDER BY score_percent) AS p25, "
        "  percentile_cont(0.75) WITHIN GROUP (ORDER BY score_percent) AS p75, "
        "  avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END) AS pass_rate, "
        "  avg(time_taken_seconds) AS mean_time_seconds, "
        "  array_agg(score_percent) AS scores "
        "FROM headline"
    )
    row = (await db.execute(text(sql), {"quiz_id": quiz_id})).one()

    def _f(value: Any) -> float | None:  # noqa: ANN401 -- SQLAlchemy Row attrs are runtime-typed (Decimal|None)
        return float(value) if value is not None else None

    raw_scores = row.scores or []
    scores = [float(s) for s in raw_scores if s is not None]

    return {
        "total_attempts": int(row.total_attempts or 0),
        "unique_students": int(row.unique_students or 0),
        "mean_score": _f(row.mean_score),
        "median_score": _f(row.median_score),
        "p25": _f(row.p25),
        "p75": _f(row.p75),
        "pass_rate": _f(row.pass_rate),
        "mean_time_seconds": _f(row.mean_time_seconds),
        "histogram": _score_histogram(scores),
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


_QUESTION_TYPES_WITH_OPTIONS = ("multiple_choice", "true_false")


async def quiz_question_breakdown(db: AsyncSession, quiz_id: UUID) -> list[dict[str, Any]]:
    """Per-question performance breakdown for a single quiz.

    For each non-soft-deleted question in ``quiz_id``, counts how students
    performed, considering ONLY answers that belong to COMPLETED attempts
    (``status IN ('submitted','graded')``). Unlike :func:`top_missed_questions`
    there is NO minimum-attempts gate — every question is returned so a
    teacher with a small cohort still sees each one.

    Returns a list with ONE dict per question, ordered by ``correctness_rate``
    ascending (hardest first) with question ``position`` ascending as a stable
    tiebreaker. Questions nobody answered sort last (NULL rate) and still
    appear with zeroed counts.

    Each dict carries:

    * ``question_id`` — :class:`~uuid.UUID`.
    * ``prompt`` — ``quiz_questions.prompt_text``.
    * ``correct_count`` — answers where ``is_correct`` (completed attempts).
    * ``answered_count`` — total answers (completed attempts) for the question.
    * ``correctness_rate`` — ``correct_count / answered_count`` (0..1);
      ``None`` when ``answered_count == 0``.
    * ``option_distribution`` — for MCQ (``multiple_choice`` / ``true_false``):
      one entry per option ``{option_id, option_key, option_text, is_correct,
      chosen_count}``; ``[]`` for non-MCQ question types.

    ``chosen_count`` counts completed-attempt answers whose
    ``selected_option_id`` is that option; options nobody picked report ``0``.
    """
    # (a) Per-question correct/answered counts over COMPLETED attempts only.
    # LEFT JOIN so questions with zero answers still return a row.
    per_question_sql = text(
        "SELECT q.id AS question_id, q.prompt_text AS prompt, "
        "       q.question_type AS question_type, q.position AS position, "
        "       count(ans.id) AS answered_count, "
        "       count(ans.id) FILTER (WHERE ans.is_correct) AS correct_count "
        "FROM quiz_questions q "
        "LEFT JOIN quiz_attempt_answers ans ON ans.question_id = q.id "
        "LEFT JOIN quiz_attempts att "
        "  ON att.id = ans.attempt_id AND att.status IN ('submitted', 'graded') "
        "WHERE q.quiz_id = :quiz_id AND q.deleted_at IS NULL "
        "  AND (ans.id IS NULL OR att.id IS NOT NULL) "
        "GROUP BY q.id, q.prompt_text, q.question_type, q.position "
        "ORDER BY q.position ASC"
    )

    # (b) Per-option chosen counts over COMPLETED attempts only.
    # LEFT JOIN so options nobody picked still return chosen_count = 0.
    per_option_sql = text(
        "SELECT o.question_id AS question_id, o.id AS option_id, "
        "       o.option_key AS option_key, o.option_text AS option_text, "
        "       o.is_correct AS is_correct, o.position AS position, "
        "       count(ans.id) AS chosen_count "
        "FROM quiz_question_options o "
        "JOIN quiz_questions q ON q.id = o.question_id "
        "LEFT JOIN quiz_attempt_answers ans ON ans.selected_option_id = o.id "
        "LEFT JOIN quiz_attempts att "
        "  ON att.id = ans.attempt_id AND att.status IN ('submitted', 'graded') "
        "WHERE q.quiz_id = :quiz_id AND q.deleted_at IS NULL "
        "  AND o.deleted_at IS NULL "
        "  AND (ans.id IS NULL OR att.id IS NOT NULL) "
        "GROUP BY o.question_id, o.id, o.option_key, o.option_text, "
        "         o.is_correct, o.position "
        "ORDER BY o.position ASC"
    )

    question_rows = (await db.execute(per_question_sql, {"quiz_id": quiz_id})).all()
    option_rows = (await db.execute(per_option_sql, {"quiz_id": quiz_id})).all()

    # Group options by question_id (already ordered by option position).
    options_by_question: dict[UUID, list[dict[str, Any]]] = {}
    for row in option_rows:
        options_by_question.setdefault(row.question_id, []).append(
            {
                "option_id": row.option_id,
                "option_key": row.option_key,
                "option_text": row.option_text,
                "is_correct": bool(row.is_correct),
                "chosen_count": int(row.chosen_count or 0),
            }
        )

    results: list[dict[str, Any]] = []
    for row in question_rows:
        answered = int(row.answered_count or 0)
        correct = int(row.correct_count or 0)
        rate = (correct / answered) if answered else None
        has_options = row.question_type in _QUESTION_TYPES_WITH_OPTIONS
        distribution = options_by_question.get(row.question_id, []) if has_options else []
        results.append(
            {
                "question_id": row.question_id,
                "prompt": row.prompt,
                "correct_count": correct,
                "answered_count": answered,
                "correctness_rate": rate,
                "option_distribution": distribution,
                "_position": row.position,
            }
        )

    # Order by correctness_rate ASC (hardest first); NULL (unanswered) sorts
    # last; position ASC as a stable tiebreaker. Done in Python so the NULL
    # ordering and rate computation stay together and testable.
    results.sort(
        key=lambda r: (
            r["correctness_rate"] is None,
            r["correctness_rate"] if r["correctness_rate"] is not None else 0.0,
            r["_position"],
        )
    )
    for r in results:
        del r["_position"]
    return results


__all__ = [
    "list_attempts_for_course",
    "list_attempts_for_student_in_course",
    "quiz_completion_rate",
    "quiz_question_breakdown",
    "quiz_results_summary",
    "top_missed_questions",
]
