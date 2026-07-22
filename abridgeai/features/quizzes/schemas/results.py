"""Quiz-results analytics DTOs (T1.5).

Response shapes for the teacher-facing quiz-results endpoint
(``GET /teacher/quizzes/{quiz_id}/results``, T1.6). The endpoint
composes three already-tested analytics queries into one payload:

* :class:`QuizResultsSummary` — grading-method-aware cohort headline
  (mirrors ``queries.analytics.quiz_results_summary``).
* :class:`QuizPerStudentRow` — one row per student with a completed
  attempt (best / latest / count / pass / recency).
* :class:`QuizQuestionBreakdown` — per-question performance
  (mirrors ``queries.analytics.quiz_question_breakdown``).

These are plain read models assembled from query dicts in the router;
they carry no ``from_attributes`` config because the source rows are
dicts, not ORM instances.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class QuizScoreBucket(BaseModel):
    """One score-histogram band (see ``analytics._score_histogram``)."""

    label: str
    lower: int
    upper: int
    count: int


class QuizResultsSummary(BaseModel):
    """Cohort headline stats over the grading-method-reduced attempt set."""

    total_attempts: int
    unique_students: int
    mean_score: float | None
    median_score: float | None
    p25: float | None
    p75: float | None
    pass_rate: float | None
    mean_time_seconds: float | None
    histogram: list[QuizScoreBucket]


class QuizPerStudentRow(BaseModel):
    """Per-student rollup over that student's completed attempts."""

    student_id: UUID
    student_name: str | None
    best_score_percent: Decimal | None
    latest_score_percent: Decimal | None
    attempts_count: int
    passed: bool | None
    last_attempt_at: datetime | None


class QuizOptionDistribution(BaseModel):
    """Per-option chosen-count for an MCQ question."""

    option_id: UUID
    option_key: str
    option_text: str
    is_correct: bool
    chosen_count: int


class QuizQuestionBreakdown(BaseModel):
    """Per-question performance breakdown (hardest-first ordering)."""

    question_id: UUID
    prompt: str
    correct_count: int
    answered_count: int
    correctness_rate: float | None
    option_distribution: list[QuizOptionDistribution]


class QuizResultsRead(BaseModel):
    """Full teacher-facing quiz-results payload."""

    quiz_id: UUID
    quiz_title: str
    passing_score_percent: Decimal
    grading_method: Literal["highest", "average", "first", "last"]
    summary: QuizResultsSummary
    per_student: list[QuizPerStudentRow]
    per_question: list[QuizQuestionBreakdown]


__all__ = [
    "QuizOptionDistribution",
    "QuizPerStudentRow",
    "QuizQuestionBreakdown",
    "QuizResultsRead",
    "QuizResultsSummary",
    "QuizScoreBucket",
]
