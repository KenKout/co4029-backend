"""Pydantic response models for SR dashboard endpoints (T7.5.12)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

LessonStatus = Literal["locked", "learning", "mature"]


class CardsDueItem(BaseModel):
    question_id: UUID
    quiz_id: UUID
    lesson_id: UUID
    lesson_title: str
    course_slug: str
    due_at: datetime
    last_q: int | None = None
    ef: float


class CardsDuePage(BaseModel):
    items: list[CardsDueItem]
    next_cursor: str | None = None


class StudentLessonSummaryRead(BaseModel):
    kr_estimate: float
    progression_ready: bool
    compliance_rate: float | None = None
    cards_total: int
    cards_due_now: int


class LessonOverviewItem(BaseModel):
    lesson_id: UUID
    lesson_title: str
    status: LessonStatus
    kr_estimate: float
    due_count: int
    # Raw unlock-gate verdict (FR-4.5). ``status`` folds eligibility AND
    # engagement into one display value ("locked" also means "no progress
    # yet"); consumers gating ACCESS must use this field, not ``status``.
    eligible: bool = True


class HistogramBucket(BaseModel):
    bucket_lower: float
    count: int


class ClassKRDistributionRead(BaseModel):
    lesson_id: UUID
    student_count: int
    histogram: list[HistogramBucket] = Field(default_factory=list)
    mean_kr: float
    median_kr: float


class DifficultCardRead(BaseModel):
    question_id: UUID
    quiz_id: UUID
    prompt_text: str
    mean_ef: float
    student_count: int


class CardStudentResultRead(BaseModel):
    student_id: UUID
    name: str
    ef: float
    total_reviews: int
    last_reviewed_at: datetime | None = None
    last_correct: bool | None = None
    correct_count: int
    review_count: int


class AtRiskStudentRead(BaseModel):
    student_id: UUID
    name: str
    low_compliance: bool
    frozen_kr: bool
    high_theory_practice_gap: bool
    last_active_at: datetime | None = None


class StudentSrDetailReviewRead(BaseModel):
    question_id: UUID
    prompt_text: str = ""
    created_at: datetime
    q_derived: int
    ef_after: float
    correct: bool


class StudentSrDetailLessonRead(BaseModel):
    lesson_id: UUID
    lesson_title: str
    kr_estimate: float
    cards_total: int
    cards_due_now: int
    status: LessonStatus


class StudentSrDetailRead(BaseModel):
    student_id: UUID
    name: str
    lessons: list[StudentSrDetailLessonRead] = Field(default_factory=list)
    recent_reviews: list[StudentSrDetailReviewRead] = Field(default_factory=list)


class StudentDashboardSummaryRead(BaseModel):
    """Cross-course SR rollup for the student dashboard landing tiles.

    The per-lesson and per-course SR endpoints already existed, but the student
    dashboard had no aggregate to show, so its Quizzes/Interviews tiles were
    hardcoded to "—". These are the headline numbers the thesis defines for the
    learner: R-hat retention, lesson maturity, and how close the next locked
    lesson is to opening.

    Every value is scoped to the calling student's ACTIVE enrolments.
    """

    #: Mean R-hat across accessible (unlocked) lessons, 0-1.
    avg_kr_estimate: float = 0.0
    #: False when there are no accessible lessons to average, so the UI can show
    #: "no data yet" instead of a misleading 0% retention. A student whose every
    #: lesson is still locked has no retention to report — that is not 0%.
    has_retention_data: bool = False
    lessons_mature: int = 0
    lessons_learning: int = 0
    lessons_locked: int = 0
    lessons_total: int = 0
    #: Uses the same joins/filters as ``GET /me/cards-due`` so the dashboard and
    #: that page can never disagree.
    cards_due_now: int = 0
    cards_total: int = 0
    #: Nearest locked lesson and how far the EF gate has progressed toward
    #: tau_unlock, from LessonUnlockStatus.current_ratio / required_ratio.
    next_unlock_lesson_id: UUID | None = None
    next_unlock_lesson_title: str | None = None
    next_unlock_progress_pct: float = 0.0


__all__ = [
    "AtRiskStudentRead",
    "CardStudentResultRead",
    "CardsDueItem",
    "CardsDuePage",
    "ClassKRDistributionRead",
    "DifficultCardRead",
    "HistogramBucket",
    "LessonOverviewItem",
    "LessonStatus",
    "StudentDashboardSummaryRead",
    "StudentLessonSummaryRead",
    "StudentSrDetailLessonRead",
    "StudentSrDetailRead",
    "StudentSrDetailReviewRead",
]
