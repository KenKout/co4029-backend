"""Pydantic DTOs returned by :mod:`progress.api.public`.

ORM models (``LessonProgress``, ``MaterialEngagement``) MUST NOT escape
the progress feature; cross-feature callers receive these immutable,
typed read-models instead.

All DTOs are ``model_config = ConfigDict(frozen=True)`` so consumers
cannot mutate cached results.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _BaseDTO(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class LessonProgressDTO(_BaseDTO):
    """Per-(student, lesson) progress read-model.

    Mirrors the projected columns of ``lesson_progress`` consumed cross
    feature. Excludes audit columns (``created_by``/``updated_by``):
    callers must not need them.
    """

    id: UUID
    user_id: UUID
    lesson_id: UUID
    status: str
    completion_percent: Decimal
    last_activity_at: datetime | None
    total_time_seconds: int


class AtRiskStudentDTO(_BaseDTO):
    """Aggregated at-risk row for a course's roster.

    Projects the scored output of ``progress.services.monitoring``, not the
    raw SQL: a row reaches a cross-feature caller only after the tunable
    thresholds and the new-enrolment grace period have been applied.

    ``primary_reason`` is the highest-severity reason's human-readable
    detail (it names the threshold that fired); ``signal_count`` is how many
    reasons fired in total, so a caller can render "inactive 12 days
    (threshold: 7) +1 more" without receiving the whole list.
    """

    user_id: UUID
    completion_percent: Decimal
    days_since_last_engagement: float | None
    primary_reason: str | None
    signal_count: int


class StudentNeedingAttentionDTO(_BaseDTO):
    """One (student, course) risk row for a cross-feature caller.

    Carries ``course_id`` because this projection spans a teacher's whole
    course set; the same student may appear once per course they are
    struggling in. Identity (name, email, avatar) is deliberately absent --
    progress does not own it, and the caller composes it from the identity
    feature's own public API.
    """

    user_id: UUID
    course_id: UUID
    completion_percent: Decimal
    last_engagement_at: datetime | None
    days_since_last_engagement: int | None
    #: Human-readable detail of the highest-severity reason; names the
    #: threshold that fired (FR-026).
    primary_reason: str
    signal_count: int
    #: "high" (absent) or "medium" (present but behind).
    severity: str


class CourseHealthSignalsDTO(_BaseDTO):
    """Progress-owned health signals for one course.

    ``at_risk_students`` counts DISTINCT students in this course with at
    least one active signal, so it is directly comparable with
    ``student_count`` — "8 of 40" rather than a signal tally that could
    exceed the roster.
    """

    course_id: UUID
    student_count: int
    avg_completion_percent: float
    at_risk_students: int


__all__ = [
    "AtRiskStudentDTO",
    "CourseHealthSignalsDTO",
    "StudentNeedingAttentionDTO",
    "LessonProgressDTO",
]
