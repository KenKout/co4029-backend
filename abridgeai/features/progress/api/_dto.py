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

    Wraps the result of ``progress/queries/sql/at_risk_students.sql``.
    A row is "at risk" when last engagement is NULL, older than 7 days,
    or the rolling completion percent is below 30.
    """

    user_id: UUID
    last_engagement_at: datetime | None
    completion_percent: Decimal
    days_since_last_engagement: float | None


__all__ = [
    "AtRiskStudentDTO",
    "LessonProgressDTO",
]
