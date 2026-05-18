"""Pydantic v2 DTOs returned by ``features.spaced_repetition.api.public``.

ORM models stay private to the feature; cross-feature consumers
(progress dashboards, admin) bind to these immutable shapes. Soft-
deleted rows are filtered upstream by ``core/db/soft_delete.py`` --
audit columns are not exposed.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _DTOBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True, extra="ignore")


class CardStateDTO(_DTOBase):
    """Per-(student, question) SM-2 scheduling state.

    Mirrors the public scheduling fields of ``StudentCardState``.
    Audit / soft-delete columns are deliberately omitted.
    """

    student_id: UUID
    question_id: UUID
    ef: Decimal
    interval_days: int
    repetition_count: int
    due_at: datetime
    last_q: int | None
    last_reviewed_at: datetime | None
    calibration_active: bool
    total_reviews: int


__all__ = ["CardStateDTO"]
