"""Regrade DTOs (Phase 1).

Back the teacher regrade endpoints: dry-run compute, read-run, commit.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RegradeScopeIn(BaseModel):
    """Body for the dry-run endpoint. None/[] = whole quiz."""

    model_config = ConfigDict(extra="forbid")

    attempt_ids: list[UUID] | None = None
    question_ids: list[UUID] | None = None


class RegradeItemRead(BaseModel):
    """One changed answer in a regrade run."""

    model_config = ConfigDict(from_attributes=True)

    attempt_id: UUID
    question_id: UUID
    old_is_correct: bool
    new_is_correct: bool
    old_points: Decimal
    new_points: Decimal


class RegradeRunRead(BaseModel):
    """A regrade run + its per-answer deltas."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    status: str
    attempts_affected: int
    answers_changed: int
    created_at: datetime
    committed_at: datetime | None = None
    items: list[RegradeItemRead] = []


__all__ = ["RegradeItemRead", "RegradeRunRead", "RegradeScopeIn"]
