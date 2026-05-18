"""Pydantic DTOs returned by :mod:`interviews.api.public`.

ORM models (``InterviewSession``, ``InterviewOutcomeEvaluation``,
``GapReport``) MUST NOT escape the interviews feature; cross-feature
callers receive these immutable, typed read-models instead.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _BaseDTO(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class SessionSummaryDTO(_BaseDTO):
    """Read-model summarising a single ``interview_sessions`` row.

    Outcome counts (``outcomes_total``, ``outcomes_met``) are populated
    by the public API call, not stored on the ORM row -- they roll up
    ``interview_outcome_evaluations`` for the session. Sensitive
    rubric reasoning (``hidden_reasoning``) is intentionally excluded.
    """

    id: UUID
    interview_config_id: UUID
    student_id: UUID
    attempt_number: int
    status: str
    input_mode: str
    started_at: datetime
    ended_at: datetime | None
    pass_verdict: bool | None
    outcomes_total: int
    outcomes_met: int


__all__ = ["SessionSummaryDTO"]
