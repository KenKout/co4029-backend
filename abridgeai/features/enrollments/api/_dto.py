"""Pydantic DTOs returned by :mod:`enrollments.api.public`.

ORM models (``Enrollment``, ``InvitationCode``) MUST NOT escape the
enrollments feature; cross-feature callers receive these immutable,
typed read-models instead.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _BaseDTO(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class EnrollmentDTO(_BaseDTO):
    """Single ``course_enrollments`` row read-model.

    Lifecycle is reflected via ``status`` (one of ``active``,
    ``completed``, ``dropped``, ``waitlisted``); the unique constraint
    ``(course_id, student_id)`` guarantees at most one row per pair.
    Audit columns and the legacy ``invitation_code_id`` are intentionally
    omitted from the cross-feature surface.
    """

    id: UUID
    course_id: UUID
    student_id: UUID
    status: str
    source: str
    enrolled_at: datetime
    completed_at: datetime | None
    dropped_at: datetime | None


__all__ = ["EnrollmentDTO"]
