"""HOD/Manager teacher-assignment DTOs for the courses aggregate.

These schemas back ``routers/assignment.py`` (T3.8) and represent the
staffing-surface projections: teacher assignments, roster entries, and
the assign-teacher request body.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TeacherAssignmentRead(BaseModel):
    """Authoring DTO for a teacher-on-course assignment.

    ``primary_email`` / ``display_name`` are joined from ``users`` +
    ``user_profiles``; ``active_until`` is non-null for soft-revoked
    rows (audit trail).
    """

    user_id: UUID
    display_name: str
    primary_email: str
    assignment_id: UUID | None = None
    active_from: datetime | None = None
    active_until: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AssignTeacherRequest(BaseModel):
    user_id: UUID

    model_config = ConfigDict(extra="forbid")


class TeacherAssignmentCreated(BaseModel):
    """Response payload for POST -- the new (or pre-existing) assignment row."""

    id: UUID
    course_id: UUID
    user_id: UUID
    role_code: str
    scope_kind: str
    organization_id: UUID
    granted_by: UUID

    model_config = ConfigDict(from_attributes=True)


class RosterEntry(BaseModel):
    """Single row in the roster response (shared by authoring + assignment routers).

    Phase 7 will replace this with the canonical ``EnrollmentRead``
    schema once the enrollments aggregate lands; the raw-SQL read here
    is the documented bridge per plan §4461.
    """

    enrollment_id: UUID
    student_id: UUID
    display_name: str | None = None
    primary_email: str
    status: str
    enrolled_at: datetime
    completed_at: datetime | None = None
    dropped_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "AssignTeacherRequest",
    "RosterEntry",
    "TeacherAssignmentCreated",
    "TeacherAssignmentRead",
]
