from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EnrollmentAuthoring(BaseModel):
    id: UUID
    course_id: UUID
    student_id: UUID
    status: str
    source: str
    invitation_code_id: UUID | None = None
    enrolled_at: datetime
    completed_at: datetime | None = None
    dropped_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    created_by: UUID | None = None
    updated_by: UUID | None = None
    # Additive (optional so existing callers that don't join user identity
    # -- bulk-enroll / CSV-import responses -- keep working unchanged).
    # Populated by list_enrollments_for_course for the Manager roster tab,
    # which was previously rendering raw student_id UUIDs with no name.
    primary_email: str | None = None
    display_name: str | None = None

    model_config = ConfigDict(from_attributes=True)


class InvitationCodeAuthoring(BaseModel):
    id: UUID
    course_id: UUID
    organization_id: UUID
    code: str
    expires_at: datetime | None = None
    max_uses: int | None = None
    current_uses: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: UUID | None = None
    updated_by: UUID | None = None
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


__all__ = ["EnrollmentAuthoring", "InvitationCodeAuthoring"]
