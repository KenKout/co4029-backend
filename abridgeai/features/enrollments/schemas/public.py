from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EnrollmentRead(BaseModel):
    course_id: UUID
    status: Literal["active", "completed", "dropped", "waitlisted"]
    enrolled_at: datetime
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


__all__ = ["EnrollmentRead"]
