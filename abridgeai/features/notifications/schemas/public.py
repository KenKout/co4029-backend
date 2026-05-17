from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class NotificationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    category: str
    entity_type: str | None
    entity_id: UUID | None
    title: str
    body: str
    scheduled_for: datetime | None
    delivered_at: datetime | None
    read_at: datetime | None
    delivery_status: str
    created_at: datetime
    updated_at: datetime


class NotificationPreferenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    category: str
    channel: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class NotificationPreferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class UnreadCount(BaseModel):
    unread: int


__all__ = [
    "NotificationPreferenceRead",
    "NotificationPreferenceUpdate",
    "NotificationRead",
    "UnreadCount",
]
