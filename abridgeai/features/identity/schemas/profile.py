from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserProfileRead(_ORMModel):
    user_id: UUID
    given_name: str | None = None
    family_name: str | None = None
    display_name: str
    avatar_object_id: UUID | None = None
    # Presigned GET URL for the avatar image, minted server-side on each read
    # (short TTL). ``None`` when no avatar is set. Not persisted — a projection.
    avatar_url: str | None = None
    bio: str | None = None
    locale: str | None = None


class UserRead(_ORMModel):
    id: UUID
    primary_email: str
    status: str
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    profile: UserProfileRead | None = None


class UserProfileUpdate(BaseModel):
    given_name: str | None = Field(default=None, max_length=100)
    family_name: str | None = Field(default=None, max_length=100)
    display_name: str | None = Field(default=None, max_length=200)
    bio: str | None = None
    # Preferred UI + notification language. Constrained to the frontend
    # SUPPORTED_LOCALES; the DB CHECK constraint (migration 0024) is the
    # backstop. ``None`` on PATCH means "leave unchanged" (exclude_unset).
    locale: Literal["en", "vi"] | None = None


class UserProfileLinkIn(BaseModel):
    link_type: str = Field(max_length=30)
    url: str
    label: str | None = Field(default=None, max_length=100)


class UserProfileLinkRead(_ORMModel):
    id: UUID
    user_id: UUID
    link_type: str
    url: str
    label: str | None = None
    created_at: datetime
    updated_at: datetime


class AuthSessionRead(_ORMModel):
    id: UUID
    expires_at: datetime
    revoked_at: datetime | None = None
    mfa_verified_at: datetime | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: datetime
    updated_at: datetime


class UserPermissionsRead(BaseModel):
    """Effective permission codes the current user holds (T1.9 ``/users/me/permissions``)."""

    permissions: list[str]


class UserListPage(BaseModel):
    """Cursor-paginated list of users (T1.9 ``GET /users``)."""

    items: list[UserRead]
    next_cursor: str | None = None
