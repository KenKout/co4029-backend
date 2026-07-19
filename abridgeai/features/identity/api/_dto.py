"""Pydantic v2 DTOs returned by ``features.identity.api.public``.

Cross-feature consumers (admin dashboards, courses ownership lookups,
enrollments, materials) bind to these immutable shapes; ORM models stay
private to the identity feature.

Security contract — these DTOs MUST NEVER expose:

* ``password_hash`` — the project is OAuth-only and the column does not
  exist on ``User`` today, but the test suite still asserts the field
  is absent so a future regression cannot leak it.
* ``mfa_secret`` / ``secret_encrypted`` — TOTP seeds live on
  ``MfaFactor`` and never cross the feature boundary.
* ``refresh_token_hash`` — session credential material on ``AuthSession``.
* Raw ``auth_identities`` / ``mfa_factors`` rows — these are
  authentication-internal records.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _DTOBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True, extra="ignore")


class UserDTO(_DTOBase):
    id: UUID
    primary_email: str
    display_name: str | None
    status: str
    created_at: datetime


class UserProfileDTO(_DTOBase):
    user_id: UUID
    display_name: str
    given_name: str | None
    family_name: str | None
    avatar_object_id: UUID | None
    bio: str | None
    locale: str | None


__all__ = [
    "UserDTO",
    "UserProfileDTO",
]
