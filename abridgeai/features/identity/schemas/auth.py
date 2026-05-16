from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .profile import UserRead


class GoogleLoginResponse(BaseModel):
    authorization_url: str
    state: str


class TokenResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 — OAuth2 scheme literal, not a credential
    expires_in: int
    requires_mfa: bool = False
    user: UserRead


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class MfaEnrollResponse(BaseModel):
    factor_id: UUID
    secret: str
    otpauth_url: str


class MfaTotpVerifyRequest(BaseModel):
    factor_id: UUID
    code: str = Field(min_length=6, max_length=8)


class MfaRecoveryCodesResponse(BaseModel):
    recovery_codes: list[str]


class MfaChallengeResponse(BaseModel):
    challenge_id: UUID
    expires_at: datetime


class MfaVerifyRequest(BaseModel):
    challenge_id: UUID
    code: str | None = Field(default=None, min_length=6, max_length=8)
    recovery_code: str | None = None
