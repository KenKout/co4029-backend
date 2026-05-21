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


class MfaStatusResponse(BaseModel):
    """Surface the user's MFA enrollment + current-session MFA state.

    ``enrolled`` is True when the user has at least one active TOTP
    factor (``verified_at IS NOT NULL AND disabled_at IS NULL``).
    ``mfa_verified_at`` is the current ``auth_sessions.mfa_verified_at``
    so the SPA can show a "verified <ago>" badge if it wants.
    """

    enrolled: bool
    mfa_verified_at: datetime | None = None


class MfaDisableRequest(BaseModel):
    """Disable MFA on the caller's account.

    Requires proof-of-possession to prevent a stolen access token from
    silently turning off the second factor: the caller must present
    either a current TOTP code or a single-use recovery code. The code
    is verified against the user's currently-active TOTP factor; on
    success that factor (and any pending unverified ones) is marked
    ``disabled_at`` and recovery codes are wiped.
    """

    code: str | None = Field(default=None, min_length=6, max_length=8)
    recovery_code: str | None = None
