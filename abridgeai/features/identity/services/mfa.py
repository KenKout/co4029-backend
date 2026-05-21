from __future__ import annotations

import secrets
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

import pyotp

from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.exceptions import NotFoundError, UnauthorizedError
from abridgeai.core.security import (
    decrypt_secret,
    encrypt_secret,
    hash_secret,
    utcnow,
    verify_secret,
)
from abridgeai.features.identity.models import (
    AuthSession,
    MfaChallenge,
    MfaFactor,
    MfaRecoveryCode,
    User,
)
from abridgeai.features.identity.queries import mfa as mfa_queries
from abridgeai.features.identity.schemas import (
    MfaChallengeResponse,
    MfaDisableRequest,
    MfaEnrollResponse,
    MfaRecoveryCodesResponse,
    MfaTotpVerifyRequest,
    MfaVerifyRequest,
)

register_conflict_mappings(
    {
        "uq_mfa_recovery_factor_code": "mfa_recovery_code_taken: this recovery code value has already been used for the factor",  # noqa: E501
        "auth_sessions_refresh_token_hash_key": "auth_session_token_already_recorded: this refresh-token hash already exists",  # noqa: E501  # nosec B105 -- error message, not a credential
    }
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_RECOVERY_CODE_COUNT = 10


async def enroll_totp(db: AsyncSession, user: User) -> MfaEnrollResponse:
    settings = get_settings()
    secret = pyotp.random_base32()
    factor = MfaFactor(
        user_id=user.id,
        factor_type="totp",
        secret_encrypted=encrypt_secret(secret),
    )
    db.add(factor)
    await db.commit()
    await db.refresh(factor)

    otpauth_url = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user.primary_email,
        issuer_name=settings.app_name,
    )
    return MfaEnrollResponse(factor_id=factor.id, secret=secret, otpauth_url=otpauth_url)


async def verify_totp_enrollment(
    db: AsyncSession,
    user: User,
    session: AuthSession,
    payload: MfaTotpVerifyRequest,
) -> MfaRecoveryCodesResponse:
    factor = await mfa_queries.get_factor_by_id(db, payload.factor_id)
    if factor is None or factor.user_id != user.id or factor.disabled_at is not None:
        raise NotFoundError("MFA factor not found")
    if not _verify_totp(factor, payload.code):
        raise UnauthorizedError("Invalid MFA code")
    factor.verified_at = utcnow()
    session.mfa_verified_at = utcnow()
    recovery_codes = await _replace_recovery_codes(db, factor.id)
    await db.commit()
    return MfaRecoveryCodesResponse(recovery_codes=recovery_codes)


async def create_mfa_challenge(
    db: AsyncSession, user: User, session: AuthSession
) -> MfaChallengeResponse:
    factor = await mfa_queries.get_verified_totp_factor(db, user.id)
    if factor is None:
        raise NotFoundError("Verified MFA factor not found")
    challenge = MfaChallenge(
        user_id=user.id,
        factor_id=factor.id,
        session_id=session.id,
        expires_at=utcnow() + timedelta(minutes=5),
    )
    db.add(challenge)
    await db.commit()
    await db.refresh(challenge)
    return MfaChallengeResponse(challenge_id=challenge.id, expires_at=challenge.expires_at)


async def verify_mfa_challenge(
    db: AsyncSession,
    user: User,
    session: AuthSession,
    payload: MfaVerifyRequest,
) -> None:
    challenge = await mfa_queries.get_challenge_by_id(db, payload.challenge_id)
    if (
        challenge is None
        or challenge.user_id != user.id
        or challenge.session_id != session.id
        or challenge.consumed_at is not None
        or challenge.expires_at <= utcnow()
    ):
        raise UnauthorizedError("Invalid MFA challenge")
    factor = await mfa_queries.get_factor_by_id(db, challenge.factor_id)
    if factor is None or factor.disabled_at is not None:
        raise UnauthorizedError("Invalid MFA factor")

    verified = False
    if payload.code:
        verified = _verify_totp(factor, payload.code)
    if not verified and payload.recovery_code:
        verified = await _consume_recovery_code(db, factor.id, payload.recovery_code)
    if not verified:
        raise UnauthorizedError("Invalid MFA verification code")

    challenge.consumed_at = utcnow()
    session.mfa_verified_at = utcnow()
    await db.commit()


async def regenerate_recovery_codes(db: AsyncSession, user: User) -> MfaRecoveryCodesResponse:
    factor = await mfa_queries.get_verified_totp_factor(db, user.id)
    if factor is None:
        raise NotFoundError("Verified MFA factor not found")
    recovery_codes = await _replace_recovery_codes(db, factor.id)
    await db.commit()
    return MfaRecoveryCodesResponse(recovery_codes=recovery_codes)


async def disable_mfa(
    db: AsyncSession,
    user: User,
    payload: MfaDisableRequest,
) -> None:
    """Turn off MFA for ``user`` after verifying proof-of-possession.

    Step-up gate: caller must present either a current TOTP code or a
    single-use recovery code matching the active factor. On success
    every not-yet-disabled factor (verified TOTP + pending enrollments)
    is marked ``disabled_at = utcnow()`` and recovery codes for the
    active factor are wiped. The session's ``mfa_verified_at`` is left
    intact so the rest of the request can finish normally; a fresh
    login will not be MFA-gated until the user re-enrolls.
    """
    factor = await mfa_queries.get_verified_totp_factor(db, user.id)
    if factor is None:
        raise NotFoundError("Verified MFA factor not found")

    verified = False
    if payload.code:
        verified = _verify_totp(factor, payload.code)
    if not verified and payload.recovery_code:
        verified = await _consume_recovery_code(db, factor.id, payload.recovery_code)
    if not verified:
        raise UnauthorizedError("Invalid MFA verification code")

    now = utcnow()
    active_factors = await mfa_queries.list_active_factors_for_user(db, user.id)
    for active in active_factors:
        active.disabled_at = now
    await mfa_queries.delete_recovery_codes_for_factor(db, factor.id)
    await db.commit()


def _verify_totp(factor: MfaFactor, code: str) -> bool:
    secret = decrypt_secret(factor.secret_encrypted)
    return bool(pyotp.TOTP(secret).verify(code, valid_window=1))


async def _replace_recovery_codes(db: AsyncSession, factor_id: UUID) -> list[str]:
    await mfa_queries.delete_recovery_codes_for_factor(db, factor_id)
    codes = [f"{secrets.token_hex(4)}-{secrets.token_hex(4)}" for _ in range(_RECOVERY_CODE_COUNT)]
    for code in codes:
        db.add(MfaRecoveryCode(factor_id=factor_id, code_hash=hash_secret(code)))
    await flush_or_conflict(db)
    return codes


async def _consume_recovery_code(db: AsyncSession, factor_id: UUID, recovery_code: str) -> bool:
    candidates = await mfa_queries.list_unused_recovery_codes(db, factor_id)
    for candidate in candidates:
        if verify_secret(recovery_code, candidate.code_hash):
            await mfa_queries.mark_recovery_code_used(db, candidate)
            return True
    return False


__all__ = [
    "create_mfa_challenge",
    "enroll_totp",
    "regenerate_recovery_codes",
    "verify_mfa_challenge",
    "verify_totp_enrollment",
]
