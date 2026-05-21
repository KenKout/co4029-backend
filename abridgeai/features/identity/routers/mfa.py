"""Identity MFA router — TOTP enroll/verify, challenge, recovery codes.

Mirrors T1.5's auth router style: thin handlers translating service-layer
exceptions to HTTP responses. The T1.7 service layer commits its own
transactions, so this router only:

1. Resolves the authenticated principal via ``get_current_user``.
2. Loads the matching ``User`` + ``AuthSession`` ORM rows (services accept
   the model objects, not bare ids).
3. Calls into ``services.mfa`` and maps :class:`AppError` subclasses.

Recovery codes are returned in plaintext exactly twice in their lifetime:
once at TOTP-enrollment verify and once at explicit regeneration. The DB
only ever stores hashes (``MfaRecoveryCode.code_hash``); regeneration
requires a fresh challenge verification (``session.mfa_verified_at``
within the last 5 minutes) so a stolen access token alone cannot mint a
new code set.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, NotFoundError, UnauthorizedError
from abridgeai.core.security import (
    CurrentUser,
    get_current_user,
    get_current_user_pre_mfa,
    utcnow,
)
from abridgeai.features.identity.models import AuthSession, User
from abridgeai.features.identity.queries import sessions as session_queries
from abridgeai.features.identity.schemas import (
    MfaChallengeResponse,
    MfaEnrollResponse,
    MfaRecoveryCodesResponse,
    MfaStatusResponse,
    MfaTotpVerifyRequest,
    MfaVerifyRequest,
)
from abridgeai.features.identity.services import mfa as mfa_service

router = APIRouter(prefix="/auth/me/mfa", tags=["auth", "mfa"])

_MFA_FRESHNESS_WINDOW = timedelta(minutes=5)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


async def _load_principal(db: AsyncSession, current_user: CurrentUser) -> tuple[User, AuthSession]:
    """Load ORM ``User`` + ``AuthSession`` for the request principal.

    ``get_current_user`` only carries ids; the MFA service layer takes the
    full ORM rows so it can mutate them (``factor.verified_at``,
    ``session.mfa_verified_at``) within its own transaction.
    """
    user = await db.get(User, current_user.user_id)
    session = await db.get(AuthSession, current_user.session_id)
    if user is None or session is None:
        raise _unauthorized("Session not found")
    return user, session


@router.get("/status", response_model=MfaStatusResponse)
async def get_mfa_status(
    current_user: Annotated[CurrentUser, Depends(get_current_user_pre_mfa)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MfaStatusResponse:
    """Return whether the caller has enrolled MFA + current-session state.

    Powers the SPA's settings page so it can render "Two-factor enabled
    ✓" instead of always showing the Enable button. Uses the pre-MFA
    dependency so the answer is reachable from the ``/login/mfa`` flow
    too (the SPA renders the user's name there).
    """
    enrolled = await session_queries.user_has_verified_mfa(db, current_user.user_id)
    session = await db.get(AuthSession, current_user.session_id)
    return MfaStatusResponse(
        enrolled=enrolled,
        mfa_verified_at=session.mfa_verified_at if session is not None else None,
    )


@router.post("/totp/enroll", response_model=MfaEnrollResponse)
async def enroll_totp(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MfaEnrollResponse:
    """Begin TOTP enrollment.

    Returns the freshly minted TOTP secret (base32) plus the
    ``otpauth://`` provisioning URL the SPA renders as a QR code. The
    factor is created in an unverified state — clients must round-trip
    through ``/auth/me/mfa/totp/verify`` before challenge issuance is
    available.
    """
    user, _session = await _load_principal(db, current_user)
    return await mfa_service.enroll_totp(db, user)


@router.post("/totp/verify", response_model=MfaRecoveryCodesResponse)
async def verify_totp_enrollment(
    payload: MfaTotpVerifyRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MfaRecoveryCodesResponse:
    """Confirm enrollment by verifying a fresh TOTP code.

    On success: marks the factor verified, marks the current session
    MFA-verified, and issues the initial 10 plaintext recovery codes.
    These are the ONLY plaintext copies the user will ever receive — the
    DB stores hashes via ``hash_secret``.
    """
    user, session = await _load_principal(db, current_user)
    try:
        return await mfa_service.verify_totp_enrollment(db, user, session, payload)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except UnauthorizedError as exc:
        raise _unauthorized(str(exc)) from exc


@router.post("/challenge", response_model=MfaChallengeResponse)
async def create_mfa_challenge(
    current_user: Annotated[CurrentUser, Depends(get_current_user_pre_mfa)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MfaChallengeResponse:
    """Issue a 5-minute MFA challenge for the current session.

    Used during step-up flows (e.g. before sensitive actions) and as the
    second leg of a login that flagged ``requires_mfa`` in
    :class:`TokenResponse`. Uses ``get_current_user_pre_mfa`` so a fresh
    post-login session (``mfa_verified_at IS NULL``) can still mint a
    challenge — that's the whole point of this endpoint.
    """
    user, session = await _load_principal(db, current_user)
    try:
        return await mfa_service.create_mfa_challenge(db, user, session)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc


@router.post("/verify", status_code=status.HTTP_204_NO_CONTENT)
async def verify_mfa_challenge(
    payload: MfaVerifyRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user_pre_mfa)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Consume a challenge using a TOTP code or single-use recovery code.

    On success the challenge row gets ``consumed_at`` stamped and
    ``session.mfa_verified_at`` is refreshed; callers can subsequently use
    ``/auth/me/mfa/recovery-codes/regenerate`` within the freshness
    window. Pre-MFA gate so the post-login session can complete the
    second leg.
    """
    user, session = await _load_principal(db, current_user)
    try:
        await mfa_service.verify_mfa_challenge(db, user, session, payload)
    except UnauthorizedError as exc:
        raise _unauthorized(str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/recovery-codes/regenerate", response_model=MfaRecoveryCodesResponse)
async def regenerate_recovery_codes(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MfaRecoveryCodesResponse:
    """Re-issue 10 recovery codes; invalidates the previous batch.

    Requires recent (<= 5 min) MFA verification on the current session
    so a stolen access token alone cannot mint fresh codes. Caller must
    issue + verify a challenge first if their freshness window has
    lapsed.
    """
    user, session = await _load_principal(db, current_user)
    if session.mfa_verified_at is None or (
        utcnow() - session.mfa_verified_at > _MFA_FRESHNESS_WINDOW
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "mfa_verification_stale"},
        )
    try:
        return await mfa_service.regenerate_recovery_codes(db, user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


__all__ = ["router"]
