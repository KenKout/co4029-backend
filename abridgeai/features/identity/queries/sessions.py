"""Auth session queries.

The active-session predicate filters on ``revoked_at IS NULL`` only; expiry
policy (``expires_at`` comparison, grace windows, etc.) is the caller's
decision and lives in the service layer (T1.7).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.identity.models import AuthSession, MfaFactor


async def get_session_by_refresh_hash(db: AsyncSession, refresh_hash: str) -> AuthSession | None:
    """Fetch a non-revoked session by its refresh-token hash.

    Returns ``None`` for either "no session" or "session was revoked"; the
    service layer treats both as identical (re-authenticate).
    """
    result = await db.execute(
        select(AuthSession).where(
            AuthSession.refresh_token_hash == refresh_hash,
            AuthSession.revoked_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def user_has_verified_mfa(db: AsyncSession, user_id: UUID) -> bool:
    """True iff the user has at least one currently-active MFA factor.

    "Active" means ``verified_at IS NOT NULL AND disabled_at IS NULL`` — the
    factor was enrolled, the user proved possession at enrollment time, and
    it has not been turned off since.
    """
    result = await db.execute(
        select(MfaFactor.id).where(
            MfaFactor.user_id == user_id,
            MfaFactor.verified_at.is_not(None),
            MfaFactor.disabled_at.is_(None),
        )
    )
    return result.first() is not None
