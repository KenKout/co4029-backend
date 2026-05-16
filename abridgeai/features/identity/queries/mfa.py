"""MFA queries (factor + challenge).

The canonical T1.1 model uses ``MfaFactor.verified_at`` (not ``enabled_at``)
and ``MfaChallenge.consumed_at`` (not ``verified_at``). These accessors stick
with that vocabulary; renaming is out of scope for the queries layer.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from abridgeai.features.identity.models import MfaChallenge, MfaFactor


async def get_verified_totp_factor(db: AsyncSession, user_id: UUID) -> MfaFactor | None:
    """Return the user's active TOTP factor, if any.

    Active means ``factor_type='totp' AND verified_at IS NOT NULL AND
    disabled_at IS NULL``. Most recent first — a user *should* only ever have
    one active factor, but ordering by ``created_at DESC`` keeps us
    deterministic if the data drifts.
    """
    result = await db.execute(
        select(MfaFactor)
        .where(
            MfaFactor.user_id == user_id,
            MfaFactor.factor_type == "totp",
            MfaFactor.verified_at.is_not(None),
            MfaFactor.disabled_at.is_(None),
        )
        .order_by(MfaFactor.created_at.desc())
    )
    return result.scalars().first()


async def get_active_challenge(
    db: AsyncSession, user_id: UUID, session_id: UUID
) -> MfaChallenge | None:
    """Return an unconsumed, non-expired challenge for ``(user_id, session_id)``.

    Active means ``consumed_at IS NULL AND expires_at > NOW()``. Most recent
    first so re-issued challenges shadow stale ones.
    """
    result = await db.execute(
        select(MfaChallenge)
        .where(
            MfaChallenge.user_id == user_id,
            MfaChallenge.session_id == session_id,
            MfaChallenge.consumed_at.is_(None),
            MfaChallenge.expires_at > func.now(),
        )
        .order_by(MfaChallenge.created_at.desc())
    )
    return result.scalars().first()
