from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import UnauthorizedError
from abridgeai.core.security import (
    create_access_token,
    generate_token,
    hash_secret,
    utcnow,
)
from abridgeai.features.identity.models import AuthSession
from abridgeai.features.identity.queries import sessions as session_queries
from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import TokenResponse

from .profile import serialize_user

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def refresh_tokens(db: AsyncSession, refresh_token: str) -> TokenResponse:
    settings = get_settings()
    session = await session_queries.get_session_by_refresh_hash(db, hash_secret(refresh_token))
    if session is None or session.expires_at <= utcnow():
        raise UnauthorizedError("Invalid refresh token")
    user = await user_queries.get_user(db, session.user_id)
    if user is None or user.status != "active":
        raise UnauthorizedError("User is not active")

    new_refresh_token = generate_token()
    session.refresh_token_hash = hash_secret(new_refresh_token)
    session.expires_at = utcnow() + timedelta(seconds=settings.session_ttl_seconds)
    await db.commit()
    profile = await user_queries.get_profile(db, user.id)
    requires_mfa = (
        await session_queries.user_has_verified_mfa(db, user.id) and session.mfa_verified_at is None
    )
    return TokenResponse(
        access_token=create_access_token(user_id=user.id, session_id=session.id),
        refresh_token=new_refresh_token,
        expires_in=settings.access_token_ttl_seconds,
        requires_mfa=requires_mfa,
        user=serialize_user(user, profile),
    )


async def logout(
    db: AsyncSession,
    *,
    refresh_token: str | None = None,
    session_id: UUID | None = None,
) -> None:
    """Revoke a session by refresh-token hash or by session id (caller's choice).

    Either argument resolves to an :class:`AuthSession`. Idempotent: revoking
    an already-revoked session is a no-op.
    """
    target: AuthSession | None = None
    if refresh_token is not None:
        target = await session_queries.get_session_by_refresh_hash(db, hash_secret(refresh_token))
    elif session_id is not None:
        target = await db.get(AuthSession, session_id)

    if target is not None and target.revoked_at is None:
        target.revoked_at = utcnow()
        await db.commit()


__all__ = ["logout", "refresh_tokens"]
