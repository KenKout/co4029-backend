from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import ForbiddenError, NotFoundError
from abridgeai.core.security import (
    create_access_token,
    generate_token,
    hash_secret,
    utcnow,
)
from abridgeai.features.identity.models import (
    AuthIdentity,
    AuthSession,
    User,
    UserProfile,
)
from abridgeai.features.identity.queries import sessions as session_queries
from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import TokenResponse
from abridgeai.infrastructure.google_oauth import fetch_google_profile

from .profile import serialize_user

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def handle_google_callback(
    db: AsyncSession,
    *,
    code: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> TokenResponse:
    """Exchange a Google OAuth code for an application token pair.

    Pre-registration gate (added 2026-05-19): the user *must* already exist in
    ``users`` (created by an admin) AND have ``status='active'``. Brand-new
    OAuth profiles are rejected with :class:`ForbiddenError` (HTTP 403) so
    the platform stays invite-only.
    """
    profile = await fetch_google_profile(code)
    identity = await user_queries.get_identity_by_provider_subject(
        db, provider="google", provider_subject=profile.subject
    )

    if identity is None:
        user = await user_queries.get_user_by_email(db, profile.email)
        if user is None:
            raise ForbiddenError(
                "This email is not registered. Ask an administrator to add "
                "your account before signing in."
            )
        if user.status != "active":
            raise ForbiddenError(
                f"Account is {user.status}; sign-in is disabled. Contact an administrator."
            )
        if await user_queries.get_profile(db, user.id) is None:
            db.add(
                UserProfile(
                    user_id=user.id,
                    given_name=profile.given_name,
                    family_name=profile.family_name,
                    display_name=profile.display_name,
                )
            )
        db.add(
            AuthIdentity(
                user_id=user.id,
                provider="google",
                provider_subject=profile.subject,
                provider_email=profile.email.lower(),
            )
        )
        user.last_login_at = utcnow()
    else:
        user = await user_queries.get_user(db, identity.user_id)
        if user is None:
            raise NotFoundError("OAuth user not found")
        if user.status != "active":
            raise ForbiddenError(
                f"Account is {user.status}; sign-in is disabled. Contact an administrator."
            )
        user.last_login_at = utcnow()

    return await _issue_tokens(db, user=user, ip_address=ip_address, user_agent=user_agent)


async def _issue_tokens(
    db: AsyncSession,
    *,
    user: User,
    ip_address: str | None,
    user_agent: str | None,
) -> TokenResponse:
    settings = get_settings()
    refresh_token = generate_token()
    requires_mfa = await session_queries.user_has_verified_mfa(db, user.id)
    session = AuthSession(
        user_id=user.id,
        refresh_token_hash=hash_secret(refresh_token),
        expires_at=utcnow() + timedelta(seconds=settings.session_ttl_seconds),
        mfa_verified_at=None if requires_mfa else utcnow(),
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    await db.refresh(user)
    profile = await user_queries.get_profile(db, user.id)
    return TokenResponse(
        access_token=create_access_token(user_id=user.id, session_id=session.id),
        refresh_token=refresh_token,
        expires_in=settings.access_token_ttl_seconds,
        requires_mfa=requires_mfa,
        user=serialize_user(user, profile),
    )


__all__ = ["handle_google_callback"]
