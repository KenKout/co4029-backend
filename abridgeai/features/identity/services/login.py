from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.exceptions import ForbiddenError, NotFoundError
from abridgeai.core.observability import get_logger
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

logger = get_logger(__name__)

register_conflict_mappings(
    {
        "uq_auth_identity_provider_subject": "auth_identity_provider_taken: this provider account is already linked to another user",  # noqa: E501
        "auth_sessions_refresh_token_hash_key": "auth_session_token_already_recorded: this refresh-token hash already exists",  # noqa: E501  # nosec B105 -- error message, not a credential
        "users_primary_email_key": "user_email_taken: this email is already registered",
    }
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    # Injected by the auth router from access_control.api.public — the
    # identity service layer must stay feature-local (enforced by
    # tests/unit/test_identity_services.py source-grep), so cross-feature
    # reach happens via these callables, not imports.
    AutoProvisionOrgResolver = Callable[["AsyncSession", str], Awaitable["UUID | None"]]
    DefaultAccessGranter = Callable[["AsyncSession", "UUID", "UUID"], Awaitable[None]]


async def handle_google_callback(
    db: AsyncSession,
    *,
    code: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
    resolve_auto_provision_org: AutoProvisionOrgResolver | None = None,
    grant_default_access: DefaultAccessGranter | None = None,
) -> TokenResponse:
    """Exchange a Google OAuth code for an application token pair.

    Pre-registration gate (added 2026-05-19): the user *must* already exist in
    ``users`` (created by an admin) AND have ``status='active'``. Brand-new
    OAuth profiles are rejected with :class:`ForbiddenError` (HTTP 403) so
    the platform stays invite-only — with ONE exception (FR-2.7/FR-2.9):
    when the Google-verified email's domain is registered in
    ``organization_domains`` with ``auto_provision = TRUE`` for an active
    organization, the account is created on the spot with an org
    membership and the least-privilege ``student`` role.
    """
    profile = await fetch_google_profile(code)
    identity = await user_queries.get_identity_by_provider_subject(
        db, provider="google", provider_subject=profile.subject
    )

    if identity is None:
        user = await user_queries.get_user_by_email(db, profile.email)
        if user is None and profile.email_verified:
            # FR-2.7 gate: never mint accounts from an unverified OIDC email.
            user = await _auto_provision_user(
                db,
                email=profile.email,
                resolve_auto_provision_org=resolve_auto_provision_org,
                grant_default_access=grant_default_access,
            )
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


async def _auto_provision_user(
    db: AsyncSession,
    *,
    email: str,
    resolve_auto_provision_org: AutoProvisionOrgResolver | None,
    grant_default_access: DefaultAccessGranter | None,
) -> User | None:
    """Create a user for a registered auto-provision email domain (FR-2.7).

    Exact CITEXT match on the Google-verified email's domain — no
    subdomain wildcards, so a spoof-adjacent domain never matches. The
    injected access_control callables resolve the org and grant the
    least-privilege defaults (active membership + org-scoped ``student``
    role). Returns ``None`` when the feature is unwired or no
    auto-provision domain matches, so the caller keeps the invite-only
    rejection.
    """
    if resolve_auto_provision_org is None or grant_default_access is None:
        return None
    domain = email.rsplit("@", 1)[-1].lower()
    org_id = await resolve_auto_provision_org(db, domain)
    if org_id is None:
        return None
    user = User(primary_email=email.lower(), status="active")
    db.add(user)
    await flush_or_conflict(db)
    await grant_default_access(db, user.id, org_id)
    logger.info(
        "identity.user_auto_provisioned",
        user_id=str(user.id),
        organization_id=str(org_id),
        email_domain=domain,
    )
    return user


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
    await flush_or_conflict(db)
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
