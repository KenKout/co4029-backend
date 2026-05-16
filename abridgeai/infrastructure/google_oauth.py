"""Google OAuth 2.0 authorization-code flow helpers.

``build_authorization_url`` constructs the redirect URL the SPA sends the
user to. ``fetch_google_profile`` exchanges the returned ``code`` for an
access token, then loads the OIDC userinfo. Both raise :class:`AppError`
when the integration is not configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from abridgeai.core.config import Settings, get_settings
from abridgeai.core.exceptions import AppError


@dataclass(frozen=True)
class GoogleProfile:
    subject: str
    email: str
    given_name: str | None
    family_name: str | None
    display_name: str


def build_authorization_url(state: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not settings.google_client_id or not settings.google_redirect_uri:
        raise AppError("Google OAuth is not configured")

    query = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account",
            "state": state,
        }
    )
    return f"https://accounts.google.com/o/oauth2/v2/auth?{query}"


async def fetch_google_profile(code: str, settings: Settings | None = None) -> GoogleProfile:
    settings = settings or get_settings()
    if (
        not settings.google_client_id
        or not settings.google_client_secret
        or not settings.google_redirect_uri
    ):
        raise AppError("Google OAuth is not configured")

    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.google_redirect_uri,
            },
        )
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]
        userinfo_response = await client.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo_response.raise_for_status()

    payload = userinfo_response.json()
    email = payload.get("email")
    subject = payload.get("sub")
    if not email or not subject:
        raise AppError("Google profile response did not include an email or subject")
    return GoogleProfile(
        subject=subject,
        email=email,
        given_name=payload.get("given_name"),
        family_name=payload.get("family_name"),
        display_name=payload.get("name") or email.split("@", 1)[0],
    )


__all__ = ["GoogleProfile", "build_authorization_url", "fetch_google_profile"]
