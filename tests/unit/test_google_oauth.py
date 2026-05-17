"""Unit tests for ``abridgeai.infrastructure.google_oauth``.

Verifies the authorization-URL builder produces a well-formed Google OIDC
URL containing every required query parameter (per plan §3356-3361).
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from abridgeai.core.config import Settings
from abridgeai.core.exceptions import AppError
from abridgeai.infrastructure.google_oauth import build_authorization_url


def _configured() -> Settings:
    return Settings(
        google_client_id="client-abc.apps.googleusercontent.com",
        google_client_secret="secret-xyz",  # noqa: S106  # test fixture, not a real credential
        google_redirect_uri="https://app.abridgeai.test/auth/google/callback",
    )


def test_authorization_url_format() -> None:
    settings = _configured()
    url = build_authorization_url(state="state-token-1234567890", settings=settings)
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.hostname == "accounts.google.com"
    assert parsed.path == "/o/oauth2/v2/auth"

    params = parse_qs(parsed.query)
    assert params["client_id"] == ["client-abc.apps.googleusercontent.com"]
    assert params["redirect_uri"] == ["https://app.abridgeai.test/auth/google/callback"]
    assert params["response_type"] == ["code"]
    assert params["scope"] == ["openid email profile"]
    assert params["access_type"] == ["offline"]
    assert params["prompt"] == ["select_account"]
    assert params["state"] == ["state-token-1234567890"]


def test_authorization_url_raises_when_unconfigured() -> None:
    settings = Settings(google_client_id=None, google_redirect_uri=None)
    with pytest.raises(AppError):
        build_authorization_url(state="x", settings=settings)
