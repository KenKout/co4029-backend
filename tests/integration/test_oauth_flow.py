from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import generate_token, hash_secret
from abridgeai.features.identity.routers.auth import router as auth_router

ROUTER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "abridgeai"
    / "features"
    / "identity"
    / "routers"
    / "auth.py"
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def test_engine_local() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def app(test_engine_local: AsyncEngine) -> AsyncIterator[FastAPI]:
    sm = async_sessionmaker(test_engine_local, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with sm() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(auth_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def google_oauth_settings(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_google_login_returns_auth_url(
    client: httpx.AsyncClient, google_oauth_settings: None
) -> None:
    response = await client.get("/api/v1/auth/google/login")
    assert response.status_code == 200
    body = response.json()
    assert "accounts.google.com" in body["authorization_url"]
    assert len(body["state"]) >= 16


async def test_oauth_callback_creates_user(
    client: httpx.AsyncClient,
    test_engine_local: AsyncEngine,
    google_oauth_settings: None,
) -> None:
    fresh_email = f"oauth-new-{uuid.uuid4().hex[:8]}@abridgeai.local"
    google_subject = f"google-uid-{uuid.uuid4().hex[:12]}"

    with respx.mock(assert_all_called=True) as router_mock:
        router_mock.post("https://oauth2.googleapis.com/token").mock(
            return_value=httpx.Response(
                200,
                json={
                    "access_token": "ya29.fake",
                    "id_token": "fake.jwt",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        )
        router_mock.get("https://openidconnect.googleapis.com/v1/userinfo").mock(
            return_value=httpx.Response(
                200,
                json={
                    "sub": google_subject,
                    "email": fresh_email,
                    "given_name": "Oauth",
                    "family_name": "New",
                    "name": "Oauth New",
                },
            )
        )

        response = await client.get("/api/v1/auth/google/callback", params={"code": "fakeAuthCode"})

    try:
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["access_token"]
        assert body["refresh_token"]
        assert body["token_type"] == "bearer"  # noqa: S105 — OAuth2 scheme literal
        assert body["user"]["primary_email"] == fresh_email

        async with test_engine_local.begin() as conn:
            user_row = (
                await conn.execute(
                    text("SELECT id FROM users WHERE primary_email = :email"),
                    {"email": fresh_email},
                )
            ).one_or_none()
            assert user_row is not None
            session_row = (
                await conn.execute(
                    text("SELECT refresh_token_hash FROM auth_sessions WHERE user_id = :uid"),
                    {"uid": user_row[0]},
                )
            ).one_or_none()
            assert session_row is not None
            assert session_row[0]
    finally:
        async with test_engine_local.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM auth_sessions WHERE user_id IN "
                    "(SELECT id FROM users WHERE primary_email = :email)"
                ),
                {"email": fresh_email},
            )
            await conn.execute(
                text("DELETE FROM auth_identities WHERE provider_subject = :sub"),
                {"sub": google_subject},
            )
            await conn.execute(
                text(
                    "DELETE FROM user_profiles WHERE user_id IN "
                    "(SELECT id FROM users WHERE primary_email = :email)"
                ),
                {"email": fresh_email},
            )
            await conn.execute(
                text("DELETE FROM users WHERE primary_email = :email"),
                {"email": fresh_email},
            )


async def test_refresh_revoked_session_fails(
    client: httpx.AsyncClient,
    test_engine_local: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    refresh_token = generate_token()
    refresh_hash = hash_secret(refresh_token)
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)

    async with test_engine_local.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at, revoked_at) "
                "VALUES (:id, :uid, :h, :exp, NOW())"
            ),
            {
                "id": session_id,
                "uid": seeded_users.student_id,
                "h": refresh_hash,
                "exp": expires_at,
            },
        )

    try:
        response = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]
    finally:
        async with test_engine_local.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": session_id},
            )


def test_no_password_endpoints_exist() -> None:
    source = ROUTER_PATH.read_text()
    code_only = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    code_only = re.sub(r"#.*", "", code_only)
    forbidden = re.compile(
        r"\b(register|forgot[_-]?password|password[_-]?hash|password[_-]?reset)\b",
        re.IGNORECASE,
    )
    matches = forbidden.findall(code_only)
    assert matches == [], f"Forbidden password-auth tokens in router: {matches}"
    assert "BYPASS" not in code_only
    assert "_DEV_PERMISSION" not in code_only


def test_router_metadata() -> None:
    assert auth_router.prefix == "/auth"
    assert auth_router.tags == ["auth"]
    paths = {(r.path, tuple(sorted(r.methods))) for r in auth_router.routes}  # type: ignore[attr-defined]
    assert ("/auth/google/login", ("GET",)) in paths
    assert ("/auth/google/callback", ("GET",)) in paths
    assert ("/auth/refresh", ("POST",)) in paths
    assert ("/auth/logout", ("POST",)) in paths
