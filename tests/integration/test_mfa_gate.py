"""Backend MFA gate tests — verifies that ``get_current_user`` enforces
the second-factor gate while ``get_current_user_pre_mfa`` does not.

Why this exists: before the gate fix, the only check on a protected
endpoint was "is the bearer token live?" — there was no validation that
the session had completed MFA. A user with a verified MFA factor could
skip the ``/login/mfa`` redirect and call any endpoint with the access
token issued by Google login, since ``mfa_verified_at`` was never
consulted. These tests pin the new behaviour:

1. With a verified MFA factor and ``mfa_verified_at IS NULL``, every
   endpoint that depends on ``get_current_user`` returns 403 with
   ``{"error": "mfa_required"}``.
2. The pre-MFA dependency lets ``/auth/me/mfa/challenge``,
   ``/auth/me/mfa/verify``, ``/auth/logout`` and ``/users/me`` through
   so the SPA can complete the second leg without bouncing.
3. Once ``mfa_verified_at`` is set (post-verify), the full gate stops
   raising.
4. A user with NO verified factor passes the gate even when
   ``mfa_verified_at IS NULL``: not enrolled = not gated.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest_asyncio
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
from abridgeai.core.security import (
    create_access_token,
    encrypt_secret,
    generate_token,
    hash_secret,
)
from abridgeai.features.identity.routers.auth import router as auth_router
from abridgeai.features.identity.routers.me import router as me_router
from abridgeai.features.identity.routers.mfa import router as mfa_router


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
    fastapi_app.include_router(mfa_router, prefix="/api/v1")
    fastapi_app.include_router(auth_router, prefix="/api/v1")
    fastapi_app.include_router(me_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def gated_session(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    """Session for a user with a verified MFA factor but no MFA on session.

    Mirrors the post-Google-login state for a user who has 2FA enabled:
    valid bearer + live ``auth_sessions`` row + at least one verified
    ``mfa_factors`` row, but ``mfa_verified_at IS NULL``. This is the
    state the gate must reject for protected endpoints.
    """
    session_id = uuid.uuid4()
    factor_id = uuid.uuid4()
    refresh_hash = hash_secret(generate_token())
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)

    async with test_engine_local.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO mfa_factors (id, user_id, factor_type, "
                "secret_encrypted, verified_at) "
                "VALUES (:fid, :uid, 'totp', :secret, NOW())"
            ),
            {
                "fid": factor_id,
                "uid": seeded_users.student_id,
                "secret": encrypt_secret("JBSWY3DPEHPK3PXP"),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at, mfa_verified_at) "
                "VALUES (:id, :uid, :h, :exp, NULL)"
            ),
            {
                "id": session_id,
                "uid": seeded_users.student_id,
                "h": refresh_hash,
                "exp": expires_at,
            },
        )

    token = create_access_token(user_id=seeded_users.student_id, session_id=session_id)
    try:
        yield session_id, token
    finally:
        async with test_engine_local.begin() as conn:
            await conn.execute(
                text("DELETE FROM mfa_challenges WHERE user_id = :uid"),
                {"uid": seeded_users.student_id},
            )
            await conn.execute(
                text("DELETE FROM mfa_factors WHERE user_id = :uid"),
                {"uid": seeded_users.student_id},
            )
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": session_id},
            )


@pytest_asyncio.fixture
async def unenrolled_session(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    """Session for a user with NO verified MFA factor.

    The gate must NOT bite this user — they haven't opted in. This is
    the default state for everyone who hasn't enrolled 2FA yet.
    """
    session_id = uuid.uuid4()
    refresh_hash = hash_secret(generate_token())
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)

    async with test_engine_local.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at, mfa_verified_at) "
                "VALUES (:id, :uid, :h, :exp, NULL)"
            ),
            {
                "id": session_id,
                "uid": seeded_users.student_id,
                "h": refresh_hash,
                "exp": expires_at,
            },
        )

    token = create_access_token(user_id=seeded_users.student_id, session_id=session_id)
    try:
        yield session_id, token
    finally:
        async with test_engine_local.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": session_id},
            )


async def test_gate_blocks_users_me_permissions_when_mfa_pending(
    client: httpx.AsyncClient, gated_session: tuple[uuid.UUID, str]
) -> None:
    """``/users/me/permissions`` uses ``get_current_user`` (full gate).

    With a verified factor and ``mfa_verified_at IS NULL`` the response
    must be 403 with ``{"error": "mfa_required"}`` so the SPA fetch
    layer can flip ``requiresMfa`` and the route guard can redirect.
    """
    _, token = gated_session
    response = await client.get(
        "/api/v1/users/me/permissions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"] == "mfa_required"


async def test_pre_mfa_dependency_allows_users_me_get(
    client: httpx.AsyncClient, gated_session: tuple[uuid.UUID, str]
) -> None:
    """``GET /users/me`` uses ``get_current_user_pre_mfa`` so the SPA
    can render the user's name on the ``/login/mfa`` page before the
    second factor completes.
    """
    _, token = gated_session
    response = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text


async def test_pre_mfa_dependency_allows_challenge_creation(
    client: httpx.AsyncClient, gated_session: tuple[uuid.UUID, str]
) -> None:
    """``POST /auth/me/mfa/challenge`` is the second leg of login —
    it must run while ``mfa_verified_at IS NULL``. If the gate fired
    here the user would never be able to complete MFA.
    """
    _, token = gated_session
    response = await client.post(
        "/api/v1/auth/me/mfa/challenge",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    assert "challenge_id" in response.json()


async def test_pre_mfa_dependency_allows_logout(
    client: httpx.AsyncClient, gated_session: tuple[uuid.UUID, str]
) -> None:
    """``POST /auth/logout`` must work mid-MFA so the user can abandon
    the login flow without first completing the second factor.
    """
    _, token = gated_session
    response = await client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 204


async def test_gate_blocks_totp_enroll_when_mfa_pending(
    client: httpx.AsyncClient, gated_session: tuple[uuid.UUID, str]
) -> None:
    """Re-enrolling a factor mid-MFA would let an attacker pivot — the
    full gate must fire on ``/auth/me/mfa/totp/enroll``.
    """
    _, token = gated_session
    response = await client.post(
        "/api/v1/auth/me/mfa/totp/enroll",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"] == "mfa_required"


async def test_gate_lets_session_through_after_mfa_verified(
    client: httpx.AsyncClient,
    gated_session: tuple[uuid.UUID, str],
    test_engine_local: AsyncEngine,
) -> None:
    """Once the verify endpoint stamps ``mfa_verified_at``, the gate
    stops raising and protected endpoints work normally.
    """
    session_id, token = gated_session
    async with test_engine_local.begin() as conn:
        await conn.execute(
            text("UPDATE auth_sessions SET mfa_verified_at = NOW() WHERE id = :sid"),
            {"sid": session_id},
        )
    response = await client.get(
        "/api/v1/users/me/permissions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text


async def test_gate_does_not_bite_unenrolled_users(
    client: httpx.AsyncClient, unenrolled_session: tuple[uuid.UUID, str]
) -> None:
    """A user with no verified MFA factor must NOT be gated even when
    ``mfa_verified_at IS NULL``. Default-deny here would lock out
    everyone who hasn't opted into 2FA yet.
    """
    _, token = unenrolled_session
    response = await client.get(
        "/api/v1/users/me/permissions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
