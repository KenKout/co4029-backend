from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pyotp
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
from abridgeai.core.security import create_access_token, generate_token, hash_secret
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
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_session(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    """Insert a live auth_sessions row for the student fixture user.

    Returns ``(session_id, bearer_token)``. A real session row is required
    because ``get_current_user`` joins ``users`` ⨝ ``auth_sessions`` and
    rejects unknown ids with 401.
    """
    session_id = uuid.uuid4()
    refresh_hash = hash_secret(generate_token())
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)

    async with test_engine_local.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
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
                text(
                    "DELETE FROM mfa_recovery_codes WHERE factor_id IN "
                    "(SELECT id FROM mfa_factors WHERE user_id = :uid)"
                ),
                {"uid": seeded_users.student_id},
            )
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


def test_router_metadata() -> None:
    assert mfa_router.prefix == "/auth/me/mfa"
    assert "auth" in mfa_router.tags
    assert "mfa" in mfa_router.tags
    paths = {(r.path, tuple(sorted(r.methods))) for r in mfa_router.routes}  # type: ignore[attr-defined]
    assert ("/auth/me/mfa/totp/enroll", ("POST",)) in paths
    assert ("/auth/me/mfa/totp/verify", ("POST",)) in paths
    assert ("/auth/me/mfa/challenge", ("POST",)) in paths
    assert ("/auth/me/mfa/verify", ("POST",)) in paths
    assert ("/auth/me/mfa/recovery-codes/regenerate", ("POST",)) in paths


async def test_unauthenticated_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/auth/me/mfa/totp/enroll")
    assert response.status_code == 401


async def test_totp_enroll_returns_otpauth_url(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
    test_engine_local: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    _, token = auth_session
    response = await client.post(
        "/api/v1/auth/me/mfa/totp/enroll",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["otpauth_url"].startswith("otpauth://totp/")
    assert "secret=" in body["otpauth_url"]
    assert body["secret"]
    assert uuid.UUID(body["factor_id"])

    async with test_engine_local.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT secret_encrypted, verified_at FROM mfa_factors WHERE user_id = :uid"),
                {"uid": seeded_users.student_id},
            )
        ).one()
        assert row[0] != body["secret"], "secret stored in plaintext"
        assert row[1] is None, "factor must be unverified at enroll time"


async def test_totp_verify_completes_enrollment_and_issues_codes(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
    test_engine_local: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    _, token = auth_session
    headers = {"Authorization": f"Bearer {token}"}

    enroll = (await client.post("/api/v1/auth/me/mfa/totp/enroll", headers=headers)).json()
    code = pyotp.TOTP(enroll["secret"]).now()

    verify = await client.post(
        "/api/v1/auth/me/mfa/totp/verify",
        headers=headers,
        json={"factor_id": enroll["factor_id"], "code": code},
    )
    assert verify.status_code == 200, verify.text
    body = verify.json()
    assert len(body["recovery_codes"]) == 10
    assert all(isinstance(c, str) and len(c) >= 16 for c in body["recovery_codes"])

    async with test_engine_local.begin() as conn:
        verified_at = (
            await conn.execute(
                text("SELECT verified_at FROM mfa_factors WHERE user_id = :uid"),
                {"uid": seeded_users.student_id},
            )
        ).scalar_one()
        assert verified_at is not None


async def test_recovery_code_is_single_use(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
) -> None:
    _, token = auth_session
    headers = {"Authorization": f"Bearer {token}"}

    enroll = (await client.post("/api/v1/auth/me/mfa/totp/enroll", headers=headers)).json()
    code = pyotp.TOTP(enroll["secret"]).now()
    recovery_codes = (
        await client.post(
            "/api/v1/auth/me/mfa/totp/verify",
            headers=headers,
            json={"factor_id": enroll["factor_id"], "code": code},
        )
    ).json()["recovery_codes"]

    challenge_one = (await client.post("/api/v1/auth/me/mfa/challenge", headers=headers)).json()
    first = await client.post(
        "/api/v1/auth/me/mfa/verify",
        headers=headers,
        json={"challenge_id": challenge_one["challenge_id"], "recovery_code": recovery_codes[0]},
    )
    assert first.status_code == 204, first.text

    challenge_two = (await client.post("/api/v1/auth/me/mfa/challenge", headers=headers)).json()
    second = await client.post(
        "/api/v1/auth/me/mfa/verify",
        headers=headers,
        json={"challenge_id": challenge_two["challenge_id"], "recovery_code": recovery_codes[0]},
    )
    assert second.status_code == 401, second.text


async def test_recovery_codes_regenerate_requires_fresh_mfa(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
    test_engine_local: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    session_id, token = auth_session
    headers = {"Authorization": f"Bearer {token}"}

    enroll = (await client.post("/api/v1/auth/me/mfa/totp/enroll", headers=headers)).json()
    code = pyotp.TOTP(enroll["secret"]).now()
    initial = (
        await client.post(
            "/api/v1/auth/me/mfa/totp/verify",
            headers=headers,
            json={"factor_id": enroll["factor_id"], "code": code},
        )
    ).json()["recovery_codes"]

    fresh = await client.post("/api/v1/auth/me/mfa/recovery-codes/regenerate", headers=headers)
    assert fresh.status_code == 200, fresh.text
    new_codes = fresh.json()["recovery_codes"]
    assert len(new_codes) == 10
    assert set(new_codes).isdisjoint(set(initial)), "old codes must be invalidated"

    async with test_engine_local.begin() as conn:
        await conn.execute(
            text(
                "UPDATE auth_sessions SET mfa_verified_at = NOW() - INTERVAL '10 minutes' "
                "WHERE id = :sid"
            ),
            {"sid": session_id},
        )

    stale = await client.post("/api/v1/auth/me/mfa/recovery-codes/regenerate", headers=headers)
    assert stale.status_code == 403
    assert stale.json()["detail"]["error"] == "mfa_verification_stale"

    async with test_engine_local.begin() as conn:
        codes_for_user = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM mfa_recovery_codes WHERE factor_id IN "
                    "(SELECT id FROM mfa_factors WHERE user_id = :uid) AND used_at IS NULL"
                ),
                {"uid": seeded_users.student_id},
            )
        ).scalar_one()
        assert codes_for_user == 10
