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

from abridgeai.core.audit import current_actor_var
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.identity.routers.me import router as me_router


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
async def auth_session(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    session_id = uuid.uuid4()
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
                "h": hash_secret(generate_token()),
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


def test_router_metadata() -> None:
    assert me_router.prefix == "/users/me"
    assert "users" in me_router.tags
    assert "me" in me_router.tags
    paths = {(r.path, tuple(sorted(r.methods))) for r in me_router.routes}  # type: ignore[attr-defined]
    assert ("/users/me", ("GET",)) in paths
    assert ("/users/me/profile", ("PATCH",)) in paths
    assert ("/users/me/permissions", ("GET",)) in paths


async def test_get_me_without_token_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401


async def test_get_me_with_valid_token_returns_user(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    _, token = auth_session
    response = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(seeded_users.student_id)
    assert body["primary_email"].endswith("student@abridgeai.local")
    assert body["status"] == "active"
    assert body["profile"] is not None
    assert body["profile"]["display_name"]
    assert "password" not in body
    assert "password_hash" not in body


async def test_patch_profile_updates_fields_and_audit_columns(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
    test_engine_local: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    _, token = auth_session
    actor_token = current_actor_var.set(seeded_users.student_id)
    try:
        response = await client.patch(
            "/api/v1/users/me/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={"display_name": "Updated Student", "bio": "Hello world"},
        )
    finally:
        current_actor_var.reset(actor_token)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile"]["display_name"] == "Updated Student"
    assert body["profile"]["bio"] == "Hello world"

    async with test_engine_local.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT display_name, bio, updated_by FROM user_profiles WHERE user_id = :uid"
                ),
                {"uid": seeded_users.student_id},
            )
        ).one()
        assert row[0] == "Updated Student"
        assert row[1] == "Hello world"
        assert row[2] == seeded_users.student_id


async def test_get_me_permissions_returns_user_perms(
    client: httpx.AsyncClient,
    auth_session: tuple[uuid.UUID, str],
) -> None:
    _, token = auth_session
    response = await client.get(
        "/api/v1/users/me/permissions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["permissions"], list)
    assert "course.read" in body["permissions"]
    assert "quiz.take" in body["permissions"]
    assert body["permissions"] == sorted(body["permissions"])
