"""T2: concurrent HTTP requests stamp ``app.actor_id`` without cross-contamination.

Fires 50 in-process FastAPI requests in parallel, each authenticated as a
different user, and asserts each request reads back its OWN actor from the
``app.actor_id`` GUC. Catches any cross-task contextvar leakage and any
listener-binding bug where actor_id from one request bleeds into another's
transaction on a shared pool connection.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.identity.models  # noqa: F401  -- register FK targets
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db, register_app_actor_listener
from abridgeai.core.security import (
    CurrentUser,
    create_access_token,
    generate_token,
    get_current_user,
    hash_secret,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    register_app_actor_listener(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()

    @fastapi_app.get("/test/actor")
    async def _actor(
        current: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> dict[str, str | None]:
        # Commit to end the txn opened by get_current_user's auth lookup, then
        # let the next execute open a fresh txn so the after_begin listener
        # runs AFTER the contextvar bind is in place.
        await db.commit()
        actor = (
            await db.execute(text("SELECT current_setting('app.actor_id', true)"))
        ).scalar_one()
        await db.rollback()
        return {"actor": actor or None, "expected": str(current.user_id)}

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


async def _seed_user_and_session(engine: AsyncEngine) -> tuple[uuid.UUID, uuid.UUID]:
    user_id = uuid.uuid4()
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": user_id, "email": f"actor-test-{user_id.hex}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": sid,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return user_id, sid


async def _purge_user(engine: AsyncEngine, user_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE user_id = :u"), {"u": user_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})


async def test_50_concurrent_requests_stamp_correct_actor(
    app: FastAPI,
    engine: AsyncEngine,
) -> None:
    seeded: list[tuple[uuid.UUID, uuid.UUID, str]] = []
    for _ in range(50):
        user_id, sid = await _seed_user_and_session(engine)
        token = create_access_token(user_id=user_id, session_id=sid)
        seeded.append((user_id, sid, token))

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:

            async def _hit(token: str) -> dict[str, str | None]:
                resp = await ac.get(
                    "/test/actor",
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert resp.status_code == 200, resp.text
                return resp.json()

            responses = await asyncio.gather(*(_hit(token) for _, _, token in seeded))

        for (user_id, _sid, _token), body in zip(seeded, responses, strict=True):
            assert body["expected"] == str(user_id)
            assert body["actor"] == str(user_id), (
                f"actor_id leaked: expected {user_id}, got {body['actor']}"
            )
    finally:
        for user_id, _sid, _token in seeded:
            await _purge_user(engine, user_id)
