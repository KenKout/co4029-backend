"""T2: ``app.actor_id`` does not leak across pooled-connection reuse.

Forces a single-connection pool, then makes two SEQUENTIAL requests as
different users on the SAME underlying DBAPI connection. The
transaction-local ``set_config('app.actor_id', :v, true)`` must reset
between requests so user B never reads user A's actor.
"""

from __future__ import annotations

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

import abridgeai.features.identity.models  # noqa: F401
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
async def single_conn_engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(
        _async_url(get_settings().database_url),
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )
    register_app_actor_listener(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def app(
    single_conn_engine: AsyncEngine,
) -> AsyncIterator[FastAPI]:
    factory = async_sessionmaker(single_conn_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    fastapi_app = FastAPI()

    @fastapi_app.get("/test/actor")
    async def _actor(
        current: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> dict[str, str | None]:
        await db.commit()
        actor = (
            await db.execute(text("SELECT current_setting('app.actor_id', true)"))
        ).scalar_one()
        await db.rollback()
        return {"actor": actor or None, "expected": str(current.user_id)}

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


async def _seed_user(engine: AsyncEngine) -> tuple[uuid.UUID, str]:
    user_id = uuid.uuid4()
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": user_id, "email": f"pool-iso-{user_id.hex}@abridgeai.local"},
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
    return user_id, create_access_token(user_id=user_id, session_id=sid)


async def _purge_user(engine: AsyncEngine, user_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE user_id = :u"), {"u": user_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})


async def test_actor_id_does_not_leak_on_shared_pool_connection(
    app: FastAPI,
    single_conn_engine: AsyncEngine,
) -> None:
    user_a, token_a = await _seed_user(single_conn_engine)
    user_b, token_b = await _seed_user(single_conn_engine)

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp_a = await ac.get("/test/actor", headers={"Authorization": f"Bearer {token_a}"})
            resp_b = await ac.get("/test/actor", headers={"Authorization": f"Bearer {token_b}"})

        assert resp_a.status_code == 200, resp_a.text
        assert resp_b.status_code == 200, resp_b.text

        body_a = resp_a.json()
        body_b = resp_b.json()
        assert body_a["actor"] == str(user_a)
        assert body_b["actor"] == str(user_b), (
            f"actor leaked across pool conn reuse: B saw {body_b['actor']!r}, expected {user_b}"
        )
    finally:
        await _purge_user(single_conn_engine, user_a)
        await _purge_user(single_conn_engine, user_b)
