"""T2: actor_id stays bound across multiple commits within one HTTP request.

Many handlers in this codebase issue 2+ ``db.commit()`` calls in a single
request. Each commit ends the transaction; the next operation begins a new
one. The ``after_begin`` listener must re-stamp ``app.actor_id`` from the
contextvar on every begin, so writes from txn1 and txn2 stamp the SAME
authenticated actor.
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
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    register_app_actor_listener(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def app(engine: AsyncEngine) -> AsyncIterator[FastAPI]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    fastapi_app = FastAPI()

    @fastapi_app.get("/test/multi-commit-actor")
    async def _multi(
        current: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> dict[str, str | None]:
        await db.commit()
        actor_txn1 = (
            await db.execute(text("SELECT current_setting('app.actor_id', true)"))
        ).scalar_one()
        await db.commit()
        actor_txn2 = (
            await db.execute(text("SELECT current_setting('app.actor_id', true)"))
        ).scalar_one()
        await db.rollback()
        return {
            "expected": str(current.user_id),
            "actor_txn1": actor_txn1 or None,
            "actor_txn2": actor_txn2 or None,
        }

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
            {"id": user_id, "email": f"multi-commit-{user_id.hex}@abridgeai.local"},
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


async def test_actor_id_persists_across_two_commits_in_one_request(
    app: FastAPI,
    engine: AsyncEngine,
) -> None:
    user_id, token = await _seed_user(engine)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get(
                "/test/multi-commit-actor",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["expected"] == str(user_id)
        assert body["actor_txn1"] == str(user_id)
        assert body["actor_txn2"] == str(user_id), (
            "actor_id was lost between commits: "
            f"txn1={body['actor_txn1']!r}, txn2={body['actor_txn2']!r}"
        )
    finally:
        await _purge_user(engine, user_id)
