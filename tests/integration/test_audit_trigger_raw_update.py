"""T3: raw ``text("UPDATE ...")`` from inside an authenticated HTTP request
stamps ``courses.updated_by`` from the ``app.actor_id`` GUC.

Closes the ORM-bypass gap: the in-process ``audit_listener`` only fires on
ORM ``flush``. Code paths that issue raw SQL ``UPDATE`` -- common for bulk
title rewrites, soft-delete cascades, admin patches -- previously left
``updated_by`` stale. The Postgres ``audit_stamp`` trigger reads
``current_setting('app.actor_id', true)`` (bound by T2's ``after_begin``
listener) and overwrites ``updated_by`` on every UPDATE.
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
async def app(engine: AsyncEngine) -> AsyncIterator[FastAPI]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    fastapi_app = FastAPI()

    @fastapi_app.post("/test/raw-update-course/{course_id}")
    async def _raw_update(
        course_id: uuid.UUID,
        new_title: str,
        current: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> dict[str, str]:
        await db.commit()
        await db.execute(
            text("UPDATE courses SET title = :t WHERE id = :id"),
            {"t": new_title, "id": course_id},
        )
        await db.commit()
        return {"actor": str(current.user_id)}

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


async def _seed_user_session_org_course(
    engine: AsyncEngine,
) -> tuple[uuid.UUID, str, uuid.UUID, uuid.UUID]:
    user_id = uuid.uuid4()
    sid = uuid.uuid4()
    org_id = uuid.uuid4()
    course_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    suffix = user_id.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": user_id, "email": f"raw-update-{suffix}@abridgeai.local"},
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
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {"id": org_id, "slug": f"raw-update-org-{suffix}", "name": "Raw Update Test Org"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": user_id,
                "slug": f"raw-update-course-{suffix}",
                "title": "Original Title",
            },
        )
    token = create_access_token(user_id=user_id, session_id=sid)
    return user_id, token, org_id, course_id


async def _purge(engine: AsyncEngine, user_id: uuid.UUID, org_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM courses WHERE organization_id = :o"), {"o": org_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})
        await conn.execute(text("DELETE FROM auth_sessions WHERE user_id = :u"), {"u": user_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})


async def test_raw_update_stamps_updated_by(app: FastAPI, engine: AsyncEngine) -> None:
    user_id, token, org_id, course_id = await _seed_user_session_org_course(engine)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.post(
                f"/test/raw-update-course/{course_id}",
                params={"new_title": "Renamed via raw SQL"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["actor"] == str(user_id)

        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT title, updated_by FROM courses WHERE id = :id"),
                    {"id": course_id},
                )
            ).one()
        assert row.title == "Renamed via raw SQL"
        assert row.updated_by == user_id, f"expected updated_by={user_id}, got {row.updated_by}"
    finally:
        await _purge(engine, user_id, org_id)
