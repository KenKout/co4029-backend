"""T3: ``created_by`` is IMMUTABLE on UPDATE; ``updated_by`` re-stamps every UPDATE.

User A inserts a course (trigger stamps ``created_by = A``, ``updated_by = A``).
User B updates the course (trigger MUST leave ``created_by = A`` untouched
and stamp ``updated_by = B``). The plpgsql ``IF TG_OP = 'INSERT'`` branch
is the only path that writes ``created_by``; the ``ELSIF TG_OP = 'UPDATE'``
branch only writes ``updated_by``. This test enforces that semantic.
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

    @fastapi_app.post("/test/insert-course")
    async def _insert(
        organization_id: uuid.UUID,
        course_id: uuid.UUID,
        slug: str,
        title: str,
        current: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> dict[str, str]:
        await db.commit()
        await db.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": organization_id,
                "owner": current.user_id,
                "slug": slug,
                "title": title,
            },
        )
        await db.commit()
        return {"actor": str(current.user_id)}

    @fastapi_app.post("/test/update-course/{course_id}")
    async def _update(
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


async def _seed_user(engine: AsyncEngine, label: str) -> tuple[uuid.UUID, str]:
    user_id = uuid.uuid4()
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": user_id, "email": f"{label}-{user_id.hex[:8]}@abridgeai.local"},
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


async def _seed_org(engine: AsyncEngine) -> uuid.UUID:
    org_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {"id": org_id, "slug": f"immutable-org-{org_id.hex[:8]}", "name": "Immutable Test Org"},
        )
    return org_id


async def _purge(
    engine: AsyncEngine,
    org_id: uuid.UUID,
    user_ids: list[uuid.UUID],
) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM courses WHERE organization_id = :o"), {"o": org_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = ANY(:u)"), {"u": user_ids}
        )
        await conn.execute(text("DELETE FROM users WHERE id = ANY(:u)"), {"u": user_ids})


async def test_created_by_immutable_updated_by_restamps(app: FastAPI, engine: AsyncEngine) -> None:
    user_a, token_a = await _seed_user(engine, "user-a")
    user_b, token_b = await _seed_user(engine, "user-b")
    org_id = await _seed_org(engine)
    course_id = uuid.uuid4()
    slug = f"immutable-course-{course_id.hex[:8]}"

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            insert_resp = await ac.post(
                "/test/insert-course",
                params={
                    "organization_id": str(org_id),
                    "course_id": str(course_id),
                    "slug": slug,
                    "title": "User A's course",
                },
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert insert_resp.status_code == 200, insert_resp.text
            assert insert_resp.json()["actor"] == str(user_a)

            async with engine.begin() as conn:
                row1 = (
                    await conn.execute(
                        text("SELECT created_by, updated_by FROM courses WHERE id = :id"),
                        {"id": course_id},
                    )
                ).one()
            assert row1.created_by == user_a
            assert row1.updated_by == user_a

            update_resp = await ac.post(
                f"/test/update-course/{course_id}",
                params={"new_title": "Renamed by User B"},
                headers={"Authorization": f"Bearer {token_b}"},
            )
            assert update_resp.status_code == 200, update_resp.text
            assert update_resp.json()["actor"] == str(user_b)

            async with engine.begin() as conn:
                row2 = (
                    await conn.execute(
                        text("SELECT created_by, updated_by, title FROM courses WHERE id = :id"),
                        {"id": course_id},
                    )
                ).one()
            assert row2.title == "Renamed by User B"
            assert row2.created_by == user_a, (
                f"created_by must be IMMUTABLE; expected {user_a}, got {row2.created_by}"
            )
            assert row2.updated_by == user_b, (
                f"updated_by must re-stamp; expected {user_b}, got {row2.updated_by}"
            )
    finally:
        await _purge(engine, org_id, [user_a, user_b])
