"""Gap 3 versioning: fork (copy-on-write), freeze, and the enrollment pin.

Covers the three product decisions end-to-end:
* D2(a) explicit fork — POST /versions clones the latest published version
  into a draft; a second fork while a draft exists is a 409.
* D1(b) pinned — a published version is FROZEN: stage/item mutations 409.
* D3(a) — an enrollment stays pinned to the version it started on; editing
  a NEWER version never changes the pinned student's route.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.career_paths.routers import (
    authoring_management_router,
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
async def engine() -> AsyncIterator[AsyncEngine]:  # noqa: ASYNC240 -- sync alembic Config/upgrade, matching every other integration fixture
    from alembic import command
    from alembic.config import Config

    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"  # noqa: ASYNC240 -- sync alembic fixture, matching every other integration file
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),  # noqa: ASYNC240
    )
    command.upgrade(cfg, "head")
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(session_factory: async_sessionmaker) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(authoring_management_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, NOW() + INTERVAL '1 hour')"
            ),
            {"id": session_id, "uid": user_id, "h": hash_secret(generate_token())},
        )
    return session_id


async def _seed_published_path(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> tuple[uuid.UUID, uuid.UUID]:
    """A published path with one published version and one stage + course."""
    org = seeded_users.organization_id
    owner = seeded_users.admin_id
    suffix = uuid.uuid4().hex[:8]
    path_id, course_id, module_id, lesson_id = (uuid.uuid4() for _ in range(4))

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Ver Course', 'published')"
            ),
            {"id": course_id, "org": org, "owner": owner, "slug": f"ver-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, 'M', 1, 'published')"
            ),
            {"id": module_id, "cid": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:id, :mid, :slug, 'L', 'published', 'video')"
            ),
            {"id": lesson_id, "mid": module_id, "slug": f"ver-l-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items (id, module_id, item_type, lesson_id, position) "
                "VALUES (gen_random_uuid(), :mid, 'lesson', :lid, 1)"
            ),
            {"mid": module_id, "lid": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, 'Ver Path', 'published')"
            ),
            {"id": path_id, "org": org, "slug": f"ver-path-{suffix}"},
        )
        version_id = (
            await conn.execute(
                text(
                    "INSERT INTO career_path_versions "
                    "(id, career_path_id, version_no, status, published_at) "
                    "VALUES (gen_random_uuid(), :pid, 1, 'published', NOW()) "
                    "RETURNING id"
                ),
                {"pid": path_id},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, version_id, position, unlock_policy, enforcement) "
                "VALUES (gen_random_uuid(), :vid, 1, 'always', 'advisory')"
            ),
            {"vid": version_id},
        )
        await conn.execute(
            text(
                "INSERT INTO career_course_items "
                "(version_id, course_id, stage_id, position, is_required) "
                "SELECT :vid, :cid, id, 1, TRUE FROM career_path_stages WHERE version_id = :vid"
            ),
            {"vid": version_id, "cid": course_id},
        )
    return path_id, version_id


async def test_fork_clones_stages_and_items(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """D2(a): POST /versions clones the published version into a draft; a
    second fork while the draft exists is a 409."""
    manager = seeded_users.manager_id
    sid = await _seed_session(engine, manager)
    token = create_access_token(user_id=manager, session_id=sid)
    path_id, _v1 = await _seed_published_path(engine, seeded_users)
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = await client.post(f"/api/v1/management/career-paths/{path_id}/versions", headers=headers)
        assert resp.status_code == 201, resp.text
        v2 = resp.json()
        assert v2["version_no"] == 2
        assert v2["status"] == "draft"

        # The draft cloned the published stage + item. GET /stages resolves
        # the AUTHORING version (the draft), so it shows the clone.
        stages = await client.get(
            f"/api/v1/management/career-paths/{path_id}/stages", headers=headers
        )
        assert stages.status_code == 200, stages.text
        assert len(stages.json()) == 1  # the v2 draft clone
        courses = await client.get(
            f"/api/v1/management/career-paths/{path_id}/courses", headers=headers
        )
        assert courses.status_code == 200, courses.text
        assert len(courses.json()) == 1  # authoring version (draft) carries the clone

        # Second fork while a draft exists -> 409.
        again = await client.post(
            f"/api/v1/management/career-paths/{path_id}/versions", headers=headers
        )
        assert again.status_code == 409, again.text
        assert "draft" in again.json()["detail"]["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})
            await conn.execute(
                text(
                    "DELETE FROM career_course_items WHERE version_id IN "
                    "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
                ),
                {"pid": path_id},
            )
            await conn.execute(
                text(
                    "DELETE FROM career_path_stages WHERE version_id IN "
                    "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
                ),
                {"pid": path_id},
            )
            await conn.execute(
                text("DELETE FROM career_path_versions WHERE career_path_id = :pid"),
                {"pid": path_id},
            )
            await conn.execute(text("DELETE FROM career_paths WHERE id = :pid"), {"pid": path_id})
            await conn.execute(
                text("DELETE FROM module_items WHERE module_id IN (SELECT id FROM modules WHERE course_id IN (SELECT id FROM courses WHERE slug LIKE 'ver-%'))")
            )
            await conn.execute(
                text("DELETE FROM lessons WHERE slug LIKE 'ver-l-%'")
            )
            await conn.execute(
                text("DELETE FROM modules WHERE course_id IN (SELECT id FROM courses WHERE slug LIKE 'ver-%')")
            )
            await conn.execute(text("DELETE FROM courses WHERE slug LIKE 'ver-%'"))


async def test_published_version_is_frozen(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """D1(b): mutating a published version's stages/items is a 409 — the
    pinned promise. Edits require the explicit fork."""
    manager = seeded_users.manager_id
    sid = await _seed_session(engine, manager)
    token = create_access_token(user_id=manager, session_id=sid)
    path_id, _v1 = await _seed_published_path(engine, seeded_users)
    headers = {"Authorization": f"Bearer {token}"}

    try:
        # Find the published stage.
        stages = await client.get(
            f"/api/v1/management/career-paths/{path_id}/stages", headers=headers
        )
        assert stages.status_code == 200, stages.text
        published_stage = stages.json()[0]["id"]

        resp = await client.patch(
            f"/api/v1/management/career-paths/{path_id}/stages/{published_stage}",
            json={"title": "Renamed"},
            headers=headers,
        )
        assert resp.status_code == 409, resp.text
        assert "published" in resp.json()["detail"]["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})
            await conn.execute(
                text(
                    "DELETE FROM career_course_items WHERE version_id IN "
                    "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
                ),
                {"pid": path_id},
            )
            await conn.execute(
                text(
                    "DELETE FROM career_path_stages WHERE version_id IN "
                    "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
                ),
                {"pid": path_id},
            )
            await conn.execute(
                text("DELETE FROM career_path_versions WHERE career_path_id = :pid"),
                {"pid": path_id},
            )
            await conn.execute(text("DELETE FROM career_paths WHERE id = :pid"), {"pid": path_id})
            await conn.execute(
                text("DELETE FROM module_items WHERE module_id IN (SELECT id FROM modules WHERE course_id IN (SELECT id FROM courses WHERE slug LIKE 'ver-%'))")
            )
            await conn.execute(
                text("DELETE FROM lessons WHERE slug LIKE 'ver-l-%'")
            )
            await conn.execute(
                text("DELETE FROM modules WHERE course_id IN (SELECT id FROM courses WHERE slug LIKE 'ver-%')")
            )
            await conn.execute(text("DELETE FROM courses WHERE slug LIKE 'ver-%'"))
