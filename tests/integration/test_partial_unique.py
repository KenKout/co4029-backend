from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace(
            "postgresql+psycopg://", "postgresql+psycopg_async://", 1
        )
    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://", "postgresql+psycopg_async://", 1
        )
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(
        _async_url(get_settings().database_url), pool_pre_ping=True
    )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def org_and_owner(engine: AsyncEngine):
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name) "
                "VALUES (:id, :slug, :name)"
            ),
            {
                "id": org_id,
                "slug": f"puq-{org_id.hex[:8]}",
                "name": "Partial UQ Test Org",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email) VALUES (:id, :email)"
            ),
            {"id": owner_id, "email": f"puq-{owner_id.hex[:8]}@test.local"},
        )
    yield org_id, owner_id
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM courses WHERE organization_id = :id"),
            {"id": org_id},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = :id"), {"id": owner_id}
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"), {"id": org_id}
        )


async def _insert_course(conn, org_id, owner_id, slug):
    await conn.execute(
        text(
            "INSERT INTO courses (organization_id, owner_user_id, slug, title) "
            "VALUES (:org, :owner, :slug, :title)"
        ),
        {
            "org": org_id,
            "owner": owner_id,
            "slug": slug,
            "title": f"course {slug}",
        },
    )


async def test_reuse_after_delete(engine: AsyncEngine, org_and_owner):
    org_id, owner_id = org_and_owner
    slug = "reuse-intro"

    async with engine.begin() as conn:
        await _insert_course(conn, org_id, owner_id, slug)
        await conn.execute(
            text(
                "UPDATE courses SET deleted_at = NOW() "
                "WHERE organization_id = :org AND slug = :slug"
            ),
            {"org": org_id, "slug": slug},
        )
        await _insert_course(conn, org_id, owner_id, slug)
        result = await conn.execute(
            text(
                "SELECT count(*) FROM courses "
                "WHERE organization_id = :org AND slug = :slug"
            ),
            {"org": org_id, "slug": slug},
        )
        assert result.scalar() == 2


async def test_no_reuse_when_active(engine: AsyncEngine, org_and_owner):
    org_id, owner_id = org_and_owner
    slug = "active-y"

    async with engine.begin() as conn:
        await _insert_course(conn, org_id, owner_id, slug)

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await _insert_course(conn, org_id, owner_id, slug)
