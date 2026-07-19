"""Integration tests for the reusable offset pagination base (W3.5).

Covers `core.pagination.paginate`: server-side search (ilike), whitelisted
sort (unknown key ignored, known key asc/desc), total/total_pages, and
offset windowing — exercised through `select(Organization)`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- FK targets
from abridgeai.core.config import get_settings
from abridgeai.core.pagination import paginate
from abridgeai.features.access_control.models import Organization


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
        "script_location", str(Path(__file__).resolve().parents[2] / "migrations")
    )
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def orgs(engine: AsyncEngine) -> AsyncIterator[str]:
    """25 orgs under a unique prefix so tests don't collide with other data."""
    tag = uuid.uuid4().hex[:8]
    ids = [uuid.uuid4() for _ in range(25)]
    async with engine.begin() as conn:
        for i, oid in enumerate(ids):
            await conn.execute(
                text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
                {"id": oid, "slug": f"{tag}-{i:02d}", "name": f"{tag} Org {i:02d}"},
            )
        # One extra whose name contains a distinctive token for search.
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": uuid.uuid4(), "slug": f"{tag}-zzz", "name": f"{tag} Zebra Special"},
        )
    yield tag
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM organizations WHERE slug LIKE :p"), {"p": f"{tag}-%"}
        )


def _base_stmt(tag: str):
    # Scope every test to this fixture's rows via the unique slug prefix.
    return select(Organization).where(
        Organization.deleted_at.is_(None), Organization.slug.like(f"{tag}-%")
    )


@pytest.mark.asyncio
async def test_offset_windows_and_total(session_factory, orgs) -> None:
    async with session_factory() as db:
        p1 = await paginate(
            db, _base_stmt(orgs), page=0, page_size=10,
            sortable={"name": Organization.name}, sort="name",
            default_order=[Organization.id],
        )
        p3 = await paginate(
            db, _base_stmt(orgs), page=2, page_size=10,
            sortable={"name": Organization.name}, sort="name",
            default_order=[Organization.id],
        )
    assert p1.total == 26  # 25 + zebra
    assert p1.total_pages == 3
    assert len(p1.items) == 10
    assert len(p3.items) == 6  # last page remainder
    # No overlap across pages.
    assert {o.id for o in p1.items}.isdisjoint({o.id for o in p3.items})


@pytest.mark.asyncio
async def test_search_narrows_and_updates_total(session_factory, orgs) -> None:
    async with session_factory() as db:
        page = await paginate(
            db, _base_stmt(orgs), page=0, page_size=10,
            search="zebra", search_columns=[Organization.name, Organization.slug],
            default_order=[Organization.id],
        )
    assert page.total == 1
    assert page.items[0].name.endswith("Zebra Special")


@pytest.mark.asyncio
async def test_sort_direction_and_whitelist(session_factory, orgs) -> None:
    async with session_factory() as db:
        asc = await paginate(
            db, _base_stmt(orgs), page=0, page_size=30,
            sortable={"name": Organization.name}, sort="name", sort_dir="asc",
            default_order=[Organization.id],
        )
        desc = await paginate(
            db, _base_stmt(orgs), page=0, page_size=30,
            sortable={"name": Organization.name}, sort="name", sort_dir="desc",
            default_order=[Organization.id],
        )
        # Unknown sort key is ignored (falls back to default_order) — no error.
        unknown = await paginate(
            db, _base_stmt(orgs), page=0, page_size=30,
            sortable={"name": Organization.name}, sort="password; DROP TABLE",
            default_order=[Organization.id],
        )
    names_asc = [o.name for o in asc.items]
    assert names_asc == sorted(names_asc)
    assert [o.name for o in desc.items] == list(reversed(names_asc))
    assert unknown.total == 26  # ignored bad sort, still returns rows
