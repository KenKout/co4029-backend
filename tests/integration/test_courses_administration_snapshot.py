"""T7 characterization tests for ``features/courses/routers/administration.py``.

Targets the four ORM-migrated sites:

* The ``restore`` endpoint preserves ``execution_options(include_deleted=True)``
  via the load-then-mutate pattern -- the underlying service still
  reaches the soft-deleted row and the post-restore ``updated_by``
  stamp lands on the in-memory ORM instance (audit trigger T3 is the
  ultimate enforcer).
* ``GET /admin/courses/_stats`` returns identical row sets vs the
  legacy raw-SQL aggregations (verified by hand-rolled assertions
  rather than a JSON snapshot file -- the totals depend on test
  ordering relative to other suites' courses fixtures, so we assert
  ``>=`` invariants on shape rather than absolute counts).

Fixtures use raw ``text()`` (allowed for test setup by T4 lint).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import administration_router

for _stub_name in ("interview_configs",):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
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
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(administration_router, prefix="/api/v1")
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
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return session_id


@pytest_asyncio.fixture
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def t7_courses(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Two courses (one soft-deleted, one published) for stats + restore probes."""
    suffix = uuid.uuid4().hex[:8]
    soft_deleted = uuid.uuid4()
    published = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status, deleted_at, deleted_by) "
                "VALUES "
                "(:sd, :org, :owner, :slug_sd, 'T7 Soft-Deleted', 'draft', NOW(), :owner), "
                "(:pb, :org, :owner, :slug_pb, 'T7 Published', 'published', NULL, NULL)"
            ),
            {
                "sd": soft_deleted,
                "pb": published,
                "org": seeded_users.organization_id,
                "owner": seeded_users.admin_id,
                "slug_sd": f"t7-sd-{suffix}",
                "slug_pb": f"t7-pb-{suffix}",
            },
        )

    yield {"soft_deleted": soft_deleted, "published": published}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [soft_deleted, published]},
        )


@pytest.mark.asyncio
async def test_undelete_preserved(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    t7_courses: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    """Restore must clear ``deleted_at`` / ``deleted_by`` AND stamp ``updated_by``.

    The migrated router uses ``db.get(Course, ...)`` for the
    ``updated_by`` touch -- this only works because the ORM-side
    ``include_deleted=True`` opt-out is applied internally by the
    underlying service (``queries.administration.get_course_including_deleted``).
    """
    response = await client.post(
        f"/api/v1/admin/courses/{t7_courses['soft_deleted']}/restore",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT deleted_at, deleted_by, updated_by FROM courses WHERE id = :id"),
                {"id": t7_courses["soft_deleted"]},
            )
        ).one()
    assert row.deleted_at is None
    assert row.deleted_by is None
    assert row.updated_by == seeded_users.admin_id


@pytest.mark.asyncio
async def test_stats_aggregations_match_orm(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    t7_courses: dict[str, uuid.UUID],
) -> None:
    """ORM GROUP BY + FILTER aggregations match raw-SQL ground truth.

    Cross-checks the migrated stats endpoint against an equivalent
    raw SQL query run directly against the database. Asserting equality
    keeps the migration honest if the ORM emits a subtly different
    aggregation plan (e.g. NULL handling on ``filter()`` semantics).
    """
    response = await client.get(
        "/api/v1/admin/courses/_stats",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    async with engine.begin() as conn:
        active = (
            await conn.execute(text("SELECT COUNT(*) FROM courses WHERE deleted_at IS NULL"))
        ).scalar_one()
        soft_deleted = (
            await conn.execute(text("SELECT COUNT(*) FROM courses WHERE deleted_at IS NOT NULL"))
        ).scalar_one()
        by_status_rows = (
            await conn.execute(
                text(
                    "SELECT status, COUNT(*) AS count FROM courses "
                    "WHERE deleted_at IS NULL GROUP BY status ORDER BY status"
                )
            )
        ).all()

    assert body["total_courses"] == int(active)
    assert body["soft_deleted_courses"] == int(soft_deleted)

    expected_by_status = [{"status": row.status, "count": int(row.count)} for row in by_status_rows]
    assert body["by_status"] == expected_by_status


@pytest.mark.asyncio
async def test_stats_includes_t7_courses(
    client: httpx.AsyncClient,
    admin_bearer: str,
    t7_courses: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        "/api/v1/admin/courses/_stats",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    statuses = {row["status"]: row["count"] for row in body["by_status"]}
    assert statuses.get("published", 0) >= 1
    assert statuses.get("draft", 0) >= 1
    assert body["soft_deleted_courses"] >= 1
    assert isinstance(body["top_draft_owners"], list)
