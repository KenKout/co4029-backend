"""Integration tests for T0.26 health check endpoints.

Covers ``/healthz`` (liveness, < 50ms, no auth), ``/healthz/deep`` (admin
composite report) and ``/readyz`` (K8s readiness with alembic head check).
"""

from __future__ import annotations

import time
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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.enrollments.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.notifications.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.api import create_app
from abridgeai.api import healthz as healthz_module
from abridgeai.api.healthz import CheckStatus
from abridgeai.core.config import get_settings
from abridgeai.core.security import create_access_token, generate_token, hash_secret


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
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
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
    return sid


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> str:
    sid = await _seed_session(engine, user_id)
    return create_access_token(user_id=user_id, session_id=sid)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _purge_sessions(engine: AsyncEngine, user_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = :u"),
            {"u": user_id},
        )


def _patch_all_checks_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok_pg() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=12.0)

    async def _ok_redis() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=3.0)

    async def _disabled_neo4j() -> CheckStatus:
        return CheckStatus(status="disabled", latency_ms=None)

    async def _disabled_s3() -> CheckStatus:
        return CheckStatus(status="disabled", latency_ms=None)

    monkeypatch.setattr(healthz_module, "_check_postgres", _ok_pg)
    monkeypatch.setattr(healthz_module, "_check_redis", _ok_redis)
    monkeypatch.setattr(healthz_module, "_check_neo4j", _disabled_neo4j)
    monkeypatch.setattr(healthz_module, "_check_garage_s3", _disabled_s3)


async def test_healthz_simple_returns_200_fast(client: httpx.AsyncClient) -> None:
    await client.get("/healthz")
    start = time.perf_counter()
    resp = await client.get("/healthz")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert elapsed_ms < 50.0, f"liveness probe took {elapsed_ms:.2f}ms (>= 50ms budget)"


async def test_healthz_simple_no_auth_required(client: httpx.AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200


async def test_healthz_deep_requires_admin(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token = await _bearer(engine, seeded_users.student_id)
    try:
        resp = await client.get("/healthz/deep", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.student_id)
    assert resp.status_code == 403


async def test_healthz_deep_returns_composite_status(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_all_checks_ok(monkeypatch)
    token = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get("/healthz/deep", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["checks"].keys()) == {"postgres", "redis", "neo4j", "garage_s3", "llm_provider"}
    assert body["status"] in {"ok", "degraded", "unhealthy"}
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["redis"]["status"] == "ok"
    assert body["checks"]["llm_provider"]["status"] == "skipped"
    assert "version" in body
    assert isinstance(body["version"], str)
    assert body["version"]
    assert "git_sha" in body


async def test_healthz_deep_neo4j_disabled_when_uri_empty(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok_pg() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=12.0)

    async def _ok_redis() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=3.0)

    async def _disabled_s3() -> CheckStatus:
        return CheckStatus(status="disabled", latency_ms=None)

    monkeypatch.setattr(healthz_module, "_check_postgres", _ok_pg)
    monkeypatch.setattr(healthz_module, "_check_redis", _ok_redis)
    monkeypatch.setattr(healthz_module, "_check_garage_s3", _disabled_s3)

    settings = get_settings()
    monkeypatch.setattr(settings, "knowledge_graph_enabled", False)
    monkeypatch.setattr(settings, "neo4j_uri", "")

    token = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get("/healthz/deep", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["checks"]["neo4j"]["status"] == "disabled"
    assert body["checks"]["neo4j"]["latency_ms"] is None


async def test_readyz_200_when_at_head(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok_pg() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=10.0)

    async def _ok_redis() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=2.0)

    async def _at_head() -> bool:
        return True

    monkeypatch.setattr(healthz_module, "_check_postgres", _ok_pg)
    monkeypatch.setattr(healthz_module, "_check_redis", _ok_redis)
    monkeypatch.setattr(healthz_module, "_check_alembic_at_head", _at_head)

    resp = await client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"postgres": "ok", "redis": "ok", "alembic_at_head": True}


async def test_readyz_503_when_db_down(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _down_pg() -> CheckStatus:
        return CheckStatus(status="unhealthy", latency_ms=None)

    async def _ok_redis() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=2.0)

    async def _at_head() -> bool:
        return True

    monkeypatch.setattr(healthz_module, "_check_postgres", _down_pg)
    monkeypatch.setattr(healthz_module, "_check_redis", _ok_redis)
    monkeypatch.setattr(healthz_module, "_check_alembic_at_head", _at_head)

    resp = await client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["postgres"] == "unhealthy"


async def test_readyz_no_auth_required(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok_pg() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=10.0)

    async def _ok_redis() -> CheckStatus:
        return CheckStatus(status="ok", latency_ms=2.0)

    async def _at_head() -> bool:
        return True

    monkeypatch.setattr(healthz_module, "_check_postgres", _ok_pg)
    monkeypatch.setattr(healthz_module, "_check_redis", _ok_redis)
    monkeypatch.setattr(healthz_module, "_check_alembic_at_head", _at_head)

    resp = await client.get("/readyz")
    assert resp.status_code == 200
