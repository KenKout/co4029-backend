"""Integration tests for T0.23 HTTP audit log middleware."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
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
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.observability.audit_log import AuditLogMiddleware
from abridgeai.core.security import (
    create_access_token,
    generate_token,
    hash_secret,
)
from abridgeai.features.admin.routers.processing import (
    get_arq_pool as get_admin_arq_pool,
)
from abridgeai.features.interviews.routers.authoring import (
    get_arq_pool as get_interview_authoring_arq_pool,
)
from abridgeai.features.interviews.routers.learner import (
    get_arq_pool as get_interview_learner_arq_pool,
)
from abridgeai.features.materials.routers.authoring import (
    get_arq_pool as get_materials_arq_pool,
)
from abridgeai.features.quizzes.routers.authoring import (
    get_arq_pool as get_quiz_arq_pool,
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
async def purge_audit(engine: AsyncEngine) -> AsyncIterator[None]:
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=1)
    yield
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM http_audit_log WHERE created_at >= :c"),
            {"c": cutoff},
        )


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _none_pool() -> object | None:
        return None

    fastapi_app = create_app()
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    for dep in (
        get_admin_arq_pool,
        get_materials_arq_pool,
        get_quiz_arq_pool,
        get_interview_authoring_arq_pool,
        get_interview_learner_arq_pool,
    ):
        fastapi_app.dependency_overrides[dep] = _none_pool
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(eng: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with eng.begin() as conn:
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


async def _admin_bearer(eng: AsyncEngine, admin_id: uuid.UUID) -> tuple[str, uuid.UUID]:
    sid = await _seed_session(eng, admin_id)
    return create_access_token(user_id=admin_id, session_id=sid), sid


async def _purge_session(eng: AsyncEngine, sid: uuid.UUID) -> None:
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


async def _row_for_request(eng: AsyncEngine, request_id: str) -> dict[str, Any] | None:
    async with eng.begin() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT method, path, query_params, headers_meta, status_code, "
                        "       latency_ms, request_id, path_params, body_size_bytes, "
                        "       user_id, user_agent "
                        "FROM http_audit_log WHERE request_id = CAST(:rid AS uuid) "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"rid": request_id},
                )
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


async def test_full_request_logged_with_redaction(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    purge_audit: None,
) -> None:
    del purge_audit
    resp = await client.post(
        "/api/v1/auth/google/callback",
        params={"code": "secret123"},
    )
    request_id = resp.headers.get("X-Request-ID")
    assert request_id is not None

    row = await _row_for_request(engine, request_id)
    assert row is not None, "audit row not persisted"
    assert row["method"] == "POST"
    assert row["path"] == "/api/v1/auth/google/callback"
    assert "secret123" not in str(row["query_params"])
    assert row["query_params"]["code"] == ["[REDACTED]"]


async def test_x_request_id_header_present(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    purge_audit: None,
) -> None:
    del purge_audit
    resp = await client.get("/api/v1/me")
    request_id = resp.headers.get("X-Request-ID")
    assert request_id is not None
    uuid.UUID(request_id)

    row = await _row_for_request(engine, request_id)
    assert row is not None
    assert str(row["request_id"]) == request_id


async def test_healthz_not_logged(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    purge_audit: None,
) -> None:
    del purge_audit
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=1)
    for _ in range(100):
        await client.get("/healthz")

    async with engine.begin() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM http_audit_log "
                    "WHERE path = '/healthz' AND created_at >= :c"
                ),
                {"c": cutoff},
            )
        ).scalar_one()
    assert count == 0


async def test_authorization_header_redacted(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    purge_audit: None,
) -> None:
    del purge_audit
    resp = await client.get(
        "/api/v1/me",
        headers={"Authorization": "Bearer XYZ-PLAINTEXT-SHOULD-NEVER-PERSIST"},
    )
    request_id = resp.headers["X-Request-ID"]

    row = await _row_for_request(engine, request_id)
    assert row is not None
    headers_meta = row["headers_meta"]
    assert "XYZ-PLAINTEXT-SHOULD-NEVER-PERSIST" not in str(headers_meta)
    assert headers_meta.get("authorization") == "[REDACTED]"


async def test_mutation_logs_path_params_and_body_size(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    purge_audit: None,
) -> None:
    del purge_audit
    course_id = uuid.uuid4()
    resp = await client.post(
        f"/api/v1/teacher/courses/{course_id}/quizzes",
        json={"title": "AuditLog test quiz", "total_questions": 5},
    )
    request_id = resp.headers["X-Request-ID"]

    row = await _row_for_request(engine, request_id)
    assert row is not None
    assert row["body_size_bytes"] is not None
    assert row["body_size_bytes"] > 0


async def test_db_failure_does_not_crash_request() -> None:
    def _failing_sessionmaker() -> Any:
        raise OperationalError("stmt", {}, Exception("boom"))

    failing_app = FastAPI()
    failing_app.add_middleware(AuditLogMiddleware, sessionmaker=_failing_sessionmaker)

    @failing_app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"ok": "yes"}

    transport = httpx.ASGITransport(app=failing_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"ok": "yes"}
    assert resp.headers["X-Request-ID"]


async def test_admin_audit_http_endpoint_now_returns_200(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    purge_audit: None,
) -> None:
    del purge_audit
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=5)
    token, sid = await _admin_bearer(engine, seeded_users.admin_id)
    try:
        warmup = await client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
        assert warmup.headers.get("X-Request-ID")

        since = (cutoff).isoformat().replace("+00:00", "Z")
        resp = await client.get(
            "/api/v1/admin/audit/http",
            params={"since": since, "limit": 50, "path_pattern": "/api/v1/%"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        await _purge_session(engine, sid)

    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert any(r["path"] == "/api/v1/me" for r in rows)
