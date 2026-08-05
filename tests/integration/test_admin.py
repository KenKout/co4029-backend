from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote_plus

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.admin.routers import (
    audit_router,
    processing_router,
    stats_router,
    users_router,
)
from abridgeai.features.admin.routers.processing import get_arq_pool


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
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[FastAPI, AsyncMock]]:
    arq_pool = AsyncMock()
    arq_pool.enqueue_job = AsyncMock()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _override_arq_pool() -> object:
        return arq_pool

    fastapi_app = FastAPI()
    fastapi_app.include_router(stats_router, prefix="/api/v1")
    fastapi_app.include_router(audit_router, prefix="/api/v1")
    fastapi_app.include_router(processing_router, prefix="/api/v1")
    fastapi_app.include_router(users_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    fastapi_app.dependency_overrides[get_arq_pool] = _override_arq_pool
    yield fastapi_app, arq_pool
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: tuple[FastAPI, AsyncMock]) -> AsyncIterator[httpx.AsyncClient]:
    fastapi_app, _ = app
    transport = httpx.ASGITransport(app=fastapi_app)
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


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> tuple[str, uuid.UUID]:
    sid = await _seed_session(engine, user_id)
    token = create_access_token(user_id=user_id, session_id=sid)
    return token, sid


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _purge_sessions(engine: AsyncEngine, user_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = :u"),
            {"u": user_id},
        )


@pytest_asyncio.fixture
async def extra_org(engine: AsyncEngine) -> AsyncIterator[dict[str, uuid.UUID]]:
    other_org = uuid.uuid4()
    other_unit = uuid.uuid4()
    other_user = uuid.uuid4()
    other_course = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Other Org', 'active')"
            ),
            {"id": other_org, "slug": f"other-{other_org}"},
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'department', 'Other Dept', 'OTHER-DEPT')"
            ),
            {"id": other_unit, "org": other_org},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": other_user, "email": f"other-{other_user}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(user_id, organization_id, status) "
                "VALUES (:u, :o, 'active')"
            ),
            {"u": other_user, "o": other_org},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Other Course', 'draft')"
            ),
            {
                "id": other_course,
                "org": other_org,
                "u": other_user,
                "slug": f"other-course-{other_course}",
            },
        )
    yield {
        "organization_id": other_org,
        "org_unit_id": other_unit,
        "user_id": other_user,
        "course_id": other_course,
    }
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": other_course})
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE user_id = :u"),
            {"u": other_user},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": other_user})
        await conn.execute(text("DELETE FROM org_units WHERE id = :id"), {"id": other_unit})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": other_org})


@pytest_asyncio.fixture
async def manager_membership(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[None]:
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "INSERT INTO organization_memberships (user_id, organization_id, status) "
                "VALUES (:u, :o, 'active') "
                "ON CONFLICT DO NOTHING "
                "RETURNING id"
            ),
            {"u": seeded_users.manager_id, "o": seeded_users.organization_id},
        )
        rows = result.all()
    yield None
    if rows:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM organization_memberships WHERE id = :id"),
                {"id": rows[0][0]},
            )


@pytest_asyncio.fixture
async def failed_job(engine: AsyncEngine) -> AsyncIterator[uuid.UUID]:
    job_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO processing_jobs "
                "(id, entity_type, entity_id, job_type, status, progress_percent, "
                " error_message, retry_count) "
                "VALUES (:id, 'material_version', :eid, 'parse_document', 'failed', 50, "
                "        'boom', 1)"
            ),
            {"id": job_id, "eid": entity_id},
        )
    yield job_id
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM processing_jobs WHERE id = :id"), {"id": job_id})


async def test_admin_global_stats(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    extra_org: dict[str, uuid.UUID],
) -> None:
    del extra_org
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get("/api/v1/admin/stats/overview", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_courses"] >= 2
    assert body["total_users"] >= 2


async def test_manager_org_scoped_stats(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    extra_org: dict[str, uuid.UUID],
    manager_membership: None,
) -> None:
    del manager_membership, extra_org
    token, _ = await _bearer(engine, seeded_users.manager_id)
    try:
        resp = await client.get("/api/v1/admin/stats/overview", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.manager_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    async with engine.begin() as conn:
        scoped = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM courses WHERE organization_id = :o AND deleted_at IS NULL"
                ),
                {"o": seeded_users.organization_id},
            )
        ).scalar_one()
    assert body["total_courses"] == scoped

    admin_token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        admin_resp = await client.get("/api/v1/admin/stats/overview", headers=_auth(admin_token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert admin_resp.status_code == 200, admin_resp.text
    assert admin_resp.json()["total_courses"] > body["total_courses"]


_DASHBOARD_INT_FIELDS = (
    "jobs_failed_7d",
    "jobs_total_7d",
    "queue_depth",
    "failed_ai_calls_30d",
    "active_users_today",
    "active_users_7d",
    "total_users",
    "quiz_sessions_completed_7d",
    "interview_sessions_7d",
    "materials_ingested_7d",
    "materials_stuck_processing",
    "published_quizzes_missing_texp",
    "interview_configs_no_reviewed_questions",
    "orgs_inactive_30d",
)
_DASHBOARD_FLOAT_FIELDS = (
    "job_failure_rate_pct",
    "spend_7d_usd",
    "spend_prev_7d_usd",
    "projected_month_end_usd",
    "top_cost_driver_usd",
    "interview_pass_rate_pct",
)


async def test_admin_dashboard(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    extra_org: dict[str, uuid.UUID],
) -> None:
    del extra_org
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get("/api/v1/admin/stats/dashboard", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for field in _DASHBOARD_INT_FIELDS:
        assert isinstance(body[field], int), field
        assert body[field] >= 0, field
    for field in _DASHBOARD_FLOAT_FIELDS:
        assert isinstance(body[field], int | float), field
        assert body[field] >= 0, field
    assert body["top_cost_driver"] is None or isinstance(body["top_cost_driver"], str)
    assert body["slowest_model"] is None or isinstance(body["slowest_model"], str)
    assert isinstance(body["slowest_model_p95_ms"], int)
    # rate is a percentage derived from the two job counters
    assert 0.0 <= body["job_failure_rate_pct"] <= 100.0
    assert 0.0 <= body["interview_pass_rate_pct"] <= 100.0
    assert body["jobs_failed_7d"] <= body["jobs_total_7d"]


async def test_manager_dashboard_is_org_scoped(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    extra_org: dict[str, uuid.UUID],
    manager_membership: None,
) -> None:
    del manager_membership, extra_org
    token, _ = await _bearer(engine, seeded_users.manager_id)
    try:
        resp = await client.get("/api/v1/admin/stats/dashboard", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.manager_id)
    assert resp.status_code == 200, resp.text
    scoped_users = resp.json()["total_users"]

    admin_token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        admin_resp = await client.get("/api/v1/admin/stats/dashboard", headers=_auth(admin_token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert admin_resp.status_code == 200, admin_resp.text
    # org-scoped caller must never see more users than the global view
    assert scoped_users <= admin_resp.json()["total_users"]


async def test_dashboard_student_token_403(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token, _ = await _bearer(engine, seeded_users.student_id)
    try:
        resp = await client.get("/api/v1/admin/stats/dashboard", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.student_id)
    assert resp.status_code == 403, resp.text


async def test_student_token_403(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token, _ = await _bearer(engine, seeded_users.student_id)
    try:
        resp = await client.get("/api/v1/admin/stats/overview", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.student_id)
    assert resp.status_code == 403


async def test_retry_failed_job(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    failed_job: uuid.UUID,
    app: tuple[FastAPI, AsyncMock],
) -> None:
    _, arq_pool = app
    arq_pool.enqueue_job.reset_mock()
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.post(
            f"/api/v1/admin/processing/jobs/{failed_job}/retry",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["retry_count"] == 2
    assert body["error_message"] is None
    arq_pool.enqueue_job.assert_called_once()
    args, _ = arq_pool.enqueue_job.call_args
    assert args[0] == "parse_document"


async def test_list_processing_jobs_smoke(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    failed_job: uuid.UUID,
) -> None:
    """Regression: ``GET /admin/processing/jobs`` must succeed with and without
    a status filter.

    The bug was ``psycopg.errors.AmbiguousParameter`` on ``:status IS NULL OR
    pj.status = :status`` when ``status`` was omitted, because PostgreSQL
    could not infer the parameter type. The SQL now wraps it in
    ``CAST(:status AS text)``; this test exercises both branches.
    """
    del failed_job  # ensures at least one row exists
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = quote_plus((datetime.now(tz=UTC) - timedelta(days=7)).isoformat())
    try:
        no_filter = await client.get(
            f"/api/v1/admin/processing/jobs?since={since}&limit=50",
            headers=_auth(token),
        )
        with_filter = await client.get(
            f"/api/v1/admin/processing/jobs?since={since}&status=failed&limit=50",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert no_filter.status_code == 200, no_filter.text
    assert with_filter.status_code == 200, with_filter.text
    assert isinstance(no_filter.json(), list)
    assert isinstance(with_filter.json(), list)
    assert all(row["status"] == "failed" for row in with_filter.json())


async def test_processing_summary_matches_unfiltered_list(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    failed_job: uuid.UUID,
) -> None:
    """``GET /admin/processing/summary`` counts must equal the per-status
    counts of the UNFILTERED jobs list over the same ``since`` window.

    Regression: the page derived its status-tab badges from the jobs list
    while that list was filtered by the selected status — so picking one tab
    collapsed every other tab's count to zero. The summary endpoint is the
    badges' source; it must stay window-wide regardless of any status filter.
    """
    del failed_job  # ensures at least one row exists
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = quote_plus((datetime.now(tz=UTC) - timedelta(days=7)).isoformat())
    try:
        summary_res = await client.get(
            f"/api/v1/admin/processing/summary?since={since}",
            headers=_auth(token),
        )
        listing_res = await client.get(
            f"/api/v1/admin/processing/jobs?since={since}&limit=500",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert summary_res.status_code == 200, summary_res.text
    assert listing_res.status_code == 200, listing_res.text
    body = summary_res.json()
    rows = listing_res.json()
    assert body["total"] == len(rows)
    for status in ("pending", "running", "completed", "failed", "cancelled"):
        assert body[status] == sum(1 for r in rows if r["status"] == status), status


async def test_list_admin_users_smoke(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Regression: ``GET /admin/users`` must succeed with no filters and with
    each filter set, covering the same ``:param IS NULL OR ...`` cast bug as
    the processing-jobs query.
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        no_filter = await client.get(
            "/api/v1/admin/users?limit=10",
            headers=_auth(token),
        )
        status_filter = await client.get(
            "/api/v1/admin/users?status=active&limit=10",
            headers=_auth(token),
        )
        role_filter = await client.get(
            "/api/v1/admin/users?role_code=teacher&limit=10",
            headers=_auth(token),
        )
        q_filter = await client.get(
            "/api/v1/admin/users?q=admin&limit=10",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert no_filter.status_code == 200, no_filter.text
    assert status_filter.status_code == 200, status_filter.text
    assert role_filter.status_code == 200, role_filter.text
    no_filter_payload = no_filter.json()
    status_filter_payload = status_filter.json()
    role_filter_payload = role_filter.json()
    assert isinstance(no_filter_payload, dict)
    assert isinstance(no_filter_payload["items"], list)
    assert "next_cursor" in no_filter_payload
    assert isinstance(status_filter_payload["items"], list)
    assert isinstance(role_filter_payload["items"], list)
    assert q_filter.status_code == 200, q_filter.text
    q_payload = q_filter.json()
    assert isinstance(q_payload, dict)
    q_rows = q_payload["items"]
    assert isinstance(q_rows, list)
    # Each row must expose ``display_name`` (nullable) for the search UI.
    for row in q_rows:
        assert "display_name" in row


async def test_disable_revokes_sessions(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    target = uuid.uuid4()
    target_session = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": target, "email": f"victim-{target}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:s, :u, :h, NOW() + interval '1 hour')"
            ),
            {"s": target_session, "u": target, "h": hash_secret(generate_token())},
        )
    admin_token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.post(
            f"/api/v1/admin/users/{target}/disable", headers=_auth(admin_token)
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "inactive"
        assert body["revoked_session_count"] == 1
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT u.status, s.revoked_at "
                        "FROM users u JOIN auth_sessions s ON s.user_id = u.id "
                        "WHERE u.id = :u"
                    ),
                    {"u": target},
                )
            ).one()
        assert row[0] == "inactive"
        assert row[1] is not None
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE user_id = :u"),
                {"u": target},
            )
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": target})
        await _purge_sessions(engine, seeded_users.admin_id)


async def test_enable_restores_status(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    target = uuid.uuid4()
    revoked_session = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'inactive')"),
            {"id": target, "email": f"benched-{target}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at, revoked_at) "
                "VALUES (:s, :u, :h, NOW() + interval '1 hour', NOW())"
            ),
            {"s": revoked_session, "u": target, "h": hash_secret(generate_token())},
        )
    admin_token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.post(f"/api/v1/admin/users/{target}/enable", headers=_auth(admin_token))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "active"
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT u.status, s.revoked_at "
                        "FROM users u JOIN auth_sessions s ON s.user_id = u.id "
                        "WHERE u.id = :u"
                    ),
                    {"u": target},
                )
            ).one()
        assert row[0] == "active"
        assert row[1] is not None
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE user_id = :u"),
                {"u": target},
            )
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": target})
        await _purge_sessions(engine, seeded_users.admin_id)


async def test_audit_role_changes_search(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = quote_plus((datetime.now(tz=UTC) - timedelta(days=365)).isoformat())
    try:
        resp = await client.get(
            f"/api/v1/admin/audit/role-changes?since={since}&limit=10",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) >= 5
    sample = body[0]
    assert {"role_code", "scope_kind", "user_id"}.issubset(sample.keys())


async def test_data_changes_lookup(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            f"/api/v1/admin/audit/data-changes?table=courses&entity_id={seeded_users.course_id}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entity_id"] == str(seeded_users.course_id)
    assert body["organization_id"] == str(seeded_users.organization_id)


async def test_data_changes_lookup_users(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """FR-6.7 — data-changes now covers the users table (was courses-only)."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            f"/api/v1/admin/audit/data-changes?table=users&entity_id={seeded_users.admin_id}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entity_id"] == str(seeded_users.admin_id)
    # Users carry no owning org / audit-by columns — surfaced as null uniformly.
    assert body["organization_id"] is None
    assert body["created_by"] is None
    # primary_email rides along via the extra="allow" passthrough.
    assert "primary_email" in body


async def test_data_changes_unsupported_table_400(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """An unknown table 400s before touching the DB (FR-6.7 guard)."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            f"/api/v1/admin/audit/data-changes?table=quizzes&entity_id={seeded_users.course_id}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"] == "unsupported_table"


async def test_data_changes_not_found_404(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """A supported table with a missing PK 404s (FR-6.7)."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    missing = uuid.uuid4()
    try:
        resp = await client.get(
            f"/api/v1/admin/audit/data-changes?table=materials&entity_id={missing}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"]["resource"] == "materials"


async def test_http_audit_endpoint_503_when_table_absent(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    async with engine.begin() as conn:
        exists = (
            await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'http_audit_log' LIMIT 1"
                )
            )
        ).first()
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = quote_plus((datetime.now(tz=UTC) - timedelta(days=1)).isoformat())
    try:
        resp = await client.get(
            f"/api/v1/admin/audit/http?since={since}&limit=10",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    if exists is None:
        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body["detail"]["error"] == "audit_log_unavailable"
    else:
        assert resp.status_code == 200, resp.text


async def test_unbounded_scan_requires_since(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get("/api/v1/admin/audit/http?limit=10", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 422


def test_router_metadata() -> None:
    assert stats_router.prefix == "/admin/stats"
    assert audit_router.prefix == "/admin/audit"
    assert processing_router.prefix == "/admin/processing"
    assert users_router.prefix == "/admin/users"
    expected = {
        "/admin/stats/overview",
        "/admin/stats/active-users",
        "/admin/stats/content",
        "/admin/stats/health",
        "/admin/stats/dashboard",
    }
    actual = {route.path for route in stats_router.routes}  # type: ignore[attr-defined]
    assert expected.issubset(actual)
