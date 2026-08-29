from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote_plus

import httpx
import pytest
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
    "window_days",
    "jobs_terminal_window",
    "jobs_failed_window",
    "jobs_terminal_prev_window",
    "jobs_failed_prev_window",
    "queue_depth",
    "queue_pending",
    "queue_running",
    "requests_window",
    "requests_5xx_window",
    "requests_4xx_window",
    "tokens_window",
    "ai_calls_window",
    "failed_ai_calls_window",
    "active_users_today",
    "active_users_window",
    "total_users",
    "materials_ingested_window",
    "orgs_total",
    "orgs_inactive_30d",
)
_DASHBOARD_FLOAT_FIELDS = (
    "spend_window_usd",
    "spend_prev_window_usd",
    "projected_month_end_usd",
    "top_cost_driver_usd",
)
# Every rate is nullable by contract: None means the denominator was empty and
# the client must render "No data" rather than 0% (PRD section 5).
_DASHBOARD_NULLABLE_RATE_FIELDS = (
    "job_failure_rate_pct",
    "job_failure_rate_prev_pct",
    "api_error_rate_pct",
    "api_client_error_rate_pct",
    "ai_failure_rate_pct",
)
_DASHBOARD_NULLABLE_INT_FIELDS = (
    "queue_oldest_age_seconds",
    "api_p50_latency_ms",
    "api_p95_latency_ms",
)
_DASHBOARD_SCOPE_FIELDS = (
    "usage_scope",
    "tenant_scope",
    "job_scope",
    "cost_scope",
    "api_scope",
)
# Metrics the PRD moved off the operator dashboard: academic signals (ADM-003)
# and the job counters that used to disagree with the processing page
# (ADM-004). Their absence is the requirement, so it is asserted.
_DASHBOARD_REMOVED_FIELDS = (
    "interview_pass_rate_pct",
    "interview_evaluated_7d",
    "interview_students_7d",
    "interview_sessions_7d",
    "quiz_sessions_completed_7d",
    "published_quizzes_missing_texp",
    "interview_configs_no_reviewed_questions",
    "jobs_failed_7d",
    "jobs_total_7d",
    "materials_stuck_processing",
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
    for field in _DASHBOARD_NULLABLE_RATE_FIELDS:
        value = body[field]
        assert value is None or 0.0 <= value <= 100.0, field
    for field in _DASHBOARD_NULLABLE_INT_FIELDS:
        assert body[field] is None or body[field] >= 0, field
    for field in _DASHBOARD_SCOPE_FIELDS:
        assert body[field] in {"global", "organization"}, field
    for field in _DASHBOARD_REMOVED_FIELDS:
        assert field not in body, field
    assert body["top_cost_driver"] is None or isinstance(body["top_cost_driver"], str)
    assert body["slowest_model"] is None or isinstance(body["slowest_model"], str)
    assert isinstance(body["slowest_model_p95_ms"], int)
    assert isinstance(body["as_of"], str)
    assert body["window_days"] == 7
    # A caller with system.administer and no filter is global everywhere.
    assert body["organization_id"] is None
    assert all(body[f] == "global" for f in _DASHBOARD_SCOPE_FIELDS)
    # Failures are a subset of the terminal population they are divided by.
    assert body["jobs_failed_window"] <= body["jobs_terminal_window"]
    assert body["requests_5xx_window"] <= body["requests_window"]
    assert body["failed_ai_calls_window"] <= body["ai_calls_window"]
    assert body["queue_pending"] + body["queue_running"] == body["queue_depth"]


async def test_dashboard_rate_is_null_not_zero_without_data(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """PRD section 5: an empty window has no rate, and must not report 0%.

    A 1-day window over a freshly seeded database has no terminal jobs and no
    AI calls, so both rates come back null with a zero denominator beside them
    -- the pair the client needs in order to render "No data".
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/stats/dashboard?window_days=1", headers=_auth(token)
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["window_days"] == 1
    if body["jobs_terminal_window"] == 0:
        assert body["job_failure_rate_pct"] is None
    if body["ai_calls_window"] == 0:
        assert body["ai_failure_rate_pct"] is None
    if body["queue_depth"] == 0:
        assert body["queue_oldest_age_seconds"] is None


async def test_dashboard_window_days_is_validated(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        too_wide = await client.get(
            "/api/v1/admin/stats/dashboard?window_days=91", headers=_auth(token)
        )
        too_narrow = await client.get(
            "/api/v1/admin/stats/dashboard?window_days=0", headers=_auth(token)
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert too_wide.status_code == 422, too_wide.text
    assert too_narrow.status_code == 422, too_narrow.text


async def test_dashboard_org_filter_honoured_for_admin_only(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    extra_org: dict[str, uuid.UUID],
    manager_membership: None,
) -> None:
    """ADM-005: an IT admin may narrow to one tenant; a manager may not.

    The manager is already pinned to their own organization, so passing some
    other org's id must not move their numbers -- that would be a cross-tenant
    read dressed up as a filter.
    """
    del manager_membership
    other_org = extra_org["organization_id"]

    admin_token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        scoped = await client.get(
            "/api/v1/admin/stats/dashboard"
            f"?organization_id={seeded_users.organization_id}",
            headers=_auth(admin_token),
        )
        unscoped = await client.get(
            "/api/v1/admin/stats/dashboard", headers=_auth(admin_token)
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert scoped.status_code == 200, scoped.text
    assert unscoped.status_code == 200, unscoped.text
    assert scoped.json()["usage_scope"] == "organization"
    assert scoped.json()["total_users"] <= unscoped.json()["total_users"]
    # Job / cost / API families have no organization edge and say so even when
    # the caller filtered.
    assert scoped.json()["job_scope"] == "global"
    assert scoped.json()["cost_scope"] == "global"
    assert scoped.json()["api_scope"] == "global"

    manager_token, _ = await _bearer(engine, seeded_users.manager_id)
    try:
        pinned = await client.get(
            f"/api/v1/admin/stats/dashboard?organization_id={other_org}",
            headers=_auth(manager_token),
        )
        default = await client.get(
            "/api/v1/admin/stats/dashboard", headers=_auth(manager_token)
        )
    finally:
        await _purge_sessions(engine, seeded_users.manager_id)
    assert pinned.status_code == 200, pinned.text
    assert default.status_code == 200, default.text
    assert pinned.json()["organization_id"] == default.json()["organization_id"]
    assert pinned.json()["total_users"] == default.json()["total_users"]


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


async def test_list_processing_jobs_honors_until_bound(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    failed_job: uuid.UUID,
) -> None:
    """``until`` (custom-range upper bound) narrows both ``/jobs`` and
    ``/summary``; omitting it keeps the old lower-bound-only behaviour.
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = quote_plus((datetime.now(tz=UTC) - timedelta(days=7)).isoformat())
    yesterday = quote_plus((datetime.now(tz=UTC) - timedelta(days=1)).isoformat())
    tomorrow = quote_plus((datetime.now(tz=UTC) + timedelta(days=1)).isoformat())
    try:
        before = await client.get(
            f"/api/v1/admin/processing/jobs?since={since}&until={yesterday}&limit=500",
            headers=_auth(token),
        )
        after = await client.get(
            f"/api/v1/admin/processing/jobs?since={since}&until={tomorrow}&limit=500",
            headers=_auth(token),
        )
        summary_before = await client.get(
            f"/api/v1/admin/processing/summary?since={since}&until={yesterday}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert before.status_code == 200, before.text
    assert after.status_code == 200, after.text
    assert summary_before.status_code == 200, summary_before.text
    # The seeded job has updated_at = now, so a yesterday upper bound must
    # exclude it while a tomorrow bound keeps it. (Other tests may leave
    # older jobs behind, so assert on the seed's own id, not on emptiness.)
    before_ids = {row["id"] for row in before.json()}
    after_ids = {row["id"] for row in after.json()}
    assert str(failed_job) not in before_ids
    assert str(failed_job) in after_ids


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


async def test_audit_role_changes_honors_until_bound(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """``until`` (range upper bound) narrows the scan window — a bound in
    the deep past must yield nothing, an open window keeps everything."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        open_ = await client.get(
            "/api/v1/admin/audit/role-changes"
            f"?since={quote_plus((datetime.now(tz=UTC) - timedelta(days=365)).isoformat())}&limit=10",
            headers=_auth(token),
        )
        empty = await client.get(
            "/api/v1/admin/audit/role-changes"
            "?since=2000-01-01T00:00:00Z"
            f"&until={quote_plus((datetime.now(tz=UTC) - timedelta(days=365)).isoformat())}&limit=10",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert open_.status_code == 200, open_.text
    assert empty.status_code == 200, empty.text
    assert len(open_.json()) >= 1
    # Every assignment is newer than a year ago, so the (since=2000, until=now-365d)
    # window is empty — proves the exclusive upper bound is actually applied.
    assert empty.json() == []


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


async def test_data_changes_list(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """FR-6.7 — recent-changes list per table, newest first."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/audit/data-changes/list?table=courses&since=2000-01-01T00:00:00Z&limit=50",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    assert any(row["entity_id"] == str(seeded_users.course_id) for row in body)
    # Newest first: updated_at must be non-increasing.
    stamps = [row["updated_at"] for row in body]
    assert stamps == sorted(stamps, reverse=True)


async def test_data_changes_list_unsupported_table_400(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """The list endpoint validates ``table`` the same way as the lookup."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/audit/data-changes/list?table=quizzes&since=2000-01-01T00:00:00Z",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"] == "unsupported_table"


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


async def test_active_users_trend(
    engine: AsyncEngine,
    client: httpx.AsyncClient,
    seeded_users: SeededUsers,
) -> None:
    """Trend returns one point per day with distinct-login counts.

    Every login creates an auth_sessions row, so the series counts distinct
    users whose session was created that calendar day. The admin bearer
    session lands today; two more sessions are pinned 3 days back.
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    three_days_ago = datetime.now(tz=UTC) - timedelta(days=3)
    for uid in (seeded_users.student_id, seeded_users.manager_id):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO auth_sessions "
                    "(id, user_id, refresh_token_hash, expires_at, created_at) "
                    "VALUES (:id, :uid, :h, :exp, :created)"
                ),
                {
                    "id": uuid.uuid4(),
                    "uid": uid,
                    "h": hash_secret(generate_token()),
                    "exp": three_days_ago + timedelta(hours=1),
                    "created": three_days_ago,
                },
            )
    try:
        resp = await client.get(
            "/api/v1/admin/stats/active-users/trend?days=7",
            headers=_auth(token),
        )
    finally:
        for uid in (
            seeded_users.admin_id,
            seeded_users.student_id,
            seeded_users.manager_id,
        ):
            await _purge_sessions(engine, uid)
    assert resp.status_code == 200, resp.text
    points = {p["date"]: p["count"] for p in resp.json()["points"]}
    assert len(points) == 7
    # Dates come from ::date casts in the DB session timezone; resolve the
    # expected day labels the same way instead of assuming UTC.
    async with engine.begin() as conn:
        labels = (
            await conn.execute(
                text(
                    "SELECT now()::date AS today, "
                    "(now() - interval '3 days')::date AS d3"
                )
            )
        ).mappings().one()
    assert points[str(labels["today"])] >= 1  # the admin bearer session
    assert points[str(labels["d3"])] == 2  # student + manager pinned sessions


async def test_api_latency_trend(
    engine: AsyncEngine,
    client: httpx.AsyncClient,
    seeded_users: SeededUsers,
) -> None:
    """Latency trend: per-day p50/p95 from http_audit_log, zero days gap-free.

    Rows are pinned 3 days back — the audit middleware only writes *now*, so
    that day's set is exactly the injected one and the percentiles are
    deterministic: latencies [10, 30] → p50 = 20, p95 = 10 + 0.95·20 = 29
    (percentile_cont linear interpolation).
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    three_days_ago = datetime.now(tz=UTC) - timedelta(days=3)
    inserted_ids: list[str] = []
    async with engine.begin() as conn:
        # Isolate the window: the suite shares one Postgres across runs, so
        # http_audit_log carries every request logged since local midnight
        # (middleware writes *now*). Purging only past days left today's
        # accumulation intact, which made the "today == 1 row" assertion
        # depend on how many requests earlier tests made. Clean the whole
        # table — tests run serially and every test creates its own rows.
        await conn.execute(text("DELETE FROM http_audit_log"))
        for offset_min, latency in ((10, 10), (40, 30)):
            row_id = str(uuid.uuid4())
            inserted_ids.append(row_id)
            await conn.execute(
                text(
                    "INSERT INTO http_audit_log "
                    "(id, request_id, method, path, status_code, latency_ms, "
                    "created_at) "
                    "VALUES (:id, :rid, 'GET', '/api/v1/test', 200, :lat, "
                    ":created)"
                ),
                {
                    "id": row_id,
                    "rid": str(uuid.uuid4()),
                    "lat": latency,
                    "created": three_days_ago + timedelta(minutes=offset_min),
                },
            )
        # One pinned row today as well. The endpoint's snapshot runs before
        # the middleware logs the trend request itself, so "today" must be
        # seeded explicitly to have deterministic content.
        today_row = str(uuid.uuid4())
        inserted_ids.append(today_row)
        await conn.execute(
            text(
                "INSERT INTO http_audit_log "
                "(id, request_id, method, path, status_code, latency_ms, "
                "created_at) "
                "VALUES (:id, :rid, 'GET', '/api/v1/test', 200, 150, now())"
            ),
            {
                "id": today_row,
                "rid": str(uuid.uuid4()),
            },
        )
    try:
        resp = await client.get(
            "/api/v1/admin/stats/latency/trend?days=7",
            headers=_auth(token),
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM http_audit_log WHERE id = ANY(:ids)"),
                {"ids": inserted_ids},
            )
    assert resp.status_code == 200, resp.text
    points = {p["day"]: p for p in resp.json()["points"]}
    assert len(points) == 7  # gap-free: one point per day in the window
    async with engine.begin() as conn:
        labels = (
            await conn.execute(
                text(
                    "SELECT now()::date AS today, "
                    "(now() - interval '3 days')::date AS d3"
                )
            )
        ).mappings().one()
    pinned = points[str(labels["d3"])]
    assert pinned["requests_total"] == 2
    assert pinned["p50_latency_ms"] == 20
    assert pinned["p95_latency_ms"] == 29  # rounded from 28.999...
    # Today carries the single seeded row (the middleware logs the trend
    # request itself only after the response snapshot was taken).
    # Today's bucket is NOT asserted exactly. The audit middleware writes a
    # row for every request the suite makes, and the purge above deliberately
    # spares today ("today's rows belong to concurrent tests' middleware
    # writes"), so the seeded row shares the day with however many other
    # requests happened to run. Asserting == 1 passed in isolation and failed
    # in a full run. The 3-days-ago bucket above is the deterministic one.
    today = points[str(labels["today"])]
    assert today["requests_total"] >= 1
    assert today["p95_latency_ms"] is not None


async def test_trends_follow_an_explicit_range(
    engine: AsyncEngine,
    client: httpx.AsyncClient,
    seeded_users: SeededUsers,
) -> None:
    """`from`/`to` drive both trend charts, and a past range stays in the past.

    This is the property a day count cannot express: a range that ENDS before
    today. Asking for a 3-day window that finished two days ago must return
    exactly those 3 days -- not "the last 3 days" running up to now, which is
    what the old `days=` parameter did and why the chart disagreed with the
    KPIs computed over the page's own range.
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    async with engine.begin() as conn:
        today = (
            await conn.execute(text("SELECT now()::date AS d"))
        ).mappings().one()["d"]
    window_to = today - timedelta(days=2)
    window_from = window_to - timedelta(days=2)  # 3 inclusive days

    try:
        latency = await client.get(
            f"/api/v1/admin/stats/latency/trend?from={window_from}&to={window_to}",
            headers=_auth(token),
        )
        active = await client.get(
            "/api/v1/admin/stats/active-users/trend"
            f"?from={window_from}&to={window_to}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)

    assert latency.status_code == 200, latency.text
    assert active.status_code == 200, active.text

    latency_days = [p["day"] for p in latency.json()["points"]]
    active_days = [p["date"] for p in active.json()["points"]]

    expected = [str(window_from + timedelta(days=i)) for i in range(3)]
    # Exactly the picked span: inclusive of `to`, and nothing after it.
    assert latency_days == expected
    assert active_days == expected
    assert str(today) not in latency_days
    assert str(today) not in active_days


async def test_trend_range_rejects_a_malformed_window(
    engine: AsyncEngine,
    client: httpx.AsyncClient,
    seeded_users: SeededUsers,
) -> None:
    """Both trends validate the range the same way the rollup does.

    The three endpoints render one window three ways on a single page. If a
    trend accepted a range the rollup rejected, the page could draw a chart
    over a span its own KPIs refused to compute.
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    async with engine.begin() as conn:
        today = (
            await conn.execute(text("SELECT now()::date AS d"))
        ).mappings().one()["d"]
    try:
        for path in (
            "/api/v1/admin/stats/latency/trend",
            "/api/v1/admin/stats/active-users/trend",
            "/api/v1/admin/stats/dashboard",
        ):
            # 'from' without 'to'
            half = await client.get(f"{path}?from={today}", headers=_auth(token))
            assert half.status_code == 422, f"{path}: {half.text}"
            # reversed range
            backwards = await client.get(
                f"{path}?from={today}&to={today - timedelta(days=3)}",
                headers=_auth(token),
            )
            assert backwards.status_code == 422, f"{path}: {backwards.text}"
            # end in the future
            future = await client.get(
                f"{path}?from={today}&to={today + timedelta(days=1)}",
                headers=_auth(token),
            )
            assert future.status_code == 422, f"{path}: {future.text}"
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)


async def test_dashboard_custom_range(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Range windows count exactly the rows in [from, to], inclusive.

    http_audit_log + ai_model_calls rows are pinned inside and outside a
    fixed two-day-old window; every windowed figure must reflect only the
    in-range rows, and the envelope echoes window_from / window_to verbatim
    so the UI label == the rows counted. The middleware's own request rows
    land today, which is outside the window, so the counts are exact.
    """
    token, _ = await _bearer(engine, seeded_users.admin_id)
    # Five days back: clear of every other suite's seeds (the AI-costs suite
    # pins calls ~2 days back for its prev-window shape, and the latency
    # suite uses day-3) so the in-window counts are exactly ours.
    d0 = datetime.now(tz=UTC).date() - timedelta(days=5)
    inside_ts = datetime.combine(d0, time(hour=1), tzinfo=UTC)
    before_ts = datetime.combine(d0 - timedelta(days=10), time(hour=1), tzinfo=UTC)
    audit_ids: list[str] = []
    call_ids: list[str] = []
    async with engine.begin() as conn:
        for created_at, latency in (
            (before_ts, 10),
            (inside_ts, 100),
            (inside_ts + timedelta(hours=1), 300),
        ):
            row_id = str(uuid.uuid4())
            audit_ids.append(row_id)
            await conn.execute(
                text(
                    "INSERT INTO http_audit_log "
                    "(id, request_id, method, path, status_code, latency_ms, "
                    "created_at) "
                    "VALUES (:id, :rid, 'GET', '/api/v1/test', 200, :lat, :ts)"
                ),
                {
                    "id": row_id,
                    "rid": row_id,
                    "lat": latency,
                    "ts": created_at,
                },
            )
        for i, (created_at, cost) in enumerate(
            ((inside_ts, 4.0), (inside_ts + timedelta(hours=1), 6.0), (before_ts, 99.0))
        ):
            cid = str(uuid.uuid4())
            call_ids.append(cid)
            await conn.execute(
                text(
                    "INSERT INTO ai_model_calls "
                    "(id, role, stage_name, model_name, total_tokens, "
                    "estimated_cost_usd, latency_ms, status, called_at) "
                    "VALUES (:id, 'ingest', 'store', 'test-model', 10, "
                    "CAST(:cost AS numeric), 5, 'success', :ts)"
                ),
                {"id": cid, "cost": cost, "ts": created_at},
            )
    try:
        resp = await client.get(
            f"/api/v1/admin/stats/dashboard?from={d0.isoformat()}&to={d0.isoformat()}",
            headers=_auth(token),
        )
    finally:
        async with engine.begin() as conn:
            if audit_ids:
                await conn.execute(
                    text("DELETE FROM http_audit_log WHERE id = ANY(:ids)"),
                    {"ids": audit_ids},
                )
            if call_ids:
                await conn.execute(
                    text("DELETE FROM ai_model_calls WHERE id = ANY(:ids)"),
                    {"ids": call_ids},
                )
            await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Envelope echoes the exact dates the picker applied.
    assert body["window_from"] == d0.isoformat()
    assert body["window_to"] == d0.isoformat()
    assert body["window_days"] == 7  # legacy relative shape untouched
    # Only the two in-range rows count; percentiles over [100, 300].
    assert body["requests_window"] == 2
    assert body["api_p50_latency_ms"] == 200
    assert body["api_p95_latency_ms"] == 290
    assert body["ai_calls_window"] == 2
    assert body["spend_window_usd"] == 10.0


@pytest.mark.parametrize(
    ("params", "needle"),
    [
        ("from=2026-08-01", "together"),
        ("from=2026-08-29&to=2026-08-01", "must not be after"),
        ("from=2026-08-01&to=2030-01-01", "after today"),
        ("from=2020-01-01&to=2022-01-01", "366"),
    ],
)
async def test_dashboard_custom_range_validation(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    params: str,
    needle: str,
) -> None:
    """Range-mode rejects malformed pairs instead of silently misreporting."""
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            f"/api/v1/admin/stats/dashboard?{params}", headers=_auth(token)
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 422, resp.text
    assert needle in resp.json()["detail"]["message"]


def test_router_metadata() -> None:
    assert stats_router.prefix == "/admin/stats"
    assert audit_router.prefix == "/admin/audit"
    assert processing_router.prefix == "/admin/processing"
    assert users_router.prefix == "/admin/users"
    expected = {
        "/admin/stats/overview",
        "/admin/stats/active-users",
        "/admin/stats/active-users/trend",
        "/admin/stats/latency/trend",
        "/admin/stats/content",
        "/admin/stats/dashboard",
    }
    actual = {route.path for route in stats_router.routes}  # type: ignore[attr-defined]
    assert expected.issubset(actual)
    # ``/admin/stats/health`` was retired with the Operations merge: it was a
    # fourth, incompatible definition of "failed jobs" (processing_jobs only,
    # window-filtered in-flight count). Its absence is the requirement.
    assert "/admin/stats/health" not in actual
