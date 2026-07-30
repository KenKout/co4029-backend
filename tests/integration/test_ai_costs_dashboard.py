"""Integration tests for the AI cost dashboard (T0.27)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

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

import abridgeai.ai.models  # noqa: F401  -- register ai_model_calls / processing_jobs / generation_runs
import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.admin.routers import ai_costs_router


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
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(ai_costs_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
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


class _Seed:
    def __init__(self) -> None:
        self.run_quiz: uuid.UUID = uuid.uuid4()
        self.run_kg: uuid.UUID = uuid.uuid4()
        self.processing_quiz: uuid.UUID = uuid.uuid4()
        self.pipeline_a: uuid.UUID = uuid.uuid4()
        self.pipeline_b: uuid.UUID = uuid.uuid4()
        self.call_ids: list[uuid.UUID] = []
        self.requested_by: uuid.UUID | None = None
        self.course_id: uuid.UUID | None = None


@pytest_asyncio.fixture
async def cost_seed(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[_Seed]:
    """Seed ai_model_calls covering 3 days, 2 roles, 2 stages, 2 pipelines."""
    seed = _Seed()
    seed.requested_by = seeded_users.teacher_id
    seed.course_id = seeded_users.course_id

    now = datetime.now(tz=UTC)
    today = now.replace(hour=12, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO generation_runs "
                "(id, generation_type, source_scope_kind, course_id, requested_by, status) "
                "VALUES (:id, 'quiz', 'course', :cid, :uid, 'completed')"
            ),
            {"id": seed.run_quiz, "cid": seed.course_id, "uid": seed.requested_by},
        )
        await conn.execute(
            text(
                "INSERT INTO generation_runs "
                "(id, generation_type, source_scope_kind, course_id, requested_by, status) "
                "VALUES (:id, 'knowledge_graph', 'course', :cid, :uid, 'completed')"
            ),
            {"id": seed.run_kg, "cid": seed.course_id, "uid": seed.requested_by},
        )
        await conn.execute(
            text(
                "INSERT INTO processing_jobs "
                "(id, entity_type, entity_id, job_type, status, progress_percent) "
                "VALUES (:id, 'generation_run', :gid, 'generate_quiz', 'completed', 100)"
            ),
            {"id": seed.processing_quiz, "gid": seed.run_quiz},
        )

        rows = [
            (
                seed.run_quiz,
                None,
                seed.pipeline_a,
                "generator_main",
                "tier_a",
                "generation",
                1000,
                "0.012",
                today,
                "gpt-4o",
                250,
            ),
            (
                seed.run_quiz,
                None,
                seed.pipeline_a,
                "generator_main",
                "tier_a",
                "generation",
                800,
                "0.010",
                today - timedelta(hours=1),
                "gpt-4o",
                230,
            ),
            (
                None,
                seed.processing_quiz,
                seed.pipeline_a,
                "validator_main",
                "tier_b",
                "validation",
                400,
                "0.005",
                today - timedelta(hours=2),
                "gpt-4o-mini",
                110,
            ),
            (
                seed.run_quiz,
                None,
                seed.pipeline_a,
                "validator_main",
                "tier_b",
                "validation",
                350,
                "0.004",
                today - timedelta(hours=3),
                "gpt-4o-mini",
                100,
            ),
            (
                seed.run_kg,
                None,
                seed.pipeline_b,
                "extractor_kg",
                "tier_a",
                "extraction",
                1500,
                "0.025",
                yesterday,
                "gpt-4o",
                600,
            ),
            (
                seed.run_kg,
                None,
                seed.pipeline_b,
                "extractor_kg",
                "tier_a",
                "extraction",
                1700,
                "0.030",
                yesterday - timedelta(hours=1),
                "gpt-4o",
                700,
            ),
            (
                seed.run_kg,
                None,
                seed.pipeline_b,
                "linker_kg",
                "tier_b",
                "linking",
                600,
                "0.008",
                yesterday - timedelta(hours=2),
                "gpt-4o-mini",
                220,
            ),
            (
                seed.run_kg,
                None,
                seed.pipeline_b,
                "linker_kg",
                "tier_b",
                "linking",
                700,
                "0.009",
                yesterday - timedelta(hours=3),
                "gpt-4o-mini",
                240,
            ),
            (
                seed.run_quiz,
                None,
                seed.pipeline_a,
                "generator_main",
                "tier_a",
                "generation",
                500,
                "0.006",
                two_days_ago,
                "gpt-4o",
                180,
            ),
            (
                seed.run_quiz,
                None,
                seed.pipeline_a,
                "validator_main",
                "tier_b",
                "validation",
                200,
                "0.002",
                two_days_ago - timedelta(hours=1),
                "gpt-4o-mini",
                90,
            ),
        ]
        for (
            grid,
            pjid,
            prid,
            role_,
            tier_,
            stage_,
            tokens_,
            cost_,
            ts,
            model_,
            latency_,
        ) in rows:
            cid = uuid.uuid4()
            seed.call_ids.append(cid)
            await conn.execute(
                text(
                    "INSERT INTO ai_model_calls "
                    "(id, generation_run_id, processing_job_id, pipeline_run_id, "
                    " role, tier, operation, stage_name, model_name, "
                    " input_tokens, output_tokens, total_tokens, "
                    " estimated_cost_usd, latency_ms, status, called_at) "
                    "VALUES (:id, :grid, :pjid, :prid, :role, :tier, "
                    " 'chat_completion', :stage, :model, "
                    " :tokens / 2, :tokens / 2, :tokens, "
                    " CAST(:cost AS numeric), :latency, 'success', :ts)"
                ),
                {
                    "id": cid,
                    "grid": grid,
                    "pjid": pjid,
                    "prid": prid,
                    "role": role_,
                    "tier": tier_,
                    "stage": stage_,
                    "model": model_,
                    "tokens": tokens_,
                    "cost": cost_,
                    "latency": latency_,
                    "ts": ts,
                },
            )

    yield seed

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM ai_model_calls WHERE id = ANY(:ids)"),
            {"ids": seed.call_ids},
        )
        await conn.execute(
            text("DELETE FROM processing_jobs WHERE id = :id"),
            {"id": seed.processing_quiz},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE id = ANY(:ids)"),
            {"ids": [seed.run_quiz, seed.run_kg]},
        )


async def test_summary_aggregation_correct(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/summary?period=day&since={quote(since)}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["totals"]["call_count"] >= 10
    assert body["totals"]["tokens"] >= 7750
    assert body["totals"]["usd"] >= 0.111

    role_costs = {item["role"]: item["usd"] for item in body["by_role"]}
    assert "generator_main" in role_costs
    assert "extractor_kg" in role_costs
    assert role_costs["extractor_kg"] >= 0.054

    stage_costs = {item["stage_name"]: item["usd"] for item in body["by_stage"]}
    assert "generation" in stage_costs
    assert "extraction" in stage_costs

    assert len(body["buckets"]) >= 3


async def test_summary_buckets_include_zero_spend_days(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    """The daily series must be gap-filled, not just the days that had calls.

    Grouping the rows alone omitted quiet days entirely, so the chart's x-axis
    jumped (observed: Jun 28 -> Jul 12 on the dev DB). That compresses the
    timeline and exaggerates later spikes. Every day in [since, today] must be
    present, with zero-spend days plotted as an explicit 0.
    """
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    days = 20
    since_date = (datetime.now(tz=UTC) - timedelta(days=days)).date()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/summary"
            f"?period=day&since={quote(since_date.isoformat())}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    buckets = resp.json()["buckets"]

    # One bucket per day inclusive of both ends.
    assert len(buckets) == days + 1, (
        f"expected {days + 1} contiguous daily buckets, got {len(buckets)}"
    )

    # Contiguous: no date gaps anywhere in the series.
    dates = [
        datetime.fromisoformat(b["bucket_start_ts"]).date() for b in buckets
    ]
    assert dates == sorted(dates)
    # NOTE: no strict=True here — dates[1:] is deliberately one shorter, which is
    # the point of pairing consecutive elements.
    for earlier, later in zip(dates, dates[1:]):  # noqa: B905
        assert (later - earlier).days == 1, f"gap between {earlier} and {later}"

    # The seed only touches a few recent days, so quiet days must exist and be 0
    # rather than absent.
    zero_days = [b for b in buckets if float(b["usd"]) == 0.0]
    assert zero_days, "expected at least one explicit zero-spend bucket"
    for b in zero_days:
        assert float(b["usd"]) == 0.0
        assert int(b["tokens"]) == 0


async def test_summary_default_period_30_days(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/ai/costs/summary",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["totals"]["call_count"] >= 10


async def test_admin_only_403_for_manager(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token, _ = await _bearer(engine, seeded_users.manager_id)
    try:
        resp = await client.get("/api/v1/admin/ai/costs/summary", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.manager_id)
    assert resp.status_code == 403


async def test_by_user_attributes_through_generation_runs(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/by-user?since={quote(since)}&top_n=20",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    assert any(row["user_id"] == str(cost_seed.requested_by) for row in body), body
    teacher_row = next(row for row in body if row["user_id"] == str(cost_seed.requested_by))
    assert teacher_row["call_count"] >= 10
    assert teacher_row["total_usd"] >= 0.111


async def test_by_pipeline_groups_correctly(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/by-pipeline?since={quote(since)}&top_n=50",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    pipeline_ids = {row["pipeline_run_id"] for row in body}
    assert str(cost_seed.pipeline_a) in pipeline_ids
    assert str(cost_seed.pipeline_b) in pipeline_ids

    a_row = next(row for row in body if row["pipeline_run_id"] == str(cost_seed.pipeline_a))
    assert a_row["call_count"] >= 5
    assert any(s["stage_name"] == "generation" for s in a_row["stages_breakdown"])
    assert any(s["stage_name"] == "validation" for s in a_row["stages_breakdown"])


async def test_recent_sorted_by_cost_desc(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/ai/costs/recent?limit=20",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body) >= 5
    usds = [row["usd"] for row in body]
    assert usds == sorted(usds, reverse=True)


async def test_no_hard_block_endpoint_exists() -> None:
    """Regression guard for the user decision: NO hard rate limit endpoint.

    Observability only; admins inspect, they do not block users on cost.
    """
    methods = {(r.path, tuple(sorted(r.methods))) for r in ai_costs_router.routes}  # type: ignore[attr-defined]
    expected = {
        ("/admin/ai/costs/summary", ("GET",)),
        ("/admin/ai/costs/by-user", ("GET",)),
        ("/admin/ai/costs/by-pipeline", ("GET",)),
        ("/admin/ai/costs/by-category", ("GET",)),
        ("/admin/ai/costs/by-model", ("GET",)),
        ("/admin/ai/costs/recent", ("GET",)),
    }
    assert expected.issubset(methods)
    for _path, verbs in methods:
        assert "POST" not in verbs, f"writes are forbidden on cost dashboard: {_path}"
        assert "PATCH" not in verbs
        assert "PUT" not in verbs
        assert "DELETE" not in verbs


async def test_invalid_period_rejected(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/ai/costs/summary?period=year",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 422


async def test_summary_exposes_token_splits_and_failed(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/summary?period=day&since={quote(since)}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    totals = body["totals"]
    # seed splits every row 50/50 into input/output; cached is unset (0)
    assert totals["input_tokens"] > 0
    assert totals["output_tokens"] > 0
    assert totals["input_tokens"] + totals["output_tokens"] == totals["tokens"]
    assert totals["cached_tokens"] == 0
    # all seeded rows are status='success' -> no failed spend
    assert body["failed"]["call_count"] == 0
    assert body["failed"]["usd"] == 0.0


async def test_by_category_groups_by_dimension(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/by-category?dimension=role&since={quote(since)}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    values = {row["dimension_value"] for row in body}
    assert "generator_main" in values
    assert "extractor_kg" in values
    # sorted by usd desc
    usds = [row["total_usd"] for row in body]
    assert usds == sorted(usds, reverse=True)
    # token splits present per category
    for row in body:
        assert row["input_tokens"] + row["output_tokens"] <= row["total_tokens"] + 1


async def test_by_category_invalid_dimension_rejected(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    token, _ = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            "/api/v1/admin/ai/costs/by-category?dimension=drop_table",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 422


async def test_by_category_filter_narrows_results(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/by-category?dimension=role"
            f"&model=gpt-4o&since={quote(since)}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # gpt-4o only appears on generator_main + extractor_kg rows, never validator/linker
    values = {row["dimension_value"] for row in body}
    assert "validator_main" not in values
    assert "linker_kg" not in values


async def test_by_model_efficiency_metrics(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    cost_seed: _Seed,
) -> None:
    del cost_seed
    token, _ = await _bearer(engine, seeded_users.admin_id)
    since = (datetime.now(tz=UTC) - timedelta(days=5)).date().isoformat()
    try:
        resp = await client.get(
            f"/api/v1/admin/ai/costs/by-model?since={quote(since)}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    models = {row["model_name"] for row in body}
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    gpt4o = next(row for row in body if row["model_name"] == "gpt-4o")
    assert gpt4o["latency_p50_ms"] > 0
    assert gpt4o["latency_p95_ms"] >= gpt4o["latency_p50_ms"]
    assert gpt4o["usd_per_1m_tokens"] > 0
