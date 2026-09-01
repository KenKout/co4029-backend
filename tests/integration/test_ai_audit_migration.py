from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings

PRIOR_HEAD = "0004_seed_permission_catalog"
NEW_HEAD = "0005_ai_audit_pipeline_run"
COMPOSITE_INDEX_NAME = "ix_ai_model_calls_role_stage_created"

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    return cfg


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest.fixture
def alembic_cfg() -> Config:
    return _alembic_config()


@pytest_asyncio.fixture
async def throwaway_db() -> AsyncIterator[str]:
    """A private, empty database for the migration round-trip.

    The round-trip rewinds the schema from real head all the way to 0004 and
    replays every migration after it. Run against the SHARED test database
    that cannot work: the replay drags data-touching migrations over whatever
    rows the other suites have left behind, and the NOT NULL columns added
    between 0005 and head have nothing to backfill those surviving rows with.
    The failure is an artefact of the residue, not of the migrations.

    So the round-trip gets its own database, created empty and dropped
    afterwards. Every assertion here is about SCHEMA — column and index
    presence — which needs no seed data.
    """
    settings_url = get_settings().database_url
    base, _, _ = settings_url.rpartition("/")
    name = f"abridgeai_aiaudit_{uuid.uuid4().hex[:12]}"

    admin = create_async_engine(
        _async_url(f"{base}/postgres"), isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await admin.dispose()

    try:
        yield f"{base}/{name}"
    finally:
        admin = create_async_engine(
            _async_url(f"{base}/postgres"), isolation_level="AUTOCOMMIT"
        )
        async with admin.connect() as conn:
            # Terminate stragglers or DROP blocks on an open connection.
            await conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": name},
            )
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        await admin.dispose()


@pytest_asyncio.fixture
async def at_new_head(alembic_cfg: Config) -> None:
    command.upgrade(alembic_cfg, "head")  # never leave the shared DB below real head
    yield


async def _column_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ai_model_calls'
                """
            )
        )
        return {row[0] for row in result}


async def _index_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'ai_model_calls'
                """
            )
        )
        return {row[0] for row in result}


async def test_migration_adds_pipeline_run_id_and_renames_stage(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    cols = await _column_names(engine)
    assert "pipeline_run_id" in cols
    assert "stage_name" in cols
    assert "pipeline_stage" not in cols
    assert "generation_run_id" in cols
    assert "processing_job_id" in cols


async def test_composite_index_present(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    indexes = await _index_names(engine)
    assert COMPOSITE_INDEX_NAME in indexes

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT a.attname, ix.indkey::text
                FROM pg_index ix
                JOIN pg_class c ON c.oid = ix.indexrelid
                JOIN pg_attribute a ON a.attrelid = ix.indrelid
                WHERE c.relname = :name
                  AND a.attnum = ANY(ix.indkey)
                ORDER BY array_position(ix.indkey, a.attnum)
                """
            ),
            {"name": COMPOSITE_INDEX_NAME},
        )
        cols_in_order = [row[0] for row in result]
    assert cols_in_order == ["role", "stage_name", "called_at"]


async def test_write_ai_model_call_accepts_new_fields(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    pipeline_run_id = uuid.uuid4()

    async with engine.connect() as conn:
        run_result = await conn.execute(
            text(
                """
                INSERT INTO generation_runs (
                    id, generation_type, source_scope_kind, status,
                    requested_by
                )
                VALUES (
                    uuid_generate_v4(), 'quiz', 'lesson', 'pending',
                    CAST('00000000-0000-0000-0000-000000000001' AS uuid)
                )
                RETURNING id
                """
            )
        )
        generation_run_id = run_result.scalar_one()

        insert_result = await conn.execute(
            text(
                """
                INSERT INTO ai_model_calls (
                    generation_run_id, processing_job_id, pipeline_run_id,
                    role, tier, stage_name, operation, model_name, base_url,
                    request_payload, response_payload,
                    input_tokens, output_tokens, total_tokens,
                    cached_input_tokens, estimated_cost_usd,
                    latency_ms, status, error_message
                )
                VALUES (
                    :gen_run_id, NULL, :pipeline_run_id,
                    :role, :tier, :stage_name, 'chat_completion', :model, :base_url,
                    '{}'::jsonb, '{}'::jsonb,
                    10, 5, 15,
                    NULL, 0.000123,
                    42, 'success', NULL
                )
                RETURNING id
                """
            ),
            {
                "gen_run_id": generation_run_id,
                "pipeline_run_id": pipeline_run_id,
                "role": "ideation",
                "tier": "standard",
                "stage_name": "ideation",
                "model": "test-model",
                "base_url": "https://example.test",
            },
        )
        inserted_id = insert_result.scalar_one()
        await conn.commit()

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT pipeline_run_id, stage_name, role, generation_run_id,
                       estimated_cost_usd
                FROM ai_model_calls
                WHERE id = :id
                """
            ),
            {"id": inserted_id},
        )
        fetched = result.one()
    assert fetched[0] == pipeline_run_id
    assert fetched[1] == "ideation"
    assert fetched[2] == "ideation"
    assert fetched[3] == generation_run_id
    assert fetched[4] == Decimal("0.000123")


def test_write_ai_model_call_signature_accepts_new_fields() -> None:
    """Pure-Python check that the audit signature exposes the new kwargs."""
    import inspect

    from abridgeai.ai.llm.audit import write_ai_model_call

    params = inspect.signature(write_ai_model_call).parameters
    assert "pipeline_run_id" in params
    assert "stage_name" in params
    assert "pipeline_stage" not in params


async def test_round_trip_upgrade_downgrade_upgrade(
    throwaway_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Own database, not the shared one — see the `throwaway_db` docstring.
    #
    # `migrations/env.py` reads `get_settings().database_url` directly and
    # ignores the Config's sqlalchemy.url, so redirecting the command means
    # patching settings. env.py is re-imported per alembic command and binds
    # the name at import time, so patching the module attribute is enough.
    from abridgeai.core import config as app_config

    redirected = app_config.get_settings().model_copy(
        update={"database_url": throwaway_db}
    )
    monkeypatch.setattr(app_config, "get_settings", lambda: redirected)

    alembic_cfg = _alembic_config()
    engine = create_async_engine(_async_url(throwaway_db), pool_pre_ping=True)
    command.upgrade(alembic_cfg, "head")

    cols_before = await _column_names(engine)
    indexes_before = await _index_names(engine)
    assert "pipeline_run_id" in cols_before
    assert "stage_name" in cols_before
    assert "pipeline_stage" not in cols_before
    assert COMPOSITE_INDEX_NAME in indexes_before

    command.downgrade(alembic_cfg, PRIOR_HEAD)

    cols_down = await _column_names(engine)
    indexes_down = await _index_names(engine)
    assert "pipeline_run_id" not in cols_down
    assert "stage_name" not in cols_down
    assert "pipeline_stage" in cols_down
    assert COMPOSITE_INDEX_NAME not in indexes_down

    command.upgrade(alembic_cfg, "head")

    cols_after = await _column_names(engine)
    indexes_after = await _index_names(engine)
    assert "pipeline_run_id" in cols_after
    assert "stage_name" in cols_after
    assert "pipeline_stage" not in cols_after
    assert COMPOSITE_INDEX_NAME in indexes_after

    await engine.dispose()  # or DROP DATABASE blocks on this connection
