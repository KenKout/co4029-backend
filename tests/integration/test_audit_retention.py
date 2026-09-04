"""Integration tests for the audit retention sweep.

The append-only triggers (migrations 0105 / 0106) mean ordinary code cannot
delete an audit row. That makes a retention job mandatory rather than optional:
without one the tables grow for the life of the deployment, which is exactly
what happened to ``http_audit_log`` (127k rows / 87 MB in ~3.5 months) before
:mod:`abridgeai.core.audit.retention` existed.

These tests drive the real sweep against the real tables, because the whole
point is the interaction between the trigger and the delete — a mocked session
would prove nothing about whether the maintenance scope was entered correctly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from abridgeai.core.audit.retention import RETAINED_TABLES, prune_audit_logs
from abridgeai.core.runtime_settings import invalidate_settings_cache

_KEY = "audit.http_log_retention_days"


@pytest_asyncio.fixture
async def session_factory(test_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def clean_setting(test_engine: AsyncEngine) -> AsyncIterator[None]:
    """Remove any global override of the retention keys, in and out.

    Cleaning on the way IN as well as OUT: a leftover override from an earlier
    failure would otherwise silently reinterpret every assertion here as a test
    of that value.
    """

    async def _purge() -> None:
        async with test_engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM system_settings WHERE setting_key = ANY(:k) "
                    "AND organization_id IS NULL"
                ),
                {"k": [spec.setting_key for spec in RETAINED_TABLES]},
            )
        invalidate_settings_cache()

    await _purge()
    yield
    await _purge()


async def _insert_http_row(engine: AsyncEngine, *, age_days: int) -> uuid.UUID:
    """One ``http_audit_log`` row stamped ``age_days`` in the past."""
    row_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO http_audit_log "
                "(id, request_id, method, path, status_code, latency_ms, created_at) "
                "VALUES (:id, :rid, 'GET', '/api/v1/retention-probe', 200, 5, :ts)"
            ),
            {
                "id": row_id,
                "rid": uuid.uuid4(),
                "ts": datetime.now(tz=UTC) - timedelta(days=age_days),
            },
        )
    return row_id


async def _exists(engine: AsyncEngine, row_id: uuid.UUID) -> bool:
    async with engine.begin() as conn:
        found = (
            await conn.execute(
                text("SELECT 1 FROM http_audit_log WHERE id = :id"), {"id": row_id}
            )
        ).scalar_one_or_none()
    return found is not None


async def _set_window(engine: AsyncEngine, days: int) -> None:
    """Write the GLOBAL override, mirroring ``admin/queries/settings.py``.

    The conflict target is the partial unique index on
    ``(setting_key) WHERE organization_id IS NULL`` — a plain
    ``(setting_key, organization_id)`` target does not match it, because NULL
    organization rows are indexed separately from tenant rows.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO system_settings "
                "(id, setting_key, setting_value_json, updated_by) "
                "VALUES (gen_random_uuid(), :k, CAST(:v AS jsonb), NULL) "
                "ON CONFLICT (setting_key) WHERE organization_id IS NULL "
                "DO UPDATE SET setting_value_json = EXCLUDED.setting_value_json, "
                "              updated_at = NOW()"
            ),
            {"k": _KEY, "v": str(days)},
        )
    invalidate_settings_cache()


async def test_sweep_deletes_rows_past_the_window_and_keeps_the_rest(
    test_engine: AsyncEngine,
    session_factory: async_sessionmaker,
    clean_setting: None,
) -> None:
    """The core contract: old rows go, recent rows stay.

    This is also the test that proves the sweep enters the maintenance scope —
    the trigger refuses the DELETE outright otherwise, so a regression there
    surfaces as a RestrictViolation rather than a silently kept row.
    """
    await _set_window(test_engine, 90)
    old_row = await _insert_http_row(test_engine, age_days=120)
    fresh_row = await _insert_http_row(test_engine, age_days=1)

    results = await prune_audit_logs(session_factory)

    assert results["http_audit_log"] >= 1
    assert await _exists(test_engine, old_row) is False
    assert await _exists(test_engine, fresh_row) is True

    # Clean up the row the sweep deliberately kept.
    from abridgeai.core.audit import audit_maintenance

    async with test_engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(
            text("DELETE FROM http_audit_log WHERE id = :id"), {"id": fresh_row}
        )


async def test_a_zero_window_disables_pruning(
    test_engine: AsyncEngine,
    session_factory: async_sessionmaker,
    clean_setting: None,
) -> None:
    """0 means keep forever — the escape hatch for a compliance regime that
    forbids deleting request logs at all."""
    await _set_window(test_engine, 0)
    ancient = await _insert_http_row(test_engine, age_days=5000)

    results = await prune_audit_logs(session_factory)

    assert results["http_audit_log"] == 0
    assert await _exists(test_engine, ancient) is True

    from abridgeai.core.audit import audit_maintenance

    async with test_engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(text("DELETE FROM http_audit_log WHERE id = :id"), {"id": ancient})


async def test_sweep_reports_every_retained_table(
    session_factory: async_sessionmaker,
    clean_setting: None,
) -> None:
    """The result dict covers every configured table, so a caller logging it
    cannot silently omit one that stopped being pruned."""
    results = await prune_audit_logs(session_factory)
    assert set(results) == {spec.table for spec in RETAINED_TABLES}


async def test_default_window_is_ninety_days(
    session_factory: async_sessionmaker,
    clean_setting: None,
) -> None:
    """With no override the registry default applies.

    Asserted explicitly because the default is the value every deployment runs
    until someone tunes it, and a silent change to it changes how long every
    tenant's audit history survives.
    """
    from abridgeai.core.settings_registry import SETTINGS_REGISTRY

    for spec in RETAINED_TABLES:
        assert SETTINGS_REGISTRY[spec.setting_key].default == 90


async def test_retention_targets_exist_and_are_guarded(
    test_engine: AsyncEngine,
) -> None:
    """Every configured table must really exist and carry the trigger.

    Guards against a typo'd table name, which would otherwise make the sweep
    silently report 0 forever while the table grew.
    """
    async with test_engine.begin() as conn:
        guarded = {
            row
            for row in (
                await conn.execute(
                    text(
                        "SELECT c.relname FROM pg_trigger tg "
                        "JOIN pg_proc p ON p.oid = tg.tgfoid "
                        "JOIN pg_class c ON c.oid = tg.tgrelid "
                        "WHERE NOT tg.tgisinternal AND p.proname = 'audit_log_immutable'"
                    )
                )
            ).scalars()
        }
    for spec in RETAINED_TABLES:
        assert spec.table in guarded, f"{spec.table} is pruned but not append-only guarded"
