"""Coverage for the runtime-settings admin surface.

Four modules meet here and none of them had a test:

* ``admin/queries/settings.py``        -- the ``system_settings`` upsert/delete
* ``admin/queries/setting_changes.py`` -- the append-only audit trail
* ``admin/services/settings.py``       -- provenance, validation, change flow
* ``admin/routers/settings.py``        -- global and per-org endpoints

The through-line worth protecting is PROVENANCE. Every one of these settings
has four possible origins (org row, global row, environment, code default) and
the service's whole reason to exist is answering "why is it this number?".
A regression there is silent: the value still resolves, the page still renders,
and the operator is simply told the wrong story about their own deployment.

The second is that a change can never land without its audit row. That pairing
is the feature; the tests below assert it from both the service and the HTTP
edge rather than trusting the shared-transaction comment.

Isolation: the suite shares one Postgres, so every test here writes only keys
it owns and deletes both the ``system_settings`` and ``system_setting_changes``
rows it created. Settings are also process-cached, so the cache is invalidated
on the way out as well as by the service under test.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

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

from abridgeai.core.audit.maintenance import audit_maintenance
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.runtime_settings import invalidate_settings_cache
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.core.settings_registry import SETTINGS_REGISTRY, SettingValidationError
from abridgeai.features.admin.queries import setting_changes as change_queries
from abridgeai.features.admin.queries import settings as settings_queries
from abridgeai.features.admin.routers import settings_router
from abridgeai.features.admin.services import settings as settings_service

# A key with no cross-field rules and a wide numeric range, so a test can move
# it freely without tripping an invariant that belongs to another test.
KEY = "chunking.max_tokens"
KEY_DEFAULT = 800
KEY_ENV_VAR = "CHUNKING_MAX_TOKENS"
BOOL_KEY = "chunking.llm_boundary_enabled"
BOOL_KEY_ENV_VAR = "CHUNKING_LLM_BOUNDARY_ENABLED"


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
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
async def db(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(settings_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


_ISSUED_SESSIONS: list[uuid.UUID] = []


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> str:
    sid = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, NOW() + INTERVAL '1 hour')"
            ),
            {"id": sid, "uid": user_id, "h": hash_secret(generate_token())},
        )
    _ISSUED_SESSIONS.append(sid)
    return create_access_token(user_id=user_id, session_id=sid)


@pytest_asyncio.fixture(autouse=True)
async def _purge_issued_sessions(engine: AsyncEngine) -> AsyncIterator[None]:
    """Delete every session ``_bearer`` handed out, pass or fail.

    Purging by recorded id rather than by user leaves sessions belonging to
    other suites alone -- the whole file shares one Postgres.
    """
    yield
    if not _ISSUED_SESSIONS:
        return
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(s) for s in _ISSUED_SESSIONS]},
        )
    _ISSUED_SESSIONS.clear()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture(autouse=True)
async def _clean_settings(engine: AsyncEngine) -> AsyncIterator[None]:
    """Own the keys under test, before and after.

    Cleaning on the way IN as well as out matters: an earlier failure that
    skipped its teardown would otherwise turn every provenance assertion here
    into a test of that leftover row instead.
    """
    keys = [
        KEY,
        BOOL_KEY,
        "courses.min_teachers_per_course",
        "courses.max_teachers_per_course",
    ]

    async def _purge() -> None:
        async with engine.begin() as conn:
            # system_setting_changes is append-only (migration 0105 trigger):
            # DELETE is refused unless the transaction opts into the audit
            # maintenance scope first. A test fixture pruning its own probe rows
            # is exactly the retention-style caller that scope exists for.
            await audit_maintenance(conn)
            await conn.execute(
                text("DELETE FROM system_setting_changes WHERE setting_key = ANY(:k)"),
                {"k": keys},
            )
            await conn.execute(
                text("DELETE FROM system_settings WHERE setting_key = ANY(:k)"),
                {"k": keys},
            )
        invalidate_settings_cache()

    await _purge()
    yield
    await _purge()


@pytest_asyncio.fixture
async def other_org(engine: AsyncEngine) -> AsyncIterator[uuid.UUID]:
    """A second live organization, for the blast-radius arithmetic."""
    org_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Settings Coverage Org', 'active')"
            ),
            {"id": org_id, "slug": f"settings-cov-{org_id.hex[:10]}"},
        )
    yield org_id
    async with engine.begin() as conn:
        # Append-only audit store — same opt-in as the key purge above.
        await audit_maintenance(conn)
        await conn.execute(
            text("DELETE FROM system_setting_changes WHERE organization_id = :id"),
            {"id": org_id},
        )
        await conn.execute(
            text("DELETE FROM system_settings WHERE organization_id = :id"),
            {"id": org_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


# ---------------------------------------------------------------------------
# queries/settings.py
# ---------------------------------------------------------------------------


async def test_upsert_global_is_idempotent_on_the_partial_index(
    db: AsyncSession, seeded_users: SeededUsers
) -> None:
    """The global upsert must UPDATE the second time, not raise or duplicate.

    ``system_settings`` has no composite UNIQUE on (organization_id,
    setting_key) -- Postgres treats NULLs as distinct, so that constraint could
    not enforce one global row per key. Two partial unique indexes do the work
    and the statement targets one of them by repeating its predicate. If that
    predicate ever drifts from the index, the ON CONFLICT stops matching and
    every save silently appends a second global row instead of updating.
    """
    for value in (900, 1100):
        await settings_queries.upsert(
            db,
            setting_key=KEY,
            value_json=json.dumps(value),
            organization_id=None,
            updated_by=seeded_users.admin_id,
        )
        await db.commit()

    rows = await settings_queries.load_rows(db, None)
    mine = [r for r in rows if r["setting_key"] == KEY]
    assert len(mine) == 1, f"expected exactly one global row, got {mine}"
    assert mine[0]["setting_value_json"] == 1100
    assert mine[0]["organization_id"] is None


async def test_global_and_org_rows_coexist_for_one_key(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """An org override is a SEPARATE row, not an overwrite of the global one.

    This is the structural precondition for the whole provenance feature: if
    the org upsert collided with the global index, setting an override would
    destroy the deployment default for every other tenant.
    """
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="700", organization_id=None, updated_by=None
    )
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="1500", organization_id=other_org, updated_by=None
    )
    await db.commit()

    rows = [r for r in await settings_queries.load_rows(db, other_org) if r["setting_key"] == KEY]
    by_scope = {r["organization_id"]: r["setting_value_json"] for r in rows}
    assert by_scope == {None: 700, other_org: 1500}


async def test_load_rows_does_not_leak_another_tenants_override(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """Asking for org A must not return org B's row."""
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="1500", organization_id=other_org, updated_by=None
    )
    await db.commit()

    unrelated = uuid.uuid4()
    rows = [r for r in await settings_queries.load_rows(db, unrelated) if r["setting_key"] == KEY]
    assert rows == [], "another organization's override must not be visible"


async def test_delete_reports_rowcount_and_is_a_no_op_when_unset(
    db: AsyncSession,
) -> None:
    """Deleting something never set is not an error; the count says so."""
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="900", organization_id=None, updated_by=None
    )
    await db.commit()

    assert await settings_queries.delete(db, setting_key=KEY, organization_id=None) == 1
    assert await settings_queries.delete(db, setting_key=KEY, organization_id=None) == 0
    await db.commit()


async def test_delete_is_scoped_and_leaves_the_global_row_alone(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """Clearing an org override must not clear the deployment default."""
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="700", organization_id=None, updated_by=None
    )
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="1500", organization_id=other_org, updated_by=None
    )
    await db.commit()

    assert await settings_queries.delete(db, setting_key=KEY, organization_id=other_org) == 1
    await db.commit()

    rows = [r for r in await settings_queries.load_rows(db, other_org) if r["setting_key"] == KEY]
    assert [r["setting_value_json"] for r in rows] == [700]


# ---------------------------------------------------------------------------
# services/settings.py -- provenance
# ---------------------------------------------------------------------------


async def _resolved(
    db: AsyncSession, key: str, org: uuid.UUID | None
) -> settings_service.ResolvedSetting:
    return next(r for r in await settings_service.list_settings(db, org) if r.key == key)


async def test_source_is_default_when_nothing_is_set(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    row = await _resolved(db, KEY, None)
    assert row.effective_value == KEY_DEFAULT
    assert row.source == "default"
    assert row.global_value is None
    assert row.org_value is None


async def test_precedence_walks_default_env_global_org(
    db: AsyncSession, other_org: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each level in turn must win over the one below it.

    Asserted as one ladder rather than four tests because the property under
    test is the ORDER, and an ordering bug is only visible when the levels are
    stacked -- any single level in isolation resolves correctly even if the
    comparison chain is wrong.
    """
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    row = await _resolved(db, KEY, other_org)
    assert (row.effective_value, row.source) == (KEY_DEFAULT, "default")

    monkeypatch.setenv(KEY_ENV_VAR, "1234")
    row = await _resolved(db, KEY, other_org)
    assert (row.effective_value, row.source) == (1234, "environment")
    assert row.env_value == 1234

    await settings_queries.upsert(
        db, setting_key=KEY, value_json="700", organization_id=None, updated_by=None
    )
    await db.commit()
    row = await _resolved(db, KEY, other_org)
    assert (row.effective_value, row.source) == (700, "global")

    await settings_queries.upsert(
        db, setting_key=KEY, value_json="1500", organization_id=other_org, updated_by=None
    )
    await db.commit()
    row = await _resolved(db, KEY, other_org)
    assert (row.effective_value, row.source) == (1500, "organization")

    # Every level stays VISIBLE even when outranked -- that is the point of
    # the page: the operator can see what they would fall back to.
    assert (row.default_value, row.env_value, row.global_value, row.org_value) == (
        KEY_DEFAULT,
        1234,
        700,
        1500,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("OFF", False),
        ("maybe", None),  # unparseable -> ignored, not crashed on
        ("", None),  # blank -> unset, not False
        ("   ", None),
    ],
)
async def test_bool_env_parsing(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool | None
) -> None:
    """A malformed env var must fall through, never raise or coerce to False.

    The blank cases matter most: ``CHUNKING_LLM_BOUNDARY_ENABLED=`` in a
    compose file is "I did not set this", and reading it as False would
    silently disable a pipeline stage.
    """
    monkeypatch.setenv(BOOL_KEY_ENV_VAR, raw)
    row = await _resolved(db, BOOL_KEY, None)
    assert row.env_value is expected


async def test_unparseable_int_env_is_ignored(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV_VAR, "not-a-number")
    row = await _resolved(db, KEY, None)
    assert row.env_value is None
    assert row.source == "default"


async def test_stored_value_outside_its_spec_is_discarded(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that no longer fits its spec must not become the effective value.

    Bounds get tightened as the product learns what is safe. When they do,
    rows written under the old spec are still in the table; serving one would
    hand an ingest worker a number the code has since declared invalid. The
    row is ignored and the next level down takes over.
    """
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    # 99999 is far above the registered maximum of 4000. Written through the
    # query layer deliberately -- the service would have rejected it, which is
    # exactly how such a row comes to exist: it predates the current bounds.
    await settings_queries.upsert(
        db, setting_key=KEY, value_json="99999", organization_id=None, updated_by=None
    )
    await db.commit()

    row = await _resolved(db, KEY, None)
    assert row.global_value is None, "out-of-spec row must not be reported as set"
    assert (row.effective_value, row.source) == (KEY_DEFAULT, "default")


async def test_every_registered_setting_resolves(db: AsyncSession) -> None:
    """The list is registry-complete and never returns a null effective value.

    Cheap insurance for a new SettingSpec landing with a default that its own
    validator rejects -- that ships a page which 500s on load.
    """
    rows = await settings_service.list_settings(db, None)
    assert [r.key for r in rows] == list(SETTINGS_REGISTRY)
    for row in rows:
        assert row.effective_value is not None, row.key
        assert row.source in {"organization", "global", "environment", "default"}


# ---------------------------------------------------------------------------
# services/settings.py -- validation
# ---------------------------------------------------------------------------


async def test_unknown_key_is_rejected_by_set_and_clear(db: AsyncSession) -> None:
    """The registry is a closed allowlist, on the write path AND the clear path.

    Without the guard a typo writes a row nothing ever reads, and the operator
    sees a saved value that does nothing.
    """
    with pytest.raises(SettingValidationError):
        await settings_service.set_setting(
            db, key="chunking.max_tokenz", value=800, organization_id=None, actor_id=None
        )
    with pytest.raises(SettingValidationError):
        await settings_service.clear_setting(db, key="chunking.max_tokenz", organization_id=None)


@pytest.mark.parametrize("value", [50, 99999, "eight hundred"])
async def test_out_of_bounds_values_are_rejected(db: AsyncSession, value: object) -> None:
    with pytest.raises(SettingValidationError):
        await settings_service.set_setting(
            db, key=KEY, value=value, organization_id=None, actor_id=None
        )


async def test_min_teachers_cannot_exceed_max(db: AsyncSession) -> None:
    """A min above the max makes every course unpublishable.

    Both directions are guarded because either field can be the one moved.
    """
    with pytest.raises(SettingValidationError, match="cannot exceed"):
        await settings_service.set_setting(
            db,
            key="courses.min_teachers_per_course",
            value=11,  # default max is 10
            organization_id=None,
            actor_id=None,
        )

    await settings_service.set_setting(
        db,
        key="courses.min_teachers_per_course",
        value=5,
        organization_id=None,
        actor_id=None,
    )
    await db.commit()
    with pytest.raises(SettingValidationError, match="cannot be below"):
        await settings_service.set_setting(
            db,
            key="courses.max_teachers_per_course",
            value=4,
            organization_id=None,
            actor_id=None,
        )


async def test_preview_enforces_the_same_invariant_as_apply(db: AsyncSession) -> None:
    """A preview that passes where the apply would fail is worse than none.

    The operator reads "this is fine", commits, and gets an error on the write
    they were just told was safe.
    """
    with pytest.raises(SettingValidationError):
        await settings_service.preview_change(
            db, key="courses.min_teachers_per_course", value=11, organization_id=None
        )


@pytest.mark.parametrize("reason", ["", "  ", "ok", "x" * 501])
async def test_reason_is_mandatory_and_bounded(db: AsyncSession, reason: str) -> None:
    with pytest.raises(SettingValidationError):
        await settings_service.apply_change(
            db,
            key=KEY,
            value=900,
            organization_id=None,
            actor_id=None,
            reason=reason,
        )


async def test_a_rejected_change_writes_neither_value_nor_audit_row(
    db: AsyncSession,
) -> None:
    """Validation failure must leave no trace at all.

    A rejected apply that still recorded an audit row would make the history
    lie about what the deployment ever ran.
    """
    with pytest.raises(SettingValidationError):
        await settings_service.apply_change(
            db, key=KEY, value=99999, organization_id=None, actor_id=None, reason="nope"
        )
    await db.rollback()

    assert (await _resolved(db, KEY, None)).global_value is None
    assert await settings_service.list_changes(db, key=KEY) == []


# ---------------------------------------------------------------------------
# services/settings.py -- change workflow + queries/setting_changes.py
# ---------------------------------------------------------------------------


async def test_apply_change_writes_the_value_and_its_audit_row(
    db: AsyncSession, seeded_users: SeededUsers
) -> None:
    """The pairing is the feature: a value and its record, or neither."""
    updated, audit = await settings_service.apply_change(
        db,
        key=KEY,
        value=1200,
        organization_id=None,
        actor_id=seeded_users.admin_id,
        reason="raising window size for the ingest backlog",
    )
    await db.commit()

    assert (updated.effective_value, updated.source) == (1200, "global")

    history = await settings_service.list_changes(db, key=KEY)
    assert len(history) == 1
    row = history[0]
    assert row["id"] == audit["id"]
    assert row["action"] == "set"
    assert row["scope"] == "global"
    assert row["before_value_json"] is None, "nothing was stored before"
    assert row["after_value_json"] == 1200
    assert row["reason"] == "raising window size for the ingest backlog"
    assert row["actor_id"] == seeded_users.admin_id
    assert row["source"] == settings_service.CHANGE_SOURCE_ADMIN_CONSOLE
    # Joined at read time so the history is readable without a second lookup.
    assert row["actor_email"] is not None


async def test_before_value_is_what_was_stored_not_what_was_inherited(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """An org with no override of its own has nothing to record as "before".

    Recording the INHERITED number instead would make a later rollback pin the
    org to a value nobody ever chose for it -- and silently detach it from the
    global default it had been following.
    """
    await settings_service.set_setting(db, key=KEY, value=700, organization_id=None, actor_id=None)
    await db.commit()

    _, audit = await settings_service.apply_change(
        db,
        key=KEY,
        value=1500,
        organization_id=other_org,
        actor_id=None,
        reason="tenant needs larger windows",
    )
    await db.commit()

    history = await settings_service.list_changes(db, key=KEY, organization_id=other_org)
    row = next(r for r in history if r["id"] == audit["id"])
    assert row["before_value_json"] is None, "inherited 700 must not be recorded as stored"
    assert row["after_value_json"] == 1500
    assert row["scope"] == "organization"


async def test_apply_clear_records_the_removed_value(db: AsyncSession) -> None:
    await settings_service.set_setting(db, key=KEY, value=1200, organization_id=None, actor_id=None)
    await db.commit()

    updated, _ = await settings_service.apply_clear(
        db, key=KEY, organization_id=None, actor_id=None, reason="reverting the experiment"
    )
    await db.commit()

    assert updated.global_value is None
    assert updated.source in {"default", "environment"}

    latest = (await settings_service.list_changes(db, key=KEY))[0]
    assert latest["action"] == "clear"
    assert latest["before_value_json"] == 1200
    assert latest["after_value_json"] is None


async def test_rollback_restores_a_replaced_value(db: AsyncSession) -> None:
    await settings_service.apply_change(
        db, key=KEY, value=700, organization_id=None, actor_id=None, reason="first value"
    )
    _, second = await settings_service.apply_change(
        db, key=KEY, value=1500, organization_id=None, actor_id=None, reason="second value"
    )
    await db.commit()

    restored, audit = await settings_service.rollback_change(
        db, change_id=second["id"], actor_id=None, reason="that was wrong"
    )
    await db.commit()

    assert restored.effective_value == 700
    assert audit["id"] != second["id"], "rollback appends; it never rewrites"

    latest = (await settings_service.list_changes(db, key=KEY))[0]
    assert latest["action"] == "rollback"
    assert latest["reverted_change_id"] == second["id"]
    assert latest["before_value_json"] == 1500
    assert latest["after_value_json"] == 700
    # The original row is untouched -- the history is append-only.
    assert len(await settings_service.list_changes(db, key=KEY)) == 3


async def test_rollback_of_a_change_that_created_an_override_removes_it_again(
    db: AsyncSession,
) -> None:
    """Undoing "someone set this" means going back to INHERITING, not to a number.

    The easy bug is to restore whatever the key happened to be resolving to,
    which pins the scope to a value it was never explicitly given and detaches
    it from the level it used to follow.
    """
    _, created = await settings_service.apply_change(
        db, key=KEY, value=1500, organization_id=None, actor_id=None, reason="set it"
    )
    await db.commit()

    restored, _ = await settings_service.rollback_change(
        db, change_id=created["id"], actor_id=None, reason="undo"
    )
    await db.commit()

    assert restored.global_value is None, "the override must be gone, not re-pinned"
    assert restored.source in {"default", "environment"}


async def test_rollback_is_scoped_to_the_callers_tenant(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """An org-scoped admin cannot undo a global change or another tenant's.

    Reported as not-found rather than forbidden on purpose: the existence of
    another tenant's configuration history is not this caller's to learn.
    """
    _, global_change = await settings_service.apply_change(
        db, key=KEY, value=1500, organization_id=None, actor_id=None, reason="global change"
    )
    await db.commit()

    with pytest.raises(NotFoundError):
        await settings_service.rollback_change(
            db,
            change_id=global_change["id"],
            actor_id=None,
            reason="not mine to undo",
            organization_id=other_org,
        )


async def test_rollback_of_a_missing_change_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await settings_service.rollback_change(
            db, change_id=uuid.uuid4(), actor_id=None, reason="does not exist"
        )


async def test_list_changes_separates_global_only_from_unfiltered(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """``global_only`` exists because a NULL org cannot express "no filter".

    Without the distinct flag, the global history page would show every
    tenant's changes, and the "all scopes" view would be unreachable.
    """
    await settings_service.apply_change(
        db, key=KEY, value=700, organization_id=None, actor_id=None, reason="global one"
    )
    await settings_service.apply_change(
        db, key=KEY, value=1500, organization_id=other_org, actor_id=None, reason="org one"
    )
    await db.commit()

    unfiltered = await settings_service.list_changes(db, key=KEY)
    assert {r["scope"] for r in unfiltered} == {"global", "organization"}

    global_only = await settings_service.list_changes(db, key=KEY, global_only=True)
    assert [r["scope"] for r in global_only] == ["global"]

    org_only = await settings_service.list_changes(db, key=KEY, organization_id=other_org)
    assert [r["organization_id"] for r in org_only] == [other_org]


async def test_change_history_is_newest_first(db: AsyncSession) -> None:
    """Committed one at a time on purpose.

    ``created_at`` defaults to NOW(), which in Postgres is the TRANSACTION
    timestamp -- three changes applied inside one transaction all carry the
    same instant, and the ordering then falls through to the ``id DESC``
    tiebreaker on a random UUID. Batching them would make this test assert a
    shuffle. Separate transactions are also what the real console does: one
    change per request.
    """
    for value in (700, 900, 1200):
        await settings_service.apply_change(
            db,
            key=KEY,
            value=value,
            organization_id=None,
            actor_id=None,
            reason=f"to {value}",
        )
        await db.commit()

    history = await settings_service.list_changes(db, key=KEY)
    assert [r["after_value_json"] for r in history] == [1200, 900, 700]


async def test_list_changes_honours_its_limit(db: AsyncSession) -> None:
    for value in (700, 900, 1200):
        await settings_service.apply_change(
            db,
            key=KEY,
            value=value,
            organization_id=None,
            actor_id=None,
            reason=f"to {value}",
        )
    await db.commit()
    assert len(await settings_service.list_changes(db, key=KEY, limit=2)) == 2


async def test_get_change_round_trips_and_misses_cleanly(db: AsyncSession) -> None:
    _, audit = await settings_service.apply_change(
        db, key=KEY, value=1200, organization_id=None, actor_id=None, reason="a change"
    )
    await db.commit()

    fetched = await change_queries.get_change(db, change_id=audit["id"])
    assert fetched is not None
    assert fetched["setting_key"] == KEY
    assert await change_queries.get_change(db, change_id=uuid.uuid4()) is None


# ---------------------------------------------------------------------------
# Blast radius
# ---------------------------------------------------------------------------


async def test_preview_counts_only_organizations_that_would_actually_feel_it(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    """An org with its own override does not feel a global change.

    Counting every organization would overstate the blast radius on exactly
    the keys most tenants have customised -- the ones where an operator most
    needs an honest number before pressing apply.
    """
    before = await settings_service.preview_change(db, key=KEY, value=1200, organization_id=None)
    assert before.affected_organizations == before.total_organizations
    assert before.total_organizations >= 1

    await settings_queries.upsert(
        db, setting_key=KEY, value_json="1500", organization_id=other_org, updated_by=None
    )
    await db.commit()

    after = await settings_service.preview_change(db, key=KEY, value=1200, organization_id=None)
    assert after.total_organizations == before.total_organizations
    assert after.affected_organizations == before.affected_organizations - 1


async def test_org_scoped_preview_reports_one_organization(
    db: AsyncSession, other_org: uuid.UUID
) -> None:
    impact = await settings_service.preview_change(
        db, key=KEY, value=1200, organization_id=other_org
    )
    assert impact.scope == "organization"
    assert impact.affected_organizations == 1


async def test_preview_flags_a_no_op_write(db: AsyncSession) -> None:
    """ "Unchanged" compares against the STORED value at this scope.

    An org inheriting 800 that is now being pinned to 800 is not a no-op: it
    detaches the tenant from the global default, which is a real change even
    though the number on screen stays put.
    """
    await settings_service.set_setting(db, key=KEY, value=1200, organization_id=None, actor_id=None)
    await db.commit()

    same = await settings_service.preview_change(db, key=KEY, value=1200, organization_id=None)
    assert same.unchanged is True

    different = await settings_service.preview_change(db, key=KEY, value=900, organization_id=None)
    assert different.unchanged is False
    assert different.current_value == 1200
    assert different.new_value == 900


async def test_preview_clear_reports_inheritance_not_an_empty_value(
    db: AsyncSession,
) -> None:
    await settings_service.set_setting(db, key=KEY, value=1200, organization_id=None, actor_id=None)
    await db.commit()

    impact = await settings_service.preview_clear(db, key=KEY, organization_id=None)
    assert impact.new_value is None
    assert impact.unchanged is False
    assert impact.current_value == 1200

    await settings_service.clear_setting(db, key=KEY, organization_id=None)
    await db.commit()
    assert (await settings_service.preview_clear(db, key=KEY, organization_id=None)).unchanged


async def test_preview_carries_the_reprocess_warning(db: AsyncSession) -> None:
    """ADM-034: some settings only bite on the NEXT ingest.

    Without the flag the operator reasonably expects already-processed
    material to change, and reads the unchanged corpus as a failed save.
    """
    impact = await settings_service.preview_change(db, key=KEY, value=1200, organization_id=None)
    assert impact.requires_reprocess is True
    assert impact.label
    assert impact.description


async def test_preview_rejects_an_unknown_key_to_clear(db: AsyncSession) -> None:
    with pytest.raises(SettingValidationError):
        await settings_service.preview_clear(db, key="nope.not.a.key", organization_id=None)


# ---------------------------------------------------------------------------
# routers/settings.py
# ---------------------------------------------------------------------------


async def test_global_settings_endpoints_require_system_administer(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """A manager is not a deployment operator."""
    token = await _bearer(engine, seeded_users.manager_id)
    resp = await client.get("/api/v1/admin/settings", headers=_auth(token))
    assert resp.status_code == 403, resp.text


async def test_settings_endpoints_reject_an_anonymous_caller(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get("/api/v1/admin/settings")
    assert resp.status_code in {401, 403}, resp.text


async def test_admin_can_read_apply_and_roll_back_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The full console round trip, at the HTTP edge.

    Exercised end to end rather than per-endpoint because the thing that
    breaks is the seam: an apply whose response omits ``change_id`` leaves the
    UI unable to offer the rollback the feature exists to provide.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)

    listed = await client.get("/api/v1/admin/settings", headers=headers)
    assert listed.status_code == 200, listed.text
    row = next(r for r in listed.json() if r["key"] == KEY)
    assert row["requires_reprocess"] is True

    preview = await client.post(
        f"/api/v1/admin/settings/{KEY}/preview", json={"value": 1200}, headers=headers
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["new_value"] == 1200
    assert preview.json()["unchanged"] is False

    applied = await client.put(
        f"/api/v1/admin/settings/{KEY}",
        json={"value": 1200, "reason": "raising the window for a backlog"},
        headers=headers,
    )
    assert applied.status_code == 200, applied.text
    body = applied.json()
    assert body["setting"]["effective_value"] == 1200
    assert body["setting"]["source"] == "global"
    change_id = body["change_id"]

    history = await client.get("/api/v1/admin/settings/changes", headers=headers)
    assert history.status_code == 200, history.text
    assert any(c["id"] == change_id for c in history.json())

    rolled = await client.post(
        f"/api/v1/admin/settings/changes/{change_id}/rollback",
        json={"reason": "backing that out again"},
        headers=headers,
    )
    assert rolled.status_code == 200, rolled.text
    assert rolled.json()["setting"]["global_value"] is None


async def test_http_apply_rejects_an_out_of_range_value_as_422(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.put(
        f"/api/v1/admin/settings/{KEY}",
        json={"value": 99999, "reason": "far too large"},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error"] == "validation"

    # And nothing was written.
    listed = await client.get("/api/v1/admin/settings", headers=_auth(token))
    assert next(r for r in listed.json() if r["key"] == KEY)["global_value"] is None


async def test_http_apply_requires_a_reason(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """ADM-033. Enforced by the request model, so it never reaches the service."""
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.put(
        f"/api/v1/admin/settings/{KEY}",
        json={"value": 1200},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


async def test_http_rollback_of_an_unknown_change_is_404(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(
        f"/api/v1/admin/settings/changes/{uuid.uuid4()}/rollback",
        json={"reason": "no such change"},
        headers=_auth(token),
    )
    assert resp.status_code == 404, resp.text


async def test_org_override_endpoints_round_trip(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Per-tenant overrides over HTTP, including the clear path back."""
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)
    org = seeded_users.organization_id

    applied = await client.put(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}",
        json={"value": 1500, "reason": "this tenant ingests larger documents"},
        headers=headers,
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["setting"]["org_value"] == 1500
    assert applied.json()["setting"]["source"] == "organization"

    listed = await client.get(f"/api/v1/admin/organizations/{org}/settings", headers=headers)
    assert listed.status_code == 200, listed.text
    assert next(r for r in listed.json() if r["key"] == KEY)["org_value"] == 1500

    changes = await client.get(
        f"/api/v1/admin/organizations/{org}/settings/changes", headers=headers
    )
    assert changes.status_code == 200, changes.text
    assert all(c["organization_id"] == str(org) for c in changes.json())

    cleared = await client.delete(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}",
        params={"reason": "back to the deployment default"},
        headers=headers,
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["setting"]["org_value"] is None


async def test_org_settings_reject_a_tenant_the_caller_cannot_reach(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """A manager scoped to one org cannot configure another."""
    token = await _bearer(engine, seeded_users.manager_id)
    resp = await client.get(
        f"/api/v1/admin/organizations/{uuid.uuid4()}/settings", headers=_auth(token)
    )
    assert resp.status_code in {403, 404}, resp.text


async def test_global_clear_over_http_records_its_reason(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The clear reason rides in the QUERY STRING, not a body.

    DELETE bodies are dropped by enough intermediaries that requiring one
    would make the audit trail unreliable in exactly the deployments that
    most need it -- so the reason has to survive as a query parameter.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)

    await client.put(
        f"/api/v1/admin/settings/{KEY}",
        json={"value": 1200, "reason": "raise it first"},
        headers=headers,
    )
    cleared = await client.delete(
        f"/api/v1/admin/settings/{KEY}",
        params={"reason": "no longer needed"},
        headers=headers,
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["setting"]["global_value"] is None

    history = await client.get(
        "/api/v1/admin/settings/changes", params={"setting_key": KEY}, headers=headers
    )
    latest = history.json()[0]
    assert latest["action"] == "clear"
    assert latest["reason"] == "no longer needed"


async def test_global_clear_without_a_reason_is_rejected(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.delete(f"/api/v1/admin/settings/{KEY}", headers=_auth(token))
    assert resp.status_code == 422, resp.text


async def test_http_preview_rejects_an_unknown_key(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The preview is a real dry run, so it rejects what the apply would."""
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(
        "/api/v1/admin/settings/not.a.real.key/preview",
        json={"value": 1},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


async def test_org_preview_dry_runs_a_change_and_a_clear(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """One endpoint serves both, switched by ``clear``.

    Previewing a clear has to report the value that would be INHERITED rather
    than an empty field, or the confirm dialog tells the operator their tenant
    is about to lose the setting entirely.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)
    org = seeded_users.organization_id

    await client.put(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}",
        json={"value": 1500, "reason": "pin this tenant"},
        headers=headers,
    )

    change = await client.post(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}/preview",
        json={"value": 900},
        headers=headers,
    )
    assert change.status_code == 200, change.text
    assert change.json()["new_value"] == 900
    assert change.json()["current_value"] == 1500
    assert change.json()["affected_organizations"] == 1

    clear = await client.post(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}/preview",
        json={"clear": True},
        headers=headers,
    )
    assert clear.status_code == 200, clear.text
    assert clear.json()["new_value"] is None
    assert clear.json()["unchanged"] is False

    # A dry run writes nothing.
    listed = await client.get(f"/api/v1/admin/organizations/{org}/settings", headers=headers)
    assert next(r for r in listed.json() if r["key"] == KEY)["org_value"] == 1500


async def test_org_preview_rejects_an_out_of_range_value(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(
        f"/api/v1/admin/organizations/{seeded_users.organization_id}/settings/{KEY}/preview",
        json={"value": 99999},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


async def test_org_rollback_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)
    org = seeded_users.organization_id

    first = await client.put(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}",
        json={"value": 700, "reason": "first tenant value"},
        headers=headers,
    )
    assert first.status_code == 200, first.text
    second = await client.put(
        f"/api/v1/admin/organizations/{org}/settings/{KEY}",
        json={"value": 1500, "reason": "second tenant value"},
        headers=headers,
    )
    assert second.status_code == 200, second.text
    change_id = second.json()["change_id"]

    rolled = await client.post(
        f"/api/v1/admin/organizations/{org}/settings/changes/{change_id}/rollback",
        json={"reason": "undo the second one"},
        headers=headers,
    )
    assert rolled.status_code == 200, rolled.text
    assert rolled.json()["setting"]["org_value"] == 700


async def test_org_rollback_refuses_a_change_from_another_scope(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """404, not 403: an org admin must not be able to probe global history.

    A 403 would confirm the change id exists, which is exactly the fact the
    scoping is meant to withhold.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)

    applied = await client.put(
        f"/api/v1/admin/settings/{KEY}",
        json={"value": 1200, "reason": "a global change"},
        headers=headers,
    )
    assert applied.status_code == 200, applied.text
    change_id = applied.json()["change_id"]
    org = seeded_users.organization_id

    resp = await client.post(
        f"/api/v1/admin/organizations/{org}/settings/changes/{change_id}/rollback",
        json={"reason": "not this tenant's to undo"},
        headers=headers,
    )
    assert resp.status_code == 404, resp.text


async def test_org_clear_without_a_reason_is_rejected(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.delete(
        f"/api/v1/admin/organizations/{seeded_users.organization_id}/settings/{KEY}",
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


async def test_org_apply_rejects_an_out_of_range_value(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.put(
        f"/api/v1/admin/organizations/{seeded_users.organization_id}/settings/{KEY}",
        json={"value": 99999, "reason": "far too large"},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


async def test_global_change_history_honours_its_limit_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)
    for value in (700, 900, 1200):
        await client.put(
            f"/api/v1/admin/settings/{KEY}",
            json={"value": value, "reason": f"set to {value}"},
            headers=headers,
        )

    resp = await client.get(
        "/api/v1/admin/settings/changes",
        params={"setting_key": KEY, "limit": 2},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()) == 2


async def test_global_change_history_rejects_a_bad_limit(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.get(
        "/api/v1/admin/settings/changes", params={"limit": 500}, headers=_auth(token)
    )
    assert resp.status_code == 422, resp.text
