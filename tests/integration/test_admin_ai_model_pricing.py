"""Coverage for ``admin/services/ai_model_pricing.py`` and its router.

This table is what turns token counts into the money figures on the AI-costs
dashboard, and it is the config replacement for what used to be a hardcoded
PRICE_TABLE. Two things make it worth testing beyond plain CRUD:

* **Cache invalidation.** Rates are cached in-process behind a TTL. Every
  write busts that cache so a corrected rate applies to the very next call
  rather than up to a TTL later. Without it, an operator fixing a wrong price
  watches the dashboard keep reporting the wrong number and reasonably
  concludes the save failed.

* **PATCH's three-way ``notes``.** ``notes`` absent means "leave it", ``notes:
  null`` means "clear it", and a string means "replace it". Collapsing the
  first two -- the obvious implementation -- silently erases an operator's
  note every time they adjust a rate.

Isolation: the suite shares one Postgres and this table is global, so every
test uses a unique model name and deletes its own rows.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
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

from abridgeai.ai.llm.pricing import compute_cost, invalidate_pricing_cache
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.admin.routers import ai_pricing_router
from abridgeai.features.admin.services import ai_model_pricing as pricing_service


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
    fastapi_app.include_router(ai_pricing_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


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
    return create_access_token(user_id=user_id, session_id=sid)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
def model_name() -> str:
    """A name no other test or seed owns, so counts and lookups are exact."""
    return f"test-model-{uuid.uuid4().hex[:12]}"


@pytest_asyncio.fixture(autouse=True)
async def _drop_test_models(engine: AsyncEngine) -> AsyncIterator[None]:
    yield
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM ai_model_pricing WHERE model_name LIKE 'test-model-%'")
        )
    # The cache is process-local and keyed by model name; a dropped row that
    # stays cached would follow this test into the next one.
    invalidate_pricing_cache()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


async def test_create_then_list_returns_the_row(db: AsyncSession, model_name: str) -> None:
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.50"),
        output_usd_per_1m=Decimal("10.00"),
        notes="list price at time of writing",
        updated_by=None,
    )
    assert created["model_name"] == model_name
    assert created["input_usd_per_1m"] == 2.50
    assert created["output_usd_per_1m"] == 10.00

    listed = await pricing_service.list_pricing(db)
    assert any(r["id"] == created["id"] for r in listed)


async def test_list_is_ordered_by_model_name(db: AsyncSession) -> None:
    """Stable ordering: the page is a reference table an operator scans."""
    suffix = uuid.uuid4().hex[:8]
    for name in (f"test-model-{suffix}-c", f"test-model-{suffix}-a", f"test-model-{suffix}-b"):
        await pricing_service.create_pricing(
            db,
            model_name=name,
            input_usd_per_1m=Decimal("1"),
            output_usd_per_1m=Decimal("1"),
            notes=None,
            updated_by=None,
        )

    mine = [
        r["model_name"]
        for r in await pricing_service.list_pricing(db)
        if r["model_name"].startswith(f"test-model-{suffix}")
    ]
    assert mine == sorted(mine)


async def test_duplicate_model_name_is_a_conflict_not_a_500(
    db: AsyncSession, model_name: str
) -> None:
    """One rate per model. The second insert must surface as a conflict.

    The unique violation has to be caught and translated, and the session
    rolled back -- otherwise the request fails with a raw IntegrityError and
    leaves the session unusable for anything after it.
    """
    kwargs = {
        "model_name": model_name,
        "input_usd_per_1m": Decimal("1.00"),
        "output_usd_per_1m": Decimal("2.00"),
        "notes": None,
        "updated_by": None,
    }
    await pricing_service.create_pricing(db, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(ConflictError):
        await pricing_service.create_pricing(db, **kwargs)  # type: ignore[arg-type]

    # The session survived the rollback and can still be used.
    assert await pricing_service.list_pricing(db) is not None


async def test_update_changes_only_what_was_given(db: AsyncSession, model_name: str) -> None:
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.00"),
        output_usd_per_1m=Decimal("8.00"),
        notes="original note",
        updated_by=None,
    )

    updated = await pricing_service.update_pricing(
        db,
        created["id"],
        input_usd_per_1m=Decimal("3.00"),
        output_usd_per_1m=None,
        notes=None,
        notes_provided=False,
        updated_by=None,
    )
    assert updated["input_usd_per_1m"] == 3.00
    assert updated["output_usd_per_1m"] == 8.00, "an omitted rate must not be zeroed"
    assert updated["notes"] == "original note", "an omitted note must not be erased"


async def test_notes_can_be_cleared_explicitly(db: AsyncSession, model_name: str) -> None:
    """``notes_provided`` is what separates "clear it" from "leave it".

    Both arrive at the service as ``notes=None``; only the flag distinguishes
    them. Drop it and an operator can never remove a stale note -- or, worse,
    every rate edit wipes one.
    """
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.00"),
        output_usd_per_1m=Decimal("8.00"),
        notes="stale note",
        updated_by=None,
    )

    updated = await pricing_service.update_pricing(
        db,
        created["id"],
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        notes=None,
        notes_provided=True,
        updated_by=None,
    )
    assert updated["notes"] is None


async def test_update_records_who_changed_it(
    db: AsyncSession, model_name: str, seeded_users: SeededUsers
) -> None:
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.00"),
        output_usd_per_1m=Decimal("8.00"),
        notes=None,
        updated_by=None,
    )
    updated = await pricing_service.update_pricing(
        db,
        created["id"],
        input_usd_per_1m=Decimal("4.00"),
        output_usd_per_1m=None,
        notes=None,
        notes_provided=False,
        updated_by=seeded_users.admin_id,
    )
    assert updated["updated_by"] == seeded_users.admin_id


async def test_update_and_delete_of_a_missing_row_are_not_found(db: AsyncSession) -> None:
    missing = uuid.uuid4()
    with pytest.raises(NotFoundError):
        await pricing_service.update_pricing(
            db,
            missing,
            input_usd_per_1m=Decimal("1"),
            output_usd_per_1m=None,
            notes=None,
            notes_provided=False,
            updated_by=None,
        )
    with pytest.raises(NotFoundError):
        await pricing_service.delete_pricing(db, missing)


async def test_delete_removes_the_row(db: AsyncSession, model_name: str) -> None:
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("1"),
        output_usd_per_1m=Decimal("1"),
        notes=None,
        updated_by=None,
    )
    await pricing_service.delete_pricing(db, created["id"])
    assert not [r for r in await pricing_service.list_pricing(db) if r["id"] == created["id"]]


# ---------------------------------------------------------------------------
# Cache invalidation -- the reason this service exists rather than plain CRUD
# ---------------------------------------------------------------------------


async def test_a_new_rate_applies_to_the_very_next_cost_calculation(
    db: AsyncSession, model_name: str
) -> None:
    """Creating a rate must not wait out the pricing cache's TTL.

    The cache is warmed here on purpose: reading a cost for the model BEFORE
    the row exists loads and caches a table that lacks it. Without the bust,
    the model would keep costing nothing until the TTL lapsed.
    """
    assert await compute_cost(db, model_name, 1_000_000, 0) is None

    await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.00"),
        output_usd_per_1m=Decimal("10.00"),
        notes=None,
        updated_by=None,
    )

    assert await compute_cost(db, model_name, 1_000_000, 1_000_000) == Decimal("12.00")


async def test_a_corrected_rate_applies_immediately(db: AsyncSession, model_name: str) -> None:
    """The case an operator actually hits: a wrong price, fixed.

    Without the invalidation the dashboard keeps reporting the old number
    after a save that reported success, which reads as a broken save.
    """
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.00"),
        output_usd_per_1m=Decimal("0"),
        notes=None,
        updated_by=None,
    )
    assert await compute_cost(db, model_name, 1_000_000, 0) == Decimal("2.00")

    await pricing_service.update_pricing(
        db,
        created["id"],
        input_usd_per_1m=Decimal("5.00"),
        output_usd_per_1m=None,
        notes=None,
        notes_provided=False,
        updated_by=None,
    )
    assert await compute_cost(db, model_name, 1_000_000, 0) == Decimal("5.00")


async def test_a_deleted_rate_stops_being_charged_immediately(
    db: AsyncSession, model_name: str
) -> None:
    created = await pricing_service.create_pricing(
        db,
        model_name=model_name,
        input_usd_per_1m=Decimal("2.00"),
        output_usd_per_1m=Decimal("0"),
        notes=None,
        updated_by=None,
    )
    assert await compute_cost(db, model_name, 1_000_000, 0) == Decimal("2.00")

    await pricing_service.delete_pricing(db, created["id"])
    assert await compute_cost(db, model_name, 1_000_000, 0) is None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


async def test_pricing_crud_over_http(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    model_name: str,
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)

    created = await client.post(
        "/api/v1/admin/ai/pricing",
        json={
            "model_name": model_name,
            "input_usd_per_1m": 2.5,
            "output_usd_per_1m": 10,
            "notes": "list price",
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text
    pricing_id = created.json()["id"]
    assert created.json()["updated_by"] == str(seeded_users.admin_id)

    listed = await client.get("/api/v1/admin/ai/pricing", headers=headers)
    assert listed.status_code == 200, listed.text
    assert any(r["id"] == pricing_id for r in listed.json())

    patched = await client.patch(
        f"/api/v1/admin/ai/pricing/{pricing_id}",
        json={"input_usd_per_1m": 3.25},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["input_usd_per_1m"] == 3.25
    assert patched.json()["notes"] == "list price", "PATCH must not erase an untouched note"

    deleted = await client.delete(f"/api/v1/admin/ai/pricing/{pricing_id}", headers=headers)
    assert deleted.status_code == 204, deleted.text

    after = await client.get("/api/v1/admin/ai/pricing", headers=headers)
    assert not [r for r in after.json() if r["id"] == pricing_id]


async def test_http_duplicate_is_409(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    model_name: str,
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    body = {"model_name": model_name, "input_usd_per_1m": 1, "output_usd_per_1m": 1}
    first = await client.post("/api/v1/admin/ai/pricing", json=body, headers=_auth(token))
    assert first.status_code == 201, first.text

    second = await client.post("/api/v1/admin/ai/pricing", json=body, headers=_auth(token))
    assert second.status_code == 409, second.text


async def test_http_patch_and_delete_of_a_missing_row_are_404(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    missing = uuid.uuid4()
    patched = await client.patch(
        f"/api/v1/admin/ai/pricing/{missing}",
        json={"input_usd_per_1m": 1},
        headers=_auth(token),
    )
    assert patched.status_code == 404, patched.text

    deleted = await client.delete(f"/api/v1/admin/ai/pricing/{missing}", headers=_auth(token))
    assert deleted.status_code == 404, deleted.text


@pytest.mark.parametrize(
    "body",
    [
        {"model_name": "", "input_usd_per_1m": 1, "output_usd_per_1m": 1},
        {"model_name": "m", "input_usd_per_1m": -1, "output_usd_per_1m": 1},
        {"model_name": "m", "input_usd_per_1m": 1, "output_usd_per_1m": -1},
    ],
)
async def test_http_rejects_malformed_pricing(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    body: dict[str, object],
) -> None:
    """A negative rate would make the costs dashboard report negative spend."""
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post("/api/v1/admin/ai/pricing", json=body, headers=_auth(token))
    assert resp.status_code == 422, resp.text


async def test_writes_require_system_administer(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    model_name: str,
) -> None:
    """Reads are open to operators; writes are not.

    Pricing drives every money figure in the product, so changing it is a
    deployment-operator action even though reading it is not.
    """
    token = await _bearer(engine, seeded_users.manager_id)
    resp = await client.post(
        "/api/v1/admin/ai/pricing",
        json={"model_name": model_name, "input_usd_per_1m": 1, "output_usd_per_1m": 1},
        headers=_auth(token),
    )
    assert resp.status_code == 403, resp.text
