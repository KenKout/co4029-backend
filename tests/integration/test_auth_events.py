"""Typed auth-event log (FR-1.6) — recorder, immutability, and admin search.

The load-bearing property: a committed auth action ALWAYS has its typed event,
written in the SAME transaction as the action. Failure paths whose request
transaction rolls back (login failure, MFA verify failure) commit their event
on a separate transaction — the trail must show the attempt even though the
request's writes were discarded.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.audit.maintenance import audit_maintenance
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.identity.services.auth_events import (
    AUTH_EVENT_TYPES,
    record_auth_event,
    record_auth_event_standalone,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


async def _count(db: AsyncSession, event_type: str) -> int:
    return (
        await db.execute(
            text("SELECT count(*) FROM auth_events WHERE event_type = :t"),
            {"t": event_type},
        )
    ).scalar_one()


async def _purge(db: AsyncSession) -> None:
    """Remove rows this test created (inside the retention scope)."""
    await audit_maintenance(db)
    await db.execute(text("DELETE FROM auth_events"))
    await db.commit()


@pytest.mark.asyncio
async def test_recorder_writes_named_event(db: AsyncSession, seeded_users) -> None:  # noqa: ANN001
    """The recorder names events semantically — login_succeeded, not a path."""
    await record_auth_event(db, event_type="login_succeeded", user_id=seeded_users.student_id)
    await db.commit()
    assert await _count(db, "login_succeeded") >= 1
    await _purge(db)


@pytest.mark.asyncio
async def test_failure_event_survives_rolled_back_request(seeded_users) -> None:  # noqa: ANN001
    """login_failed commits on its OWN transaction.

    A request whose service raises rolls its transaction back (get_db never
    commits) — the event must outlive that rollback.
    """
    from abridgeai.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    assert settings.test_database_url  # under pytest, recorder writes to the TEST db

    await record_auth_event_standalone(
        "login_failed", detail={"reason": "test_reason"}
    )

    async with get_sessionmaker()() as session:
        assert await _count(session, "login_failed") >= 1
        await _purge(session)


@pytest.mark.asyncio
async def test_unknown_event_type_is_rejected(db: AsyncSession) -> None:  # noqa: ANN001
    """A typo cannot create a silent, unqueryable event class."""
    with pytest.raises(ValueError, match="Unknown auth event type"):
        await record_auth_event(db, event_type="login_suceeded")  # typo
    assert "login_suceeded" not in AUTH_EVENT_TYPES


@pytest.mark.asyncio
async def test_auth_events_are_append_only(db: AsyncSession) -> None:  # noqa: ANN001
    """The 0105 rule holds: UPDATE refused, DELETE only inside maintenance."""
    await record_auth_event(db, event_type="logout")
    await db.commit()

    with pytest.raises(Exception, match="append-only"):  # noqa: B017, PT011, PT012
        await db.execute(text("UPDATE auth_events SET event_type = 'logout'"))
        await db.commit()
    await db.rollback()  # the aborted UPDATE leaves the txn unusable

    # DELETE outside the scope is refused too.
    with pytest.raises(Exception, match="append-only"):  # noqa: B017, PT011, PT012
        await db.execute(text("DELETE FROM auth_events"))
        await db.commit()
    await db.rollback()

    await _purge(db)


@pytest.mark.asyncio
async def test_role_change_events_carry_actor_and_org(db: AsyncSession, seeded_users) -> None:  # noqa: ANN001
    """Access-control events name both the subject and the performing admin."""

    org_id = (
        await db.execute(text("SELECT id FROM organizations ORDER BY created_at LIMIT 1"))
    ).scalar_one()
    assert org_id is not None
    await record_auth_event(
        db,
        event_type="role_assigned",
        user_id=seeded_users.student_id,
        actor_user_id=seeded_users.admin_id,
        organization_id=org_id,
        detail={"role_code": "teacher", "scope_kind": "organization"},
    )
    await db.commit()
    row = (
        await db.execute(
            text(
                "SELECT actor_user_id, organization_id, detail->>'role_code' "
                "FROM auth_events WHERE event_type = 'role_assigned' "
                "ORDER BY occurred_at DESC LIMIT 1"
            )
        )
    ).one()
    assert row[0] == seeded_users.admin_id
    assert row[1] == org_id
    assert row[2] == "teacher"
    await _purge(db)


@pytest.mark.asyncio
async def test_admin_auth_events_endpoint_returns_typed_rows(db: AsyncSession, seeded_users) -> None:  # noqa: ANN001
    """GET /admin/audit/auth-events returns the event with its semantic name."""
    from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

    from abridgeai.api import create_app  # noqa: PLC0415

    await record_auth_event(
        db,
        event_type="mfa_verified",
        user_id=seeded_users.student_id,
        detail={"method": "totp"},
    )
    await db.commit()

    # A session for the admin caller.
    sid = uuid4()
    refresh = generate_token()
    await db.execute(
        text(
            "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
            "VALUES (:s, :u, :h, NOW() + interval '1 hour')"
        ),
        {"s": sid, "u": seeded_users.admin_id, "h": hash_secret(refresh)},
    )
    await db.commit()

    app = create_app()
    token = create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    since = (datetime.now(tz=UTC) - timedelta(minutes=5)).isoformat()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/api/v1/admin/audit/auth-events",
            params={"since": since, "event_type": "mfa_verified"},
            headers={"authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert rows, "expected at least one mfa_verified row"
    assert rows[0]["event_type"] == "mfa_verified"
    assert rows[0]["detail"]["method"] == "totp"

    await audit_maintenance(db)
    await db.execute(text("DELETE FROM auth_sessions WHERE id = :s"), {"s": sid})
    await db.commit()
    await _purge(db)
