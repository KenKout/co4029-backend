"""Integration tests for the Redis cache infrastructure (T0.29)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from abridgeai.core.cache import (
    INVALIDATION_RULES,
    PERM_USER,
    SESSION,
    SESSION_BY_REFRESH_HASH,
    USER_SESSION_CASCADE_MODELS,
    USER_SESSIONS_INDEX,
    USER_STATUS,
    CacheKey,
    cached,
    drain_invalidations,
    get_cache,
    read_session_cache,
    register_cache_invalidator,
)
from abridgeai.core.cache.client import reset_cache_client_for_tests

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380/15")


class _Base(DeclarativeBase):
    pass


class FakeUserRoleAssignment(_Base):
    __tablename__ = "_cache_test_user_role_assignment"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))


class FakeAuthSession(_Base):
    __tablename__ = "_cache_test_auth_session"
    session_id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64))


class FakeUser(_Base):
    __tablename__ = "_cache_test_user"
    id: Mapped[str] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="active")


@pytest.fixture(autouse=True)
def _redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)
    from abridgeai.core import config as config_mod

    config_mod.get_settings.cache_clear()
    reset_cache_client_for_tests()
    yield
    reset_cache_client_for_tests()
    config_mod.get_settings.cache_clear()


@pytest_asyncio.fixture
async def redis_clean() -> Any:
    client = get_cache()
    await client.flushdb()
    yield client
    await client.flushdb()


@pytest.fixture
def invalidator_rules() -> Any:
    snapshot_rules = dict(INVALIDATION_RULES)
    snapshot_cascades = set(USER_SESSION_CASCADE_MODELS)
    INVALIDATION_RULES.clear()
    USER_SESSION_CASCADE_MODELS.clear()
    INVALIDATION_RULES[FakeUserRoleAssignment] = [PERM_USER]
    INVALIDATION_RULES[FakeAuthSession] = [SESSION, SESSION_BY_REFRESH_HASH]
    INVALIDATION_RULES[FakeUser] = [USER_STATUS, PERM_USER]
    USER_SESSION_CASCADE_MODELS.add(FakeUser)
    register_cache_invalidator()
    yield
    INVALIDATION_RULES.clear()
    USER_SESSION_CASCADE_MODELS.clear()
    INVALIDATION_RULES.update(snapshot_rules)
    USER_SESSION_CASCADE_MODELS.update(snapshot_cascades)


@pytest.fixture
def sqlite_session(invalidator_rules: None) -> Any:
    engine = create_engine("sqlite://")
    _Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    _Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.mark.asyncio
async def test_hit_miss_flow(redis_clean: Any) -> None:
    counter = {"calls": 0}

    @cached(PERM_USER)
    async def load_perms(user_id: str) -> dict[str, Any]:
        counter["calls"] += 1
        return {"user_id": user_id, "perms": ["read"]}

    first = await load_perms(user_id="u-1")
    second = await load_perms(user_id="u-1")

    assert first == second
    assert counter["calls"] == 1


@pytest.mark.asyncio
async def test_invalidation_after_write(redis_clean: Any, sqlite_session: Session) -> None:
    key = PERM_USER.format(user_id="u-42")
    await redis_clean.set(key, json.dumps({"cached": True}), ex=60)
    assert await redis_clean.exists(key) == 1

    sqlite_session.add(FakeUserRoleAssignment(user_id="u-42"))
    sqlite_session.flush()
    await drain_invalidations()

    assert await redis_clean.exists(key) == 0


@pytest.mark.asyncio
async def test_redis_down_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1/0")
    from abridgeai.core import config as config_mod

    config_mod.get_settings.cache_clear()
    reset_cache_client_for_tests()

    counter = {"calls": 0}

    @cached(PERM_USER)
    async def load_perms(user_id: str) -> str:
        counter["calls"] += 1
        return f"value-{user_id}"

    result = await load_perms(user_id="u-down")
    assert result == "value-u-down"
    assert counter["calls"] == 1


@pytest.mark.asyncio
async def test_no_cross_user_leak(redis_clean: Any) -> None:
    counter = {"calls": 0}

    @cached(PERM_USER)
    async def load_perms(user_id: str) -> dict[str, str]:
        counter["calls"] += 1
        return {"user": user_id}

    a1 = await load_perms(user_id="user-A")
    b1 = await load_perms(user_id="user-B")
    a2 = await load_perms(user_id="user-A")
    b2 = await load_perms(user_id="user-B")

    assert a1 == {"user": "user-A"} == a2
    assert b1 == {"user": "user-B"} == b2
    assert counter["calls"] == 2


@pytest.mark.asyncio
async def test_jwt_validation_cache_hit(redis_clean: Any) -> None:
    db_calls = {"count": 0}

    @cached(SESSION)
    async def load_session(session_id: str) -> dict[str, Any]:
        db_calls["count"] += 1
        return {"session_id": session_id, "user_id": "u-7", "revoked_at": None}

    sid = "sess-abc"
    first = await load_session(session_id=sid)
    second = await load_session(session_id=sid)

    assert first == second
    assert db_calls["count"] == 1


@pytest.mark.asyncio
async def test_logout_invalidates_session(redis_clean: Any, sqlite_session: Session) -> None:
    sid = "sess-logout"
    key = SESSION.format(session_id=sid)
    await redis_clean.set(key, json.dumps({"session_id": sid, "revoked_at": None}), ex=60)
    assert await redis_clean.exists(key) == 1

    sqlite_session.add(FakeAuthSession(session_id=sid, user_id="u-9"))
    sqlite_session.flush()
    await drain_invalidations()

    assert await redis_clean.exists(key) == 0


@pytest.mark.asyncio
async def test_admin_disable_cascades_user_sessions(
    redis_clean: Any, sqlite_session: Session
) -> None:
    user_id = "u-disabled"
    sid_1 = "sess-1"
    sid_2 = "sess-2"
    sid_3 = "sess-3"

    index_key = USER_SESSIONS_INDEX.format(user_id=user_id)
    await redis_clean.sadd(index_key, sid_1, sid_2, sid_3)
    for sid in (sid_1, sid_2, sid_3):
        await redis_clean.set(
            SESSION.format(session_id=sid),
            json.dumps({"session_id": sid, "revoked_at": None}),
            ex=60,
        )
    await redis_clean.set(
        USER_STATUS.format(user_id=user_id),
        json.dumps("active"),
        ex=60,
    )

    sqlite_session.add(FakeUser(id=user_id, status="disabled"))
    sqlite_session.flush()
    await drain_invalidations()

    for sid in (sid_1, sid_2, sid_3):
        assert await redis_clean.exists(SESSION.format(session_id=sid)) == 0
    assert await redis_clean.exists(index_key) == 0
    assert await redis_clean.exists(USER_STATUS.format(user_id=user_id)) == 0


@pytest.mark.asyncio
async def test_revoked_session_warn_log(redis_clean: Any, caplog: pytest.LogCaptureFixture) -> None:
    sid = "sess-stale"
    key = SESSION.format(session_id=sid)
    await redis_clean.set(
        key,
        json.dumps({"session_id": sid, "revoked_at": "2026-05-16T12:00:00Z"}),
        ex=60,
    )

    with caplog.at_level(logging.WARNING, logger="abridgeai.core.cache.client"):
        payload = await read_session_cache(sid)

    assert payload is not None
    assert payload.get("revoked_at") == "2026-05-16T12:00:00Z"
    matching = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING
        and getattr(rec, "event", None) == "cache_revoked_session_hit"
    ]
    assert matching, "expected WARN log with event=cache_revoked_session_hit"
    assert getattr(matching[0], "session_id", None) == sid


@pytest.mark.asyncio
async def test_cache_key_format_missing_placeholder() -> None:
    with pytest.raises(KeyError):
        PERM_USER.format()


@pytest.mark.asyncio
async def test_cache_key_placeholders_introspection() -> None:
    assert PERM_USER.placeholders == ("user_id",)
    assert CacheKey(pattern="x:{a}:{b}", ttl_seconds=1, description="").placeholders == (
        "a",
        "b",
    )
