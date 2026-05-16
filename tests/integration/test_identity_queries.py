from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from conftest import SeededUsers
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.features.identity.queries import (
    get_active_challenge,
    get_profile,
    get_session_by_refresh_hash,
    get_user,
    get_user_by_email,
    get_verified_totp_factor,
    user_has_verified_mfa,
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
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def test_get_user_by_id_returns_seeded_fixture(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
) -> None:
    async with session_factory() as session:
        user = await get_user(session, seeded_users.student_id)
    assert user is not None
    assert user.id == seeded_users.student_id
    assert user.primary_email == "test-student@abridgeai.local"


async def test_get_user_by_email_case_insensitive(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
) -> None:
    async with session_factory() as session:
        user = await get_user_by_email(session, "TEST-STUDENT@ABRIDGEAI.LOCAL")
    assert user is not None
    assert user.id == seeded_users.student_id


async def test_get_profile_returns_none_for_unknown_user(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        profile = await get_profile(session, uuid.uuid4())
    assert profile is None


@pytest_asyncio.fixture
async def sessions_pair(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[str, str]]:
    active_hash = f"active-{uuid.uuid4().hex}"
    revoked_hash = f"revoked-{uuid.uuid4().hex}"
    expires = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": uuid.uuid4(),
                "uid": seeded_users.student_id,
                "h": active_hash,
                "exp": expires,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at, revoked_at) "
                "VALUES (:id, :uid, :h, :exp, NOW())"
            ),
            {
                "id": uuid.uuid4(),
                "uid": seeded_users.student_id,
                "h": revoked_hash,
                "exp": expires,
            },
        )
    yield active_hash, revoked_hash
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE refresh_token_hash IN (:a, :r)"),
            {"a": active_hash, "r": revoked_hash},
        )


async def test_get_session_by_refresh_hash_filters_revoked(
    session_factory: async_sessionmaker[AsyncSession],
    sessions_pair: tuple[str, str],
) -> None:
    active_hash, revoked_hash = sessions_pair
    async with session_factory() as session:
        active = await get_session_by_refresh_hash(session, active_hash)
        revoked = await get_session_by_refresh_hash(session, revoked_hash)
    assert active is not None
    assert active.refresh_token_hash == active_hash
    assert revoked is None


@pytest_asyncio.fixture
async def mfa_setup(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, uuid.UUID]]:
    factor_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(tz=UTC)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO mfa_factors "
                "(id, user_id, factor_type, secret_encrypted, verified_at) "
                "VALUES (:id, :uid, 'totp', 'enc', :now)"
            ),
            {"id": factor_id, "uid": seeded_users.teacher_id, "now": now},
        )
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": seeded_users.teacher_id,
                "h": f"mfa-sess-{session_id.hex}",
                "exp": now + timedelta(hours=1),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO mfa_challenges "
                "(id, user_id, factor_id, session_id, expires_at) "
                "VALUES (:id, :uid, :fid, :sid, :exp)"
            ),
            {
                "id": uuid.uuid4(),
                "uid": seeded_users.teacher_id,
                "fid": factor_id,
                "sid": session_id,
                "exp": now + timedelta(minutes=5),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO mfa_challenges "
                "(id, user_id, factor_id, session_id, expires_at) "
                "VALUES (:id, :uid, :fid, :sid, :exp)"
            ),
            {
                "id": uuid.uuid4(),
                "uid": seeded_users.teacher_id,
                "fid": factor_id,
                "sid": session_id,
                "exp": now - timedelta(minutes=5),
            },
        )
    yield factor_id, session_id
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM mfa_challenges WHERE factor_id = :fid"),
            {"fid": factor_id},
        )
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = :sid"),
            {"sid": session_id},
        )
        await conn.execute(
            text("DELETE FROM mfa_factors WHERE id = :fid"),
            {"fid": factor_id},
        )


async def test_user_has_verified_mfa_and_factor_lookup(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    mfa_setup: tuple[uuid.UUID, uuid.UUID],
) -> None:
    factor_id, _session_id = mfa_setup
    async with session_factory() as session:
        has_mfa = await user_has_verified_mfa(session, seeded_users.teacher_id)
        no_mfa = await user_has_verified_mfa(session, seeded_users.student_id)
        factor = await get_verified_totp_factor(session, seeded_users.teacher_id)
    assert has_mfa is True
    assert no_mfa is False
    assert factor is not None
    assert factor.id == factor_id


async def test_get_active_challenge_filters_expired(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    mfa_setup: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _factor_id, session_id = mfa_setup
    async with session_factory() as session:
        challenge = await get_active_challenge(session, seeded_users.teacher_id, session_id)
    assert challenge is not None
    assert challenge.consumed_at is None
