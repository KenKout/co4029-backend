"""Membership-first primary-org resolution + ORM equivalence checks.

Companion to ``test_access_control_full.py`` — focuses narrowly on
:func:`access_control.api.public.get_user_primary_org` and
:func:`is_user_member_of_org` after the membership-first refactor.

Each test seeds the minimum row set inside its own transaction and
cleans up via DELETE so the suite stays isolated from the shared
``seeded_users`` fixture.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.features.access_control.api.public import (
    get_user_primary_org,
    is_user_member_of_org,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture(scope="module")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def _seed_user(engine: AsyncEngine, user_id: uuid.UUID, email: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": user_id, "em": email},
        )


async def _seed_org(engine: AsyncEngine, org_id: uuid.UUID, slug: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {"id": org_id, "slug": slug, "name": f"Membership-Test Org {slug}"},
        )


async def _seed_membership(
    engine: AsyncEngine,
    *,
    user_id: uuid.UUID,
    org_id: uuid.UUID,
    status: str = "active",
    soft_deleted: bool = False,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, status, deleted_at) "
                "VALUES (:id, :uid, :oid, :status, "
                "CASE WHEN :soft THEN NOW() ELSE NULL END)"
            ),
            {
                "id": uuid.uuid4(),
                "uid": user_id,
                "oid": org_id,
                "status": status,
                "soft": soft_deleted,
            },
        )


async def _seed_global_admin_assignment(engine: AsyncEngine, user_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind) "
                "SELECT :uid, r.id, 'global' FROM roles r WHERE r.code = 'admin'"
            ),
            {"uid": user_id},
        )


async def _seed_org_role_assignment(
    engine: AsyncEngine,
    *,
    user_id: uuid.UUID,
    org_id: uuid.UUID,
    role_code: str = "teacher",
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id) "
                "SELECT :uid, r.id, 'organization', :oid "
                "FROM roles r WHERE r.code = :role_code"
            ),
            {"uid": user_id, "oid": org_id, "role_code": role_code},
        )


async def _cleanup(engine: AsyncEngine, user_ids: list[uuid.UUID], org_ids: list[uuid.UUID]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE user_id = ANY(:ids)"),
            {"ids": user_ids},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
            {"ids": user_ids},
        )
        await conn.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": user_ids})
        await conn.execute(
            text("DELETE FROM organizations WHERE id = ANY(:ids)"), {"ids": org_ids}
        )


# --------------------------------------------------------------------------
# get_user_primary_org
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_org_via_membership(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Active membership → returns that org."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"membership-only-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"member-only-{org_id.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=org_id)

    try:
        async with session_factory() as session:
            org = await get_user_primary_org(session, user_id)
        assert org is not None
        assert org.id == org_id
        assert org.status == "active"
    finally:
        await _cleanup(engine, [user_id], [org_id])


@pytest.mark.asyncio
async def test_primary_org_role_alone_returns_none(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Org-scoped role WITHOUT membership → None.

    Membership is the sole source of truth. Permissions-in-org and
    belonging-to-org are independent concepts.
    """
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"role-only-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"role-only-{org_id.hex[:6]}")
    await _seed_org_role_assignment(engine, user_id=user_id, org_id=org_id)

    try:
        async with session_factory() as session:
            org = await get_user_primary_org(session, user_id)
        assert org is None, "role assignments must NOT confer primary-org membership"
    finally:
        await _cleanup(engine, [user_id], [org_id])


@pytest.mark.asyncio
async def test_primary_org_membership_wins_when_role_points_elsewhere(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Membership in org A + role in org B → membership org A wins."""
    user_id = uuid.uuid4()
    membership_org = uuid.uuid4()
    role_org = uuid.uuid4()
    await _seed_user(engine, user_id, f"both-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, membership_org, f"both-mem-{membership_org.hex[:6]}")
    await _seed_org(engine, role_org, f"both-role-{role_org.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=membership_org)
    await _seed_org_role_assignment(engine, user_id=user_id, org_id=role_org)

    try:
        async with session_factory() as session:
            org = await get_user_primary_org(session, user_id)
        assert org is not None
        assert org.id == membership_org
    finally:
        await _cleanup(engine, [user_id], [membership_org, role_org])


@pytest.mark.asyncio
async def test_primary_org_global_admin_returns_none(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Global admin with no membership → None."""
    user_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"global-admin-{user_id.hex[:6]}@test.local")
    await _seed_global_admin_assignment(engine, user_id)

    try:
        async with session_factory() as session:
            org = await get_user_primary_org(session, user_id)
        assert org is None, "global admin must NOT be implicitly attached to any org"
    finally:
        await _cleanup(engine, [user_id], [])


@pytest.mark.asyncio
async def test_primary_org_inactive_membership_returns_none(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Inactive membership + role → None (no fallback)."""
    user_id = uuid.uuid4()
    membership_org = uuid.uuid4()
    role_org = uuid.uuid4()
    await _seed_user(engine, user_id, f"inactive-mem-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, membership_org, f"inactive-mem-{membership_org.hex[:6]}")
    await _seed_org(engine, role_org, f"inactive-role-{role_org.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=membership_org, status="inactive")
    await _seed_org_role_assignment(engine, user_id=user_id, org_id=role_org)

    try:
        async with session_factory() as session:
            org = await get_user_primary_org(session, user_id)
        assert org is None, "inactive membership skipped, no role fallback"
    finally:
        await _cleanup(engine, [user_id], [membership_org, role_org])


@pytest.mark.asyncio
async def test_primary_org_soft_deleted_membership_excluded(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Soft-deleted membership → ignored; returns None."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"soft-del-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"soft-del-{org_id.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=org_id, soft_deleted=True)

    try:
        async with session_factory() as session:
            org = await get_user_primary_org(session, user_id)
        assert org is None
    finally:
        await _cleanup(engine, [user_id], [org_id])


# --------------------------------------------------------------------------
# is_user_member_of_org
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_member_of_org_active(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"is-mem-active-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"is-mem-active-{org_id.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=org_id)

    try:
        async with session_factory() as session:
            assert await is_user_member_of_org(session, user_id=user_id, org_id=org_id) is True
    finally:
        await _cleanup(engine, [user_id], [org_id])


@pytest.mark.asyncio
async def test_is_member_of_org_inactive_returns_false(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"is-mem-inactive-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"is-mem-inactive-{org_id.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=org_id, status="inactive")

    try:
        async with session_factory() as session:
            assert await is_user_member_of_org(session, user_id=user_id, org_id=org_id) is False
    finally:
        await _cleanup(engine, [user_id], [org_id])


@pytest.mark.asyncio
async def test_is_member_of_org_soft_deleted_returns_false(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"is-mem-soft-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"is-mem-soft-{org_id.hex[:6]}")
    await _seed_membership(engine, user_id=user_id, org_id=org_id, soft_deleted=True)

    try:
        async with session_factory() as session:
            assert await is_user_member_of_org(session, user_id=user_id, org_id=org_id) is False
    finally:
        await _cleanup(engine, [user_id], [org_id])


@pytest.mark.asyncio
async def test_is_member_of_org_no_membership_returns_false(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    await _seed_user(engine, user_id, f"is-mem-none-{user_id.hex[:6]}@test.local")
    await _seed_org(engine, org_id, f"is-mem-none-{org_id.hex[:6]}")

    try:
        async with session_factory() as session:
            assert await is_user_member_of_org(session, user_id=user_id, org_id=org_id) is False
    finally:
        await _cleanup(engine, [user_id], [org_id])
