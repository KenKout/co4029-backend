from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
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
from abridgeai.features.access_control.queries import (
    load_course_permissions,
    load_user_permissions,
)
from abridgeai.features.access_control.queries.permissions import (
    clear_permissions_cache,
)

_TEST_PERM_CODE = "_test.scope_probe"


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
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@dataclass
class _Scenario:
    perm_id: uuid.UUID
    master_dean_id: uuid.UUID
    other_org_id: uuid.UUID
    other_org_course_id: uuid.UUID
    sibling_unit_id: uuid.UUID
    sibling_unit_course_id: uuid.UUID
    extra_course_same_org_id: uuid.UUID


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[_Scenario]:
    perm_id = uuid.uuid4()
    master_dean_id = uuid.uuid4()
    other_org_id = uuid.uuid4()
    other_org_course_id = uuid.uuid4()
    sibling_unit_id = uuid.uuid4()
    sibling_unit_course_id = uuid.uuid4()
    extra_course_same_org_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO permissions (id, code, name) VALUES (:id, :code, :name)"),
            {"id": perm_id, "code": _TEST_PERM_CODE, "name": "scope probe"},
        )
        await conn.execute(
            text(
                "INSERT INTO role_permissions (role_id, permission_id) "
                "SELECT id, :perm_id FROM roles "
                "WHERE code IN ('student','teacher','hod','manager','admin')"
            ),
            {"perm_id": perm_id},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {
                "id": master_dean_id,
                "email": f"master-dean-{master_dean_id.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(user_id, organization_id, status) VALUES (:id, :org, 'active')"
            ),
            {"id": master_dean_id, "org": seeded_users.organization_id},
        )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id) "
                "SELECT :id, id, 'organization', :org FROM roles WHERE code = 'hod'"
            ),
            {"id": master_dean_id, "org": seeded_users.organization_id},
        )

        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": other_org_id,
                "slug": f"other-{other_org_id.hex[:8]}",
                "name": "Other Org",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) VALUES (:id, :org, :owner, :slug, "
                ":title, 'draft')"
            ),
            {
                "id": other_org_course_id,
                "org": other_org_id,
                "owner": seeded_users.admin_id,
                "slug": f"other-course-{other_org_course_id.hex[:8]}",
                "title": "Other-Org Course",
            },
        )

        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, "
                "name, code) VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": sibling_unit_id,
                "org": seeded_users.organization_id,
                "name": "Sibling Faculty",
                "code": f"SIB-{sibling_unit_id.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, faculty_id, "
                "owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :unit, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": sibling_unit_course_id,
                "org": seeded_users.organization_id,
                "unit": sibling_unit_id,
                "owner": seeded_users.admin_id,
                "slug": f"sib-course-{sibling_unit_course_id.hex[:8]}",
                "title": "Sibling Unit Course",
            },
        )

        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) VALUES (:id, :org, :owner, :slug, "
                ":title, 'draft')"
            ),
            {
                "id": extra_course_same_org_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.admin_id,
                "slug": f"extra-course-{extra_course_same_org_id.hex[:8]}",
                "title": "Extra Org Course (no unit)",
            },
        )

    # Effective permissions are cached process-locally for 30s
    # (queries/permissions.py). This fixture grants the probe by writing
    # role_permissions directly, bypassing the service write paths that
    # normally invalidate — so any user whose permissions were resolved
    # earlier in the session (another test file making authenticated
    # requests) would be answered from a pre-probe cache entry. The grant
    # spans five roles, so every holder is affected: clear the lot.
    clear_permissions_cache()

    yield _Scenario(
        perm_id=perm_id,
        master_dean_id=master_dean_id,
        other_org_id=other_org_id,
        other_org_course_id=other_org_course_id,
        sibling_unit_id=sibling_unit_id,
        sibling_unit_course_id=sibling_unit_course_id,
        extra_course_same_org_id=extra_course_same_org_id,
    )

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = :id"),
            {"id": master_dean_id},
        )
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE user_id = :id"),
            {"id": master_dean_id},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {
                "ids": [
                    other_org_course_id,
                    sibling_unit_course_id,
                    extra_course_same_org_id,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM org_units WHERE id = ANY(:ids)"),
            {"ids": [sibling_unit_id]},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"),
            {"id": other_org_id},
        )
        await conn.execute(
            text("DELETE FROM role_permissions WHERE permission_id = :id"),
            {"id": perm_id},
        )
        await conn.execute(text("DELETE FROM permissions WHERE id = :id"), {"id": perm_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": master_dean_id})


async def test_global_scope_resolves(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        for course_id in (
            seeded_users.course_id,
            scenario.other_org_course_id,
            scenario.sibling_unit_course_id,
        ):
            perms = await load_course_permissions(session, seeded_users.admin_id, course_id)
            assert _TEST_PERM_CODE in perms, (
                f"admin (global) should resolve perm on course {course_id}"
            )


async def test_organization_scope_resolves(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        same_org = await load_course_permissions(
            session, scenario.master_dean_id, seeded_users.course_id
        )
        assert _TEST_PERM_CODE in same_org

        same_org_no_unit = await load_course_permissions(
            session, scenario.master_dean_id, scenario.extra_course_same_org_id
        )
        assert _TEST_PERM_CODE in same_org_no_unit

        other_org = await load_course_permissions(
            session, scenario.master_dean_id, scenario.other_org_course_id
        )
        assert _TEST_PERM_CODE not in other_org


async def test_faculty_scope_resolves_only_its_courses(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        direct = await load_course_permissions(session, seeded_users.hod_id, seeded_users.course_id)
        assert _TEST_PERM_CODE in direct, (
            "Faculty Dean must resolve permissions on a course owned by their Faculty"
        )

        sibling = await load_course_permissions(
            session, seeded_users.hod_id, scenario.sibling_unit_course_id
        )
        assert _TEST_PERM_CODE not in sibling, (
            "Faculty Dean must NOT resolve permissions in an unrelated Faculty"
        )

        no_unit = await load_course_permissions(
            session, seeded_users.hod_id, scenario.extra_course_same_org_id
        )
        assert _TEST_PERM_CODE not in no_unit, (
            "Faculty scope must not leak to an organization-wide course"
        )


async def test_course_scope_resolves(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        target = await load_course_permissions(
            session, seeded_users.teacher_id, seeded_users.course_id
        )
        assert _TEST_PERM_CODE in target

        other = await load_course_permissions(
            session, seeded_users.teacher_id, scenario.sibling_unit_course_id
        )
        assert _TEST_PERM_CODE not in other


async def test_no_role_assignment_returns_empty(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        random_user = uuid.uuid4()
        course_perms = await load_course_permissions(session, random_user, seeded_users.course_id)
        assert course_perms == set()
        user_perms = await load_user_permissions(session, random_user)
        assert user_perms == set()


async def test_active_window_filter(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    expired_user = uuid.uuid4()
    future_user = uuid.uuid4()

    async with engine.begin() as conn:
        for uid in (expired_user, future_user):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"win-{uid.hex[:8]}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, active_from, active_until) "
                "SELECT :uid, id, 'global', :af, :au "
                "FROM roles WHERE code = 'admin'"
            ),
            {
                "uid": expired_user,
                "af": datetime.now(UTC) - timedelta(days=10),
                "au": datetime.now(UTC) - timedelta(days=1),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, active_from, active_until) "
                "SELECT :uid, id, 'global', :af, NULL "
                "FROM roles WHERE code = 'admin'"
            ),
            {
                "uid": future_user,
                "af": datetime.now(UTC) + timedelta(days=1),
            },
        )

    try:
        async with session_factory() as session:
            assert _TEST_PERM_CODE not in await load_user_permissions(session, expired_user)
            assert _TEST_PERM_CODE not in await load_user_permissions(session, future_user)
            assert _TEST_PERM_CODE not in await load_course_permissions(
                session, expired_user, seeded_users.course_id
            )
            assert _TEST_PERM_CODE not in await load_course_permissions(
                session, future_user, seeded_users.course_id
            )

            backdated = datetime.now(UTC) - timedelta(days=5)
            assert _TEST_PERM_CODE in await load_user_permissions(
                session, expired_user, at=backdated
            )
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
                {"ids": [expired_user, future_user]},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = ANY(:ids)"),
                {"ids": [expired_user, future_user]},
            )


async def test_direct_grant_via_user_permission_grants(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    grant_perm_id = uuid.uuid4()
    grant_user = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO permissions (id, code, name) VALUES (:id, :code, :name)"),
            {
                "id": grant_perm_id,
                "code": "_test.direct_grant",
                "name": "direct grant probe",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": grant_user, "email": f"grant-{grant_user.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO user_permission_grants "
                "(user_id, permission_id, scope_kind, organization_id, "
                "course_id) VALUES (:u, :p, 'course', :org, :course)"
            ),
            {
                "u": grant_user,
                "p": grant_perm_id,
                "org": seeded_users.organization_id,
                "course": seeded_users.course_id,
            },
        )

    try:
        async with session_factory() as session:
            user_perms = await load_user_permissions(session, grant_user)
            assert "_test.direct_grant" in user_perms

            target = await load_course_permissions(session, grant_user, seeded_users.course_id)
            assert "_test.direct_grant" in target

            other = await load_course_permissions(session, grant_user, scenario.other_org_course_id)
            assert "_test.direct_grant" not in other
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_permission_grants WHERE user_id = :id"),
                {"id": grant_user},
            )
            await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": grant_user})
            await conn.execute(
                text("DELETE FROM permissions WHERE id = :id"),
                {"id": grant_perm_id},
            )


async def test_load_user_permissions_includes_role_perms(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        admin_perms = await load_user_permissions(session, seeded_users.admin_id)
        assert _TEST_PERM_CODE in admin_perms

        student_perms = await load_user_permissions(session, seeded_users.student_id)
        assert _TEST_PERM_CODE in student_perms


_ = pytest.fixture
