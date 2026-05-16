"""End-to-end access-control integration suite (T1.13).

Validates the four scope kinds end-to-end against the canonical query
(:mod:`abridgeai.features.access_control.queries.permissions`) and the admin
router (T1.10) by driving real HTTPX requests + raw inserts against the
docker postgres on port 5433.

Companion to ``test_identity_full.py``: identity is the OAuth + session +
admin lookup surface; access-control is the role assignment + permission
resolution surface. Together they constitute the Phase 1 exit gate.

Self-contained module-scoped fixtures avoid depending on the session-scoped
``seeded_users`` from ``tests/conftest.py`` because the destructive
``test_catalog_seed_migration`` round-trip can invalidate that data when it
runs in the same suite.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.access_control.queries import (
    load_course_permissions,
    load_user_permissions,
)
from abridgeai.features.access_control.routers.admin import router as admin_router


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


@dataclass(frozen=True)
class _Scenario:
    organization_id: uuid.UUID
    org_unit_id: uuid.UUID
    course_id: uuid.UUID
    student_id: uuid.UUID
    teacher_id: uuid.UUID
    hod_id: uuid.UUID
    manager_id: uuid.UUID
    admin_id: uuid.UUID


@pytest_asyncio.fixture(scope="module")
async def scenario(engine: AsyncEngine) -> AsyncIterator[_Scenario]:
    organization_id = uuid.uuid4()
    org_unit_id = uuid.uuid4()
    course_id = uuid.uuid4()
    user_ids: dict[str, uuid.UUID] = {
        "student": uuid.uuid4(),
        "teacher": uuid.uuid4(),
        "hod": uuid.uuid4(),
        "manager": uuid.uuid4(),
        "admin": uuid.uuid4(),
    }

    role_assignments = (
        ("student", "organization", organization_id, None, None),
        ("teacher", "course", organization_id, None, course_id),
        ("hod", "org_unit", organization_id, org_unit_id, None),
        ("manager", "organization", organization_id, None, None),
        ("admin", "global", None, None, None),
    )

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": organization_id,
                "slug": f"t113-ac-{organization_id.hex[:8]}",
                "name": "T1.13 AC Suite Org",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'department', :name, :code)"
            ),
            {
                "id": org_unit_id,
                "org": organization_id,
                "name": "T1.13 AC Dept",
                "code": f"T113-AC-{org_unit_id.hex[:6]}",
            },
        )
        for role, uid in user_ids.items():
            await conn.execute(
                text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
                {"id": uid, "em": f"t113-ac-{role}-{uid.hex[:6]}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, org_unit_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :unit, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": organization_id,
                "unit": org_unit_id,
                "owner": user_ids["admin"],
                "slug": f"t113-ac-course-{course_id.hex[:8]}",
                "title": "T1.13 AC Course",
            },
        )
        for role_code, scope_kind, org_id, ou_id, c_id in role_assignments:
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments "
                    "(user_id, role_id, scope_kind, organization_id, "
                    "org_unit_id, course_id) "
                    "SELECT :uid, r.id, :scope_kind, :organization_id, "
                    ":org_unit_id, :course_id "
                    "FROM roles r WHERE r.code = :role_code"
                ),
                {
                    "uid": user_ids[role_code],
                    "role_code": role_code,
                    "scope_kind": scope_kind,
                    "organization_id": org_id,
                    "org_unit_id": ou_id,
                    "course_id": c_id,
                },
            )

    yield _Scenario(
        organization_id=organization_id,
        org_unit_id=org_unit_id,
        course_id=course_id,
        student_id=user_ids["student"],
        teacher_id=user_ids["teacher"],
        hod_id=user_ids["hod"],
        manager_id=user_ids["manager"],
        admin_id=user_ids["admin"],
    )

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
            {"ids": list(user_ids.values())},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM org_units WHERE id = :id"), {"id": org_unit_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": list(user_ids.values())},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"),
            {"id": organization_id},
        )


@pytest_asyncio.fixture
async def app(engine: AsyncEngine) -> AsyncIterator[FastAPI]:
    sm = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with sm() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(admin_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _open_session(engine: AsyncEngine, user_id: uuid.UUID) -> tuple[uuid.UUID, str]:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return session_id, create_access_token(user_id=user_id, session_id=session_id)


async def _close_session(engine: AsyncEngine, session_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = :id"),
            {"id": session_id},
        )


async def test_student_perms_resolve_to_4_codes(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        perms = await load_user_permissions(session, scenario.student_id)

    expected = {"course.read", "quiz.take", "interview.take", "progress.read.self"}
    missing = expected - perms
    assert missing == set(), (
        f"student must resolve to T1.3 catalog student permissions; missing={missing}"
    )


async def test_teacher_course_scope_resolves_for_own_course(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        perms = await load_course_permissions(session, scenario.teacher_id, scenario.course_id)

    assert "course.update" in perms, (
        f"teacher (scope=course) must resolve course.update on own course; got {sorted(perms)}"
    )
    assert "lesson.manage" in perms
    assert "quiz.manage" in perms


async def test_hod_org_unit_scope_resolves_descendants(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    child_unit_id = uuid.uuid4()
    child_course_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, parent_unit_id, "
                "unit_type, name, code) "
                "VALUES (:id, :org, :parent, 'program', :name, :code)"
            ),
            {
                "id": child_unit_id,
                "org": scenario.organization_id,
                "parent": scenario.org_unit_id,
                "name": "T1.13 AC Child Program",
                "code": f"T113-AC-CHILD-{child_unit_id.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, org_unit_id, "
                "owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :unit, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": child_course_id,
                "org": scenario.organization_id,
                "unit": child_unit_id,
                "owner": scenario.admin_id,
                "slug": f"t113-ac-child-course-{child_course_id.hex[:8]}",
                "title": "T1.13 AC Child Course",
            },
        )

    try:
        async with session_factory() as session:
            perms = await load_course_permissions(session, scenario.hod_id, child_course_id)

        assert "course.read.draft" in perms, (
            "HOD at parent unit must resolve permission on course in DESCENDANT unit "
            "(recursive ancestor walk via org_unit_tree CTE)"
        )
        assert "course.assign_teacher" in perms
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM courses WHERE id = :id"),
                {"id": child_course_id},
            )
            await conn.execute(
                text("DELETE FROM org_units WHERE id = :id"),
                {"id": child_unit_id},
            )


async def test_manager_organization_scope_resolves(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    extra_course_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": extra_course_id,
                "org": scenario.organization_id,
                "owner": scenario.admin_id,
                "slug": f"t113-mgr-course-{extra_course_id.hex[:8]}",
                "title": "T1.13 Manager Org-Scope Course",
            },
        )

    try:
        async with session_factory() as session:
            perms = await load_course_permissions(session, scenario.manager_id, extra_course_id)

        assert "course.publish" in perms, (
            "manager at scope=organization must resolve permissions on every course in same org"
        )
        assert "course.delete" in perms
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM courses WHERE id = :id"),
                {"id": extra_course_id},
            )


async def test_admin_global_scope_resolves_anywhere(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    other_org_id = uuid.uuid4()
    other_course_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": other_org_id,
                "slug": f"t113-other-{other_org_id.hex[:8]}",
                "name": "T1.13 Other Org",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": other_course_id,
                "org": other_org_id,
                "owner": scenario.admin_id,
                "slug": f"t113-other-course-{other_course_id.hex[:8]}",
                "title": "T1.13 Other Org Course",
            },
        )

    try:
        async with session_factory() as session:
            perms = await load_course_permissions(session, scenario.admin_id, other_course_id)

        assert "system.administer" in perms, (
            "admin at scope=global must resolve catalog permissions on courses in any org"
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM courses WHERE id = :id"),
                {"id": other_course_id},
            )
            await conn.execute(
                text("DELETE FROM organizations WHERE id = :id"),
                {"id": other_org_id},
            )


async def test_role_assignment_via_admin_router_grants_perms(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    target_user_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": target_user_id, "em": f"t113-target-{target_user_id.hex[:8]}@test.local"},
        )

    admin_session_id, admin_token = await _open_session(engine, scenario.admin_id)

    try:
        async with session_factory() as session:
            before = await load_course_permissions(session, target_user_id, scenario.course_id)
        assert before == set(), f"unassigned user must have no perms; got {sorted(before)}"

        response = await client.post(
            f"/api/v1/admin/users/{target_user_id}/assignments",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "role_code": "teacher",
                "scope_kind": "course",
                "organization_id": str(scenario.organization_id),
                "course_id": str(scenario.course_id),
            },
        )
        assert response.status_code == 201, response.text

        async with session_factory() as session:
            after = await load_course_permissions(session, target_user_id, scenario.course_id)
        assert "course.update" in after, (
            f"after admin POST grants teacher role, course.update must resolve; got {sorted(after)}"
        )
        assert "lesson.manage" in after
    finally:
        await _close_session(engine, admin_session_id)
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE user_id = :uid"),
                {"uid": target_user_id},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = :id"),
                {"id": target_user_id},
            )


async def test_role_revocation_removes_perms(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: _Scenario,
) -> None:
    target_user_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": target_user_id, "em": f"t113-revoke-{target_user_id.hex[:8]}@test.local"},
        )

    admin_session_id, admin_token = await _open_session(engine, scenario.admin_id)

    try:
        grant = await client.post(
            f"/api/v1/admin/users/{target_user_id}/assignments",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "role_code": "teacher",
                "scope_kind": "course",
                "organization_id": str(scenario.organization_id),
                "course_id": str(scenario.course_id),
            },
        )
        assert grant.status_code == 201, grant.text
        assignment_id = uuid.UUID(grant.json()["id"])

        async with session_factory() as session:
            after_grant = await load_course_permissions(session, target_user_id, scenario.course_id)
        assert "course.update" in after_grant

        revoke = await client.delete(
            f"/api/v1/admin/users/{target_user_id}/assignments/{assignment_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert revoke.status_code == 204, revoke.text

        async with session_factory() as session:
            after_revoke = await load_course_permissions(
                session, target_user_id, scenario.course_id
            )
        assert after_revoke == set(), (
            f"after revocation perms must disappear; got {sorted(after_revoke)}"
        )
    finally:
        await _close_session(engine, admin_session_id)
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE user_id = :uid"),
                {"uid": target_user_id},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = :id"),
                {"id": target_user_id},
            )


async def test_active_window_filter_excludes_expired(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    expired_user_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {
                "id": expired_user_id,
                "em": f"t113-expired-{expired_user_id.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, active_from, active_until) "
                "SELECT :uid, id, 'global', :af, :au FROM roles WHERE code = 'admin'"
            ),
            {
                "uid": expired_user_id,
                "af": datetime.now(UTC) - timedelta(days=10),
                "au": datetime.now(UTC) - timedelta(days=1),
            },
        )

    try:
        async with session_factory() as session:
            now_perms = await load_user_permissions(session, expired_user_id)
        assert now_perms == set(), (
            f"expired admin assignment must NOT resolve at now; got {sorted(now_perms)}"
        )

        async with session_factory() as session:
            past_perms = await load_user_permissions(
                session,
                expired_user_id,
                at=datetime.now(UTC) - timedelta(days=5),
            )
        assert "system.administer" in past_perms, (
            "time-travel read inside the active window must still resolve"
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE user_id = :uid"),
                {"uid": expired_user_id},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = :id"),
                {"id": expired_user_id},
            )


async def test_admin_router_catalog_reads(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: _Scenario,
) -> None:
    admin_session_id, admin_token = await _open_session(engine, scenario.admin_id)
    try:
        permissions = await client.get(
            "/api/v1/admin/permissions",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert permissions.status_code == 200, permissions.text
        codes = {row["code"] for row in permissions.json()}
        assert "course.read" in codes
        assert "system.administer" in codes

        roles = await client.get(
            "/api/v1/admin/roles",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert roles.status_code == 200, roles.text
        role_codes = {row["role"]["code"] for row in roles.json()}
        assert {"student", "teacher", "hod", "manager", "admin"} <= role_codes

        assignments = await client.get(
            f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert assignments.status_code == 200, assignments.text
        assert any(a["scope_kind"] == "course" for a in assignments.json())
    finally:
        await _close_session(engine, admin_session_id)


async def test_admin_router_grant_lifecycle(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: _Scenario,
) -> None:
    admin_session_id, admin_token = await _open_session(engine, scenario.admin_id)
    target_user_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": target_user_id, "em": f"t113-grant-{target_user_id.hex[:8]}@test.local"},
        )
        permission_id = (
            await conn.execute(
                text("SELECT id FROM permissions WHERE code = 'course.read' LIMIT 1"),
            )
        ).scalar_one()

    try:
        listing_before = await client.get(
            f"/api/v1/admin/users/{target_user_id}/grants",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listing_before.status_code == 200, listing_before.text
        assert listing_before.json() == []

        created = await client.post(
            f"/api/v1/admin/users/{target_user_id}/grants",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "permission_id": str(permission_id),
                "scope_kind": "global",
            },
        )
        assert created.status_code == 201, created.text
        grant_id = uuid.UUID(created.json()["id"])

        listing_after = await client.get(
            f"/api/v1/admin/users/{target_user_id}/grants",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listing_after.status_code == 200, listing_after.text
        assert any(uuid.UUID(g["id"]) == grant_id for g in listing_after.json())

        revoked = await client.delete(
            f"/api/v1/admin/users/{target_user_id}/grants/{grant_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert revoked.status_code == 204, revoked.text
    finally:
        await _close_session(engine, admin_session_id)
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_permission_grants WHERE user_id = :uid"),
                {"uid": target_user_id},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = :id"),
                {"id": target_user_id},
            )


async def test_admin_router_membership_endpoints(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: _Scenario,
) -> None:
    admin_session_id, admin_token = await _open_session(engine, scenario.admin_id)
    member_user_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": member_user_id, "em": f"t113-member-{member_user_id.hex[:8]}@test.local"},
        )

    try:
        empty = await client.get(
            f"/api/v1/admin/organizations/{scenario.organization_id}/memberships",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert empty.status_code == 200, empty.text
        assert isinstance(empty.json(), list)

        added = await client.post(
            f"/api/v1/admin/organizations/{scenario.organization_id}/memberships",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "user_id": str(member_user_id),
                "org_unit_id": str(scenario.org_unit_id),
                "status": "active",
                "student_code": "T113-MEM-001",
            },
        )
        assert added.status_code == 201, added.text
        assert added.json()["user_id"] == str(member_user_id)

        listing = await client.get(
            f"/api/v1/admin/organizations/{scenario.organization_id}/memberships",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listing.status_code == 200, listing.text
        assert any(m["user_id"] == str(member_user_id) for m in listing.json())
    finally:
        await _close_session(engine, admin_session_id)
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM organization_memberships WHERE user_id = :uid"),
                {"uid": member_user_id},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = :id"),
                {"id": member_user_id},
            )


def test_no_dev_bypass_in_repo() -> None:
    backend_new = Path(__file__).resolve().parents[2]
    abridgeai_root = backend_new / "abridgeai"
    assert abridgeai_root.is_dir(), abridgeai_root

    needles = ("_DEV_PERMISSION_BYPASS", "PERMISSION_BYPASS =", "if _DEV_")
    offending: list[str] = []
    for path in abridgeai_root.rglob("*.py"):
        body = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in body:
                offending.append(f"{path}: contains {needle!r}")

    assert offending == [], (
        "FIX-CRIT-1 violated: production code under abridgeai/ contains the "
        "legacy permission-bypass flag pattern:\n  " + "\n  ".join(offending)
    )
