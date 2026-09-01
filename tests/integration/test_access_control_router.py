from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token
from abridgeai.features.access_control import models as _ac_models  # noqa: F401
from abridgeai.features.access_control.routers.admin import router as admin_router
from abridgeai.features.identity import models as _identity_models  # noqa: F401


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


@dataclass(frozen=True)
class _AdminScenario:
    organization_id: uuid.UUID
    org_unit_id: uuid.UUID
    course_id: uuid.UUID
    student_id: uuid.UUID
    teacher_id: uuid.UUID
    hod_id: uuid.UUID
    manager_id: uuid.UUID
    admin_id: uuid.UUID
    auth_sessions: dict[uuid.UUID, uuid.UUID]


@pytest_asyncio.fixture(scope="module")
async def scenario(engine: AsyncEngine) -> AsyncIterator[_AdminScenario]:
    """Self-contained module-scoped scenario.

    Avoids depending on the conftest ``seeded_users`` (session-scoped)
    fixture so this test file is order-independent against the destructive
    ``test_catalog_seed_migration`` migration round-trip elsewhere in the
    suite.
    """
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
    auth_sessions: dict[uuid.UUID, uuid.UUID] = {}

    role_assignments = [
        ("student", "organization", organization_id, None, None),
        ("teacher", "course", organization_id, None, course_id),
        ("hod", "org_unit", organization_id, org_unit_id, None),
        ("manager", "organization", organization_id, None, None),
        ("admin", "global", None, None, None),
    ]

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": organization_id,
                "slug": f"adminsuite-{organization_id.hex[:8]}",
                "name": "Admin Router Suite Org",
            },
        )
        await conn.execute(
            text(
                # Migration 0094 flattened org units into FACULTY roots
                # (ck_org_units_live_faculty_root forbids a live non-faculty
                # unit). This scenario's unit is the Dean's faculty.
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": org_unit_id,
                "org": organization_id,
                "name": "Admin Suite Dept",
                "code": f"ASD-{org_unit_id.hex[:6]}",
            },
        )
        for role, uid in user_ids.items():
            await conn.execute(
                text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
                {"id": uid, "em": f"adminsuite-{role}-{uid.hex[:6]}@test.local"},
            )
        # 0094's faculty-scoped assignment guard requires the TARGET to be an
        # active org member (active_org_member_user_ids). Every user in the
        # scenario is in the org; the fixture inserts memberships directly.
        for role, uid in user_ids.items():
            await conn.execute(
                text(
                    "INSERT INTO organization_memberships "
                    "(id, user_id, organization_id, status, joined_at) "
                    "VALUES (:id, :u, :org, 'active', NOW())"
                ),
                {
                    "id": uuid.uuid4(),
                    "u": uid,
                    "org": organization_id,
                },
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": organization_id,
                "owner": user_ids["admin"],
                "slug": f"adminsuite-course-{course_id.hex[:8]}",
                "title": "Admin Suite Course",
            },
        )

        for role_code, scope_kind, org_id, ou_id, c_id in role_assignments:
            assignment_id = uuid.uuid4()
            # Migration 0093 made course-scoped assignments require a title
            # flag (is_instructor/is_assistant, checked by
            # ck_user_role_assignments_course_title). The fixture's teacher
            # row is course-scoped, so it must carry one or the seed itself
            # violates the constraint before any test runs.
            is_instructor = scope_kind == "course"
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments "
                    "(id, user_id, role_id, scope_kind, organization_id, "
                    "org_unit_id, course_id, is_instructor, is_assistant) "
                    "SELECT :assignment_id, :user_id, r.id, :scope_kind, "
                    ":organization_id, :org_unit_id, :course_id, "
                    ":is_instructor, FALSE "
                    "FROM roles r WHERE r.code = :role_code"
                ),
                {
                    "assignment_id": assignment_id,
                    "user_id": user_ids[role_code],
                    "role_code": role_code,
                    "scope_kind": scope_kind,
                    "organization_id": org_id,
                    "org_unit_id": ou_id,
                    "course_id": c_id,
                    "is_instructor": is_instructor,
                },
            )
            # Migration 0094 flattened units into FACULTIES and made faculty-
            # scoped staff roles (hod in this scenario) require the matching
            # user_faculty_assignments affiliation: user_has_role_scope() only
            # sees a faculty-scoped Dean when the affiliation row exists. The
            # service normally creates it in the same transaction; the fixture
            # seeds rows directly, so it must seed the affiliation too.
            if scope_kind == "org_unit" and ou_id is not None:
                await conn.execute(
                    text(
                        "INSERT INTO user_faculty_assignments "
                        "(id, user_id, organization_id, faculty_id, status, "
                        "active_from, created_at, updated_at) "
                        "VALUES (:id, :user_id, :org, :faculty_id, 'active', "
                        "NOW(), NOW(), NOW())"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "user_id": user_ids[role_code],
                        "org": org_id,
                        "faculty_id": ou_id,
                    },
                )

        for uid in user_ids.values():
            sid = uuid.uuid4()
            auth_sessions[uid] = sid
            await conn.execute(
                text(
                    "INSERT INTO auth_sessions "
                    "(id, user_id, expires_at, refresh_token_hash) "
                    "VALUES (:id, :u, NOW() + INTERVAL '1 hour', :rt)"
                ),
                {"id": sid, "u": uid, "rt": f"rth-adminsuite-{sid.hex}"},
            )

    yield _AdminScenario(
        organization_id=organization_id,
        org_unit_id=org_unit_id,
        course_id=course_id,
        student_id=user_ids["student"],
        teacher_id=user_ids["teacher"],
        hod_id=user_ids["hod"],
        manager_id=user_ids["manager"],
        admin_id=user_ids["admin"],
        auth_sessions=auth_sessions,
    )

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = ANY(:ids)"),
            {"ids": list(auth_sessions.values())},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
            {"ids": list(user_ids.values())},
        )
        await conn.execute(
            text("DELETE FROM user_faculty_assignments WHERE user_id = ANY(:ids)"),
            {"ids": list(user_ids.values())},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = :id"),
            {"id": course_id},
        )
        await conn.execute(
            text("DELETE FROM org_units WHERE id = :id"),
            {"id": org_unit_id},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": list(user_ids.values())},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"),
            {"id": organization_id},
        )


def _token_for(user_id: uuid.UUID, session_id: uuid.UUID) -> str:
    return create_access_token(user_id=user_id, session_id=session_id)


@pytest_asyncio.fixture(scope="module")
async def admin_app(engine: AsyncEngine) -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router, prefix="/api/v1")

    sm = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _override_db() -> AsyncIterator[AsyncSession]:
        async with sm() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db
    return app


@pytest_asyncio.fixture
async def http_client(admin_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_catalog_readable_by_role_assign_holders_forbidden_for_others(
    http_client: AsyncClient, scenario: _AdminScenario
) -> None:
    """The role/permission catalog must follow the ASSIGNMENT gate, not out-rank it.

    The SPA loads ``/admin/roles`` to build the role filter AND to resolve role
    ids when posting an assignment. Gating the catalog tighter than the write
    (admin-only) left a Dean/manager holding ``user.role_assign`` able to POST
    ``/admin/users/{id}/assignments`` but 403'd on the catalog their own
    management-users page needs — the Make/Remove-manager flow threw "Role
    catalog not loaded". Admin still gets the full catalog; a user with NO
    role-assign permission (student) is still refused.
    """
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    student_token = _token_for(scenario.student_id, scenario.auth_sessions[scenario.student_id])

    r_admin = await http_client.get("/api/v1/admin/permissions", headers=_hdr(admin_token))
    assert r_admin.status_code == 200, r_admin.text
    body = r_admin.json()
    assert isinstance(body, list)
    assert any(p["code"] == "user.role_assign" for p in body)

    # manager holds user.role_assign (and user.role_assign.hod is the Dean's) —
    # the same perms the assignment-write endpoint requires, so the catalog
    # read must be open to them or the assign flows cannot render.
    r_mgr = await http_client.get("/api/v1/admin/roles", headers=_hdr(manager_token))
    assert r_mgr.status_code == 200, r_mgr.text
    assert isinstance(r_mgr.json(), list)

    r_student = await http_client.get("/api/v1/admin/roles", headers=_hdr(student_token))
    assert r_student.status_code == 403, r_student.text
    assert r_student.json()["detail"]["error"] == "permission_denied"


@pytest.mark.asyncio
async def test_unauthenticated_returns_401(http_client: AsyncClient) -> None:
    r = await http_client.get("/api/v1/admin/permissions")
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_student_cannot_create_role_assignment(
    http_client: AsyncClient, scenario: _AdminScenario
) -> None:
    student_token = _token_for(scenario.student_id, scenario.auth_sessions[scenario.student_id])
    r = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(student_token),
        json={
            "role_code": "teacher",
            "scope_kind": "course",
            "organization_id": str(scenario.organization_id),
            "course_id": str(scenario.course_id),
        },
    )
    assert r.status_code == 403, r.text


@pytest.mark.asyncio
async def test_create_role_assignment_validates_scope_shape(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])

    bad = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(admin_token),
        json={"role_code": "teacher", "scope_kind": "org_unit"},
    )
    assert bad.status_code == 422, bad.text
    assert bad.json()["detail"]["error"] == "scope_invalid"

    good = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(admin_token),
        json={
            "role_code": "teacher",
            "scope_kind": "course",
            "organization_id": str(scenario.organization_id),
            "course_id": str(scenario.course_id),
        },
    )
    assert good.status_code == 201, good.text
    body = good.json()
    assert body["role_id"]
    assert body["scope_kind"] == "course"
    assignment_id = uuid.UUID(body["id"])

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE id = :id"),
            {"id": assignment_id},
        )


@pytest.mark.asyncio
async def test_manager_cannot_promote_to_hod_admin_can(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """manager may not grant ANY hod; a system admin may bootstrap a Master Dean.

    Migration 0094 split the Dean role: a Master Dean is the organization-scoped
    hod (only a system admin may bootstrap one) and appoints Faculty Deans at
    faculty scope. A plain manager holds neither ``user.role_assign.hod`` nor
    ``system.administer``, so both attempts must be 403.
    """
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])

    payload = {
        "role_code": "hod",
        "scope_kind": "organization",
        "organization_id": str(scenario.organization_id),
    }
    r_mgr = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(manager_token),
        json=payload,
    )
    assert r_mgr.status_code == 403, r_mgr.text
    assert r_mgr.json()["detail"]["error"] == "forbidden"

    r_adm = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(admin_token),
        json=payload,
    )
    assert r_adm.status_code == 201, r_adm.text
    assignment_id = uuid.UUID(r_adm.json()["id"])

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE id = :id"),
            {"id": assignment_id},
        )


@pytest.mark.asyncio
async def test_manager_cannot_promote_to_manager_hod_can(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """manager may not grant the manager role (HOD-gated); a Faculty Dean may.

    Migration 0094 made delegated manager assignments faculty-scoped: the Dean
    grants at his OWN faculty (``org_unit``), where he holds both the hod role
    and the matching affiliation, and the target gets the manager role for that
    faculty. A plain manager holds ``user.role_assign`` but not
    ``user.role_assign.hod``, so his attempt is forbidden before scope matters.
    """
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    hod_token = _token_for(scenario.hod_id, scenario.auth_sessions[scenario.hod_id])

    payload = {
        "role_code": "manager",
        "scope_kind": "org_unit",
        "organization_id": str(scenario.organization_id),
        "org_unit_id": str(scenario.org_unit_id),
    }
    r_mgr = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(manager_token),
        json=payload,
    )
    assert r_mgr.status_code == 403, r_mgr.text

    r_hod = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(hod_token),
        json=payload,
    )
    assert r_hod.status_code == 201, r_hod.text
    assignment_id = uuid.UUID(r_hod.json()["id"])

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE id = :id"),
            {"id": assignment_id},
        )
        await conn.execute(
            text("DELETE FROM user_faculty_assignments WHERE user_id = :id"),
            {"id": scenario.teacher_id},
        )


@pytest.mark.asyncio
async def test_manager_cannot_assign_outside_org(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """Org-scope: a manager can only assign in orgs where they hold
    ``user.role_assign``. Creating a second org and assigning there → 403."""
    other_org_id = uuid.uuid4()
    other_teacher_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": other_org_id,
                "slug": f"crossorg-{other_org_id.hex[:8]}",
                "name": "Cross-Org Test",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": other_teacher_id, "em": f"cross-{other_teacher_id.hex[:6]}@test.local"},
        )

    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    r = await http_client.post(
        f"/api/v1/admin/users/{other_teacher_id}/assignments",
        headers=_hdr(manager_token),
        json={
            "role_code": "teacher",
            "scope_kind": "organization",
            "organization_id": str(other_org_id),
        },
    )
    assert r.status_code == 403, r.text

    # Admin can — cross-org assignment is an admin privilege.
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])
    r_adm = await http_client.post(
        f"/api/v1/admin/users/{other_teacher_id}/assignments",
        headers=_hdr(admin_token),
        json={
            "role_code": "teacher",
            "scope_kind": "organization",
            "organization_id": str(other_org_id),
        },
    )
    assert r_adm.status_code == 201, r_adm.text
    assignment_id = uuid.UUID(r_adm.json()["id"])

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE id = :id"),
            {"id": assignment_id},
        )


@pytest.mark.asyncio
async def test_non_admin_cannot_assign_global_scope(
    http_client: AsyncClient, scenario: _AdminScenario
) -> None:
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    r = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(manager_token),
        json={"role_code": "teacher", "scope_kind": "global"},
    )
    assert r.status_code == 403, r.text


@pytest.mark.asyncio
async def test_cannot_assign_role_to_self(
    http_client: AsyncClient, scenario: _AdminScenario
) -> None:
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    r = await http_client.post(
        f"/api/v1/admin/users/{scenario.manager_id}/assignments",
        headers=_hdr(manager_token),
        json={
            "role_code": "teacher",
            "scope_kind": "organization",
            "organization_id": str(scenario.organization_id),
        },
    )
    assert r.status_code == 403, r.text


@pytest.mark.asyncio
async def test_manager_cannot_revoke_manager_role(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """Revoke mirrors the HOD gate: a manager cannot strip a manager role.

    The manager role being revoked lives at FACULTY scope (0094): the Dean who
    granted it (master, org-scope hod in this scenario) created it against his
    faculty. The plain manager's revoke attempt is forbidden; an admin may
    revoke anything.
    """
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])

    r = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(admin_token),
        json={
            "role_code": "manager",
            "scope_kind": "org_unit",
            "organization_id": str(scenario.organization_id),
            "org_unit_id": str(scenario.org_unit_id),
        },
    )
    assert r.status_code == 201, r.text
    assignment_id = uuid.UUID(r.json()["id"])

    r_revoke = await http_client.delete(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments/{assignment_id}",
        headers=_hdr(manager_token),
    )
    assert r_revoke.status_code == 403, r_revoke.text

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE id = :id"),
            {"id": assignment_id},
        )
        await conn.execute(
            text("DELETE FROM user_faculty_assignments WHERE user_id = :id"),
            {"id": scenario.teacher_id},
        )


@pytest.mark.asyncio
async def test_list_assignments_org_scoped_for_manager(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """A manager listing a user's assignments only sees the org they manage."""
    other_org_id = uuid.uuid4()
    other_teacher_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": other_org_id,
                "slug": f"listorg-{other_org_id.hex[:8]}",
                "name": "List Org Test",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": other_teacher_id, "em": f"list-{other_teacher_id.hex[:6]}@test.local"},
        )

    # Admin puts the teacher in BOTH orgs.
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])
    for org_id in (scenario.organization_id, other_org_id):
        r = await http_client.post(
            f"/api/v1/admin/users/{other_teacher_id}/assignments",
            headers=_hdr(admin_token),
            json={
                "role_code": "teacher",
                "scope_kind": "organization",
                "organization_id": str(org_id),
            },
        )
        assert r.status_code == 201, r.text

    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    r = await http_client.get(
        f"/api/v1/admin/users/{other_teacher_id}/assignments",
        headers=_hdr(manager_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Only the assignment in the manager's own org is visible.
    assert len(body) == 1
    assert body[0]["organization_id"] == str(scenario.organization_id)

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = :uid"),
            {"uid": other_teacher_id},
        )


@pytest.mark.asyncio
async def test_hod_cannot_assign_or_revoke_hod(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """Only admin may create/revoke HOD roles — a HOD manages managers."""
    hod_token = _token_for(scenario.hod_id, scenario.auth_sessions[scenario.hod_id])
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])

    payload = {
        "role_code": "hod",
        "scope_kind": "organization",
        "organization_id": str(scenario.organization_id),
    }
    r_hod = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(hod_token),
        json=payload,
    )
    assert r_hod.status_code == 403, r_hod.text

    r_adm = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(admin_token),
        json=payload,
    )
    assert r_adm.status_code == 201, r_adm.text
    assignment_id = uuid.UUID(r_adm.json()["id"])

    r_revoke_hod = await http_client.delete(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments/{assignment_id}",
        headers=_hdr(hod_token),
    )
    assert r_revoke_hod.status_code == 403, r_revoke_hod.text

    r_revoke_adm = await http_client.delete(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments/{assignment_id}",
        headers=_hdr(admin_token),
    )
    assert r_revoke_adm.status_code == 204, r_revoke_adm.text


@pytest.mark.asyncio
async def test_hod_can_revoke_manager_they_granted(
    http_client: AsyncClient, scenario: _AdminScenario, engine: AsyncEngine
) -> None:
    """A Faculty Dean grants manager, then revokes it — the happy path.

    Faculty-scoped (0094): the Dean creates the manager role against his own
    faculty and revokes the same assignment he created.
    """
    hod_token = _token_for(scenario.hod_id, scenario.auth_sessions[scenario.hod_id])

    r = await http_client.post(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments",
        headers=_hdr(hod_token),
        json={
            "role_code": "manager",
            "scope_kind": "org_unit",
            "organization_id": str(scenario.organization_id),
            "org_unit_id": str(scenario.org_unit_id),
        },
    )
    assert r.status_code == 201, r.text
    assignment_id = uuid.UUID(r.json()["id"])

    r_revoke = await http_client.delete(
        f"/api/v1/admin/users/{scenario.teacher_id}/assignments/{assignment_id}",
        headers=_hdr(hod_token),
    )
    assert r_revoke.status_code == 204, r_revoke.text

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_faculty_assignments WHERE user_id = :id"),
            {"id": scenario.teacher_id},
        )


def _walk_dependants(dependant: object, names: list[str]) -> None:
    call = getattr(dependant, "call", None)
    if call is not None:
        names.append(getattr(call, "__name__", repr(call)))
    for sub in getattr(dependant, "dependencies", []):
        _walk_dependants(sub, names)


def test_every_endpoint_has_permission_dependency() -> None:
    """FIX-CRIT-4 perimeter: every route must have a require_* dep."""
    offenders: list[str] = []
    for route in admin_router.routes:
        if not isinstance(route, APIRoute):
            continue
        names: list[str] = []
        for sub in route.dependant.dependencies:
            _walk_dependants(sub, names)
        has_permission_check = any(
            name == "dependency" or name.startswith("require_") for name in names
        )
        if not has_permission_check:
            offenders.append(f"{list(route.methods)} {route.path} -> deps={names}")

    assert offenders == [], (
        "FIX-CRIT-4 violated: the following admin routes have no "
        "require_* permission dependency:\n  " + "\n  ".join(offenders)
    )


def test_admin_router_has_expected_route_count() -> None:
    api_routes = [r for r in admin_router.routes if isinstance(r, APIRoute)]
    assert len(api_routes) >= 8


def test_no_get_current_user_only_endpoints() -> None:
    """FIX-CRIT-4 source-grep: no naked Depends(get_current_user) without require_*."""
    src = Path(admin_router.routes[0].endpoint.__code__.co_filename).read_text()
    offending: list[str] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if "Depends(get_current_user)" in stripped and "require_" not in stripped:
            offending.append(f"{lineno}: {stripped}")
    assert offending == [], (
        "FIX-CRIT-4 violated: admin router contains naked get_current_user dependencies:\n"
        + "\n".join(offending)
    )


def test_router_is_apirouter() -> None:
    assert isinstance(admin_router, APIRouter)
