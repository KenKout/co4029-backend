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
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'department', :name, :code)"
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
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments "
                    "(id, user_id, role_id, scope_kind, organization_id, "
                    "org_unit_id, course_id) "
                    "SELECT :assignment_id, :user_id, r.id, :scope_kind, "
                    ":organization_id, :org_unit_id, :course_id "
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
async def test_list_permissions_admin_200_manager_403(
    http_client: AsyncClient, scenario: _AdminScenario
) -> None:
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])

    r_admin = await http_client.get("/api/v1/admin/permissions", headers=_hdr(admin_token))
    assert r_admin.status_code == 200, r_admin.text
    body = r_admin.json()
    assert isinstance(body, list)
    assert any(p["code"] == "user.role_assign" for p in body)

    r_mgr = await http_client.get("/api/v1/admin/permissions", headers=_hdr(manager_token))
    assert r_mgr.status_code == 403, r_mgr.text
    assert r_mgr.json()["detail"]["error"] == "permission_denied"


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
    manager_token = _token_for(scenario.manager_id, scenario.auth_sessions[scenario.manager_id])
    admin_token = _token_for(scenario.admin_id, scenario.auth_sessions[scenario.admin_id])

    payload = {
        "role_code": "hod",
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
