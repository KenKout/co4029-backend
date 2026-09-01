from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from conftest import SeededUsers
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.security import CurrentUser, create_access_token
from abridgeai.features.access_control.policies import (
    can_manage_course,
    require_any_permission,
    require_course_permission,
    require_permission,
)

_PROBE_PERM_OWN = "_test.policy_owned_by_a_only"
_PROBE_PERM_SHARED = "_test.policy_shared_by_b"


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


@dataclass
class _Scenario:
    sibling_unit_id: uuid.UUID
    sibling_unit_course_id: uuid.UUID
    dept_a_course_id: uuid.UUID
    ownerless_user_id: uuid.UUID
    owned_course_id: uuid.UUID
    perm_only_a_user_id: uuid.UUID
    auth_session_ids: dict[uuid.UUID, uuid.UUID]


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[_Scenario]:
    sibling_unit_id = uuid.uuid4()
    sibling_unit_course_id = uuid.uuid4()
    dept_a_course_id = uuid.uuid4()
    ownerless_user_id = uuid.uuid4()
    owned_course_id = uuid.uuid4()
    perm_only_a_user_id = uuid.uuid4()
    perm_a_id = uuid.uuid4()
    perm_b_id = uuid.uuid4()

    auth_session_ids: dict[uuid.UUID, uuid.UUID] = {}

    async with engine.begin() as conn:
        for code, pid in ((_PROBE_PERM_OWN, perm_a_id), (_PROBE_PERM_SHARED, perm_b_id)):
            await conn.execute(
                text("INSERT INTO permissions (id, code, name) VALUES (:id, :code, :name)"),
                {"id": pid, "code": code, "name": code},
            )

        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                # 0094_flat_faculties: every LIVE unit must be a top-level
                # faculty. Still a SIBLING of the dean's own faculty, which
                # is what this scenario needs — the dean holds no faculty
                # assignment here, so the org_unit branch must not resolve.
                "VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": sibling_unit_id,
                "org": seeded_users.organization_id,
                "name": "Policy Sibling Dept",
                "code": f"PSIB-{sibling_unit_id.hex[:6]}",
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
                "slug": f"policy-sib-{sibling_unit_course_id.hex[:8]}",
                "title": "Policy Sibling Course",
            },
        )

        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, faculty_id, "
                "owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :unit, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": dept_a_course_id,
                "org": seeded_users.organization_id,
                "unit": seeded_users.org_unit_id,
                "owner": seeded_users.admin_id,
                "slug": f"policy-dept-a-{dept_a_course_id.hex[:8]}",
                "title": "Policy DeptA Course",
            },
        )

        for uid, email in (
            (ownerless_user_id, f"policy-owner-{ownerless_user_id.hex[:8]}@test.local"),
            (perm_only_a_user_id, f"policy-perma-{perm_only_a_user_id.hex[:8]}@test.local"),
        ):
            await conn.execute(
                text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
                {"id": uid, "em": email},
            )
            sid = uuid.uuid4()
            auth_session_ids[uid] = sid
            await conn.execute(
                text(
                    "INSERT INTO auth_sessions "
                    "(id, user_id, expires_at, refresh_token_hash) "
                    "VALUES (:id, :u, NOW() + INTERVAL '1 hour', :rt)"
                ),
                {"id": sid, "u": uid, "rt": f"rth-{sid.hex}"},
            )

        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": owned_course_id,
                "org": seeded_users.organization_id,
                "owner": ownerless_user_id,
                "slug": f"policy-owned-{owned_course_id.hex[:8]}",
                "title": "Policy Owned Course",
            },
        )

        await conn.execute(
            text(
                "INSERT INTO user_permission_grants "
                "(user_id, permission_id, scope_kind) VALUES (:u, :p, 'global')"
            ),
            {"u": perm_only_a_user_id, "p": perm_a_id},
        )

        for fixture_uid in (
            seeded_users.student_id,
            seeded_users.teacher_id,
            seeded_users.hod_id,
            seeded_users.manager_id,
            seeded_users.admin_id,
        ):
            sid = uuid.uuid4()
            auth_session_ids[fixture_uid] = sid
            await conn.execute(
                text(
                    "INSERT INTO auth_sessions "
                    "(id, user_id, expires_at, refresh_token_hash) "
                    "VALUES (:id, :u, NOW() + INTERVAL '1 hour', :rt)"
                ),
                {"id": sid, "u": fixture_uid, "rt": f"rth-{sid.hex}"},
            )

    yield _Scenario(
        sibling_unit_id=sibling_unit_id,
        sibling_unit_course_id=sibling_unit_course_id,
        dept_a_course_id=dept_a_course_id,
        ownerless_user_id=ownerless_user_id,
        owned_course_id=owned_course_id,
        perm_only_a_user_id=perm_only_a_user_id,
        auth_session_ids=auth_session_ids,
    )

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = ANY(:ids)"),
            {"ids": list(auth_session_ids.values())},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [sibling_unit_course_id, dept_a_course_id, owned_course_id]},
        )
        await conn.execute(
            text("DELETE FROM org_units WHERE id = :id"),
            {"id": sibling_unit_id},
        )
        await conn.execute(
            text("DELETE FROM user_permission_grants WHERE user_id = :u"),
            {"u": perm_only_a_user_id},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [ownerless_user_id, perm_only_a_user_id]},
        )
        await conn.execute(
            text("DELETE FROM permissions WHERE id = ANY(:ids)"),
            {"ids": [perm_a_id, perm_b_id]},
        )


def _token_for(user_id: uuid.UUID, session_id: uuid.UUID) -> str:
    return create_access_token(user_id=user_id, session_id=session_id)


def _build_app() -> FastAPI:
    app = FastAPI()

    global_dep = Depends(require_permission("course.update"))
    course_dep = Depends(require_course_permission("course_id", "course.update"))
    any_dep = Depends(require_any_permission(_PROBE_PERM_OWN, _PROBE_PERM_SHARED))

    @app.get("/protected/global")
    async def global_route(user: CurrentUser = global_dep) -> dict[str, str]:
        return {"user_id": str(user.user_id)}

    @app.get("/protected/courses/{course_id}")
    async def course_route(user: CurrentUser = course_dep) -> dict[str, str]:
        return {"user_id": str(user.user_id)}

    @app.get("/protected/any")
    async def any_route(user: CurrentUser = any_dep) -> dict[str, str]:
        return {"user_id": str(user.user_id)}

    return app


@pytest_asyncio.fixture
async def http_client() -> AsyncIterator[AsyncClient]:
    app = _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_require_permission_grants_admin(
    http_client: AsyncClient,
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    token = _token_for(seeded_users.admin_id, scenario.auth_session_ids[seeded_users.admin_id])
    response = await http_client.get(
        "/protected/global",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == str(seeded_users.admin_id)


@pytest.mark.asyncio
async def test_require_permission_blocks_student(
    http_client: AsyncClient,
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    token = _token_for(seeded_users.student_id, scenario.auth_session_ids[seeded_users.student_id])
    response = await http_client.get(
        "/protected/global",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["detail"]["error"] == "permission_denied"
    assert body["detail"]["required"] == ["course.update"]
    assert body["detail"]["scope"] == "global"


@pytest.mark.asyncio
async def test_hod_org_unit_propagation(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    async with session_factory() as session:
        in_dept_a = await can_manage_course(
            session,
            seeded_users.hod_id,
            scenario.dept_a_course_id,
            manage_perm="course.read.draft",
        )
        assert in_dept_a is True, (
            "HOD scoped to DeptA must manage Course-X in DeptA via scope_kind='org_unit' resolution"
        )

        in_dept_b = await can_manage_course(
            session,
            seeded_users.hod_id,
            scenario.sibling_unit_course_id,
            manage_perm="course.read.draft",
        )
        assert in_dept_b is False, (
            "HOD scoped to DeptA must NOT manage Course-Y in DeptB (sibling unit)"
        )


@pytest.mark.asyncio
async def test_owner_bypasses_permission_requirement(
    http_client: AsyncClient,
    scenario: _Scenario,
) -> None:
    token = _token_for(
        scenario.ownerless_user_id,
        scenario.auth_session_ids[scenario.ownerless_user_id],
    )
    response = await http_client.get(
        f"/protected/courses/{scenario.owned_course_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == str(scenario.ownerless_user_id)


@pytest.mark.asyncio
async def test_owner_bypass_does_not_apply_to_non_owned_course(
    http_client: AsyncClient,
    seeded_users: SeededUsers,
    scenario: _Scenario,
) -> None:
    token = _token_for(seeded_users.student_id, scenario.auth_session_ids[seeded_users.student_id])
    response = await http_client.get(
        f"/protected/courses/{seeded_users.course_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["detail"]["error"] == "permission_denied"
    assert body["detail"]["scope"] == "course"


@pytest.mark.asyncio
async def test_require_any_permission_passes_on_first_match(
    http_client: AsyncClient,
    scenario: _Scenario,
) -> None:
    token = _token_for(
        scenario.perm_only_a_user_id,
        scenario.auth_session_ids[scenario.perm_only_a_user_id],
    )
    response = await http_client.get(
        "/protected/any",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text


def test_no_devmode_permission_flag() -> None:
    backend_new = Path(__file__).resolve().parents[2]
    abridgeai = backend_new / "abridgeai"
    assert abridgeai.is_dir(), abridgeai

    legacy_flag_pattern = re.compile(r"_DEV_PERMISSION_BYPASS|PERMISSION_BYPASS\s*=|if\s+_DEV_")
    offending: list[str] = []
    for path in abridgeai.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if legacy_flag_pattern.search(line):
                offending.append(f"{path}:{lineno}:{line.strip()}")

    assert offending == [], (
        "FIX-CRIT-1 violated: production code under abridgeai/ contains the "
        "legacy permission-bypass flag pattern:\n" + "\n".join(offending)
    )
