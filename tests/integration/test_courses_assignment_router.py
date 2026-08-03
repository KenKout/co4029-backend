"""Integration tests for ``features.courses.routers.assignment`` (T3.8).

Covers the HOD/Manager teacher-staffing surface at ``/api/v1/dept``.
The acceptance criteria from plan §4485-4489 + §4491-4505 map to:

* ``test_router_metadata`` -- 6 endpoints registered under ``/dept``.
* ``test_unauthenticated_returns_401`` -- bearer-less requests rejected.
* ``test_student_403_on_assignment`` -- seeded student lacks
  ``course.assign_teacher`` / ``user.role_assign`` / ``system.administer``;
  every endpoint returns 403.
* ``test_hod_scope_bound_can_assign_in_dept`` -- HOD assigns Teacher-Bob
  to a course in their org_unit -> 201, ``user_role_assignments`` row
  created with role=teacher, scope_kind=course, granted_by=HOD.
* ``test_hod_scope_bound_blocks_outside_dept`` -- HOD assigning to a
  course in a sibling org_unit -> 403.
* ``test_manager_can_assign_org_wide`` -- Manager (scope=organization)
  assigns within own org -> 201; outside org -> 403.
* ``test_admin_can_assign_globally`` -- Admin -> 201 across org boundary.
* ``test_remove_sets_active_until`` -- DELETE flips ``active_until`` to
  NOW (within 5s); preserves audit trail.
* ``test_list_courses_filters_by_caller_scope`` -- HOD sees only dept
  courses; admin sees all; manager sees org courses.
* ``test_get_teachers_for_course_returns_active_only`` -- teachers with
  ``active_until`` in the past are excluded.
* ``test_get_org_unit_courses_hod_blocked_outside`` -- HOD passing a
  sibling org_unit_id -> 403.
* ``test_get_roster_returns_enrolled_students`` -- raw-SQL roster shape.
* ``test_no_bare_get_current_user`` -- source-level FIX-CRIT-4 guard.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import assignment_router

for _stub_name in ("interview_configs",):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
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
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(assignment_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return session_id


@pytest_asyncio.fixture
async def hod_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.hod_id)
    yield create_access_token(user_id=seeded_users.hod_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def manager_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.manager_id)
    yield create_access_token(user_id=seeded_users.manager_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Two-org / two-dept staffing layout.

    * ``course_a`` -- in HOD's ``org_unit`` (seeded test_org_unit).
    * ``org_unit_b`` -- sibling dept in same organization.
    * ``course_b`` -- in ``org_unit_b``; HOD has NO scope here.
    * ``other_org`` + ``course_other_org`` -- separate organization.
    * ``bob_id`` -- the teacher being staffed.
    * ``stale_teacher_id`` -- staffed on course_a but with
      ``active_until`` in the past (audit-trail / list filter test).
    * ``enrolled_student_id`` -- enrolled in course_a (roster test).
    """
    suffix = uuid.uuid4().hex[:8]
    course_a = uuid.uuid4()
    course_b = uuid.uuid4()
    course_other_org = uuid.uuid4()
    org_unit_b = uuid.uuid4()
    other_org = uuid.uuid4()
    other_owner = uuid.uuid4()
    bob_id = uuid.uuid4()
    stale_teacher_id = uuid.uuid4()
    enrolled_student_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'department', :name, :code)"
            ),
            {
                "id": org_unit_b,
                "org": seeded_users.organization_id,
                "name": f"Other Dept {suffix}",
                "code": f"OTHER-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": other_org,
                "slug": f"other-org-{suffix}",
                "name": f"Other Org {suffix}",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": other_owner, "email": f"other-owner-{suffix}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) "
                "VALUES (:id, :email, 'active'), (:bid, :bemail, 'active'), "
                "(:sid, :semail, 'active'), (:eid, :eemail, 'active')"
            ),
            {
                "id": uuid.uuid4(),
                "email": f"placeholder-{suffix}@abridgeai.local",
                "bid": bob_id,
                "bemail": f"bob-{suffix}@abridgeai.local",
                "sid": stale_teacher_id,
                "semail": f"stale-{suffix}@abridgeai.local",
                "eid": enrolled_student_id,
                "eemail": f"enrolled-{suffix}@abridgeai.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_profiles (user_id, given_name, family_name, display_name) "
                "VALUES (:id, 'Bob', 'Tester', 'Bob Tester'), "
                "(:sid, 'Stale', 'Teacher', 'Stale Teacher'), "
                "(:eid, 'Enrolled', 'Student', 'Enrolled Student')"
            ),
            {"id": bob_id, "sid": stale_teacher_id, "eid": enrolled_student_id},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, org_unit_id, owner_user_id, "
                "slug, title, status) VALUES "
                "(:a, :org_a, :ou_a, :owner_a, :slug_a, 'Course A', 'draft'), "
                "(:b, :org_b, :ou_b, :owner_b, :slug_b, 'Course B', 'draft'), "
                "(:c, :org_c, NULL, :owner_c, :slug_c, 'Other Course', 'draft')"
            ),
            {
                "a": course_a,
                "org_a": seeded_users.organization_id,
                "ou_a": seeded_users.org_unit_id,
                "owner_a": seeded_users.admin_id,
                "slug_a": f"course-a-{suffix}",
                "b": course_b,
                "org_b": seeded_users.organization_id,
                "ou_b": org_unit_b,
                "owner_b": seeded_users.manager_id,
                "slug_b": f"course-b-{suffix}",
                "c": course_other_org,
                "org_c": other_org,
                "owner_c": other_owner,
                "slug_c": f"other-course-{suffix}",
            },
        )
        teacher_role_id = (
            await conn.execute(text("SELECT id FROM roles WHERE code = 'teacher'"))
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id, course_id, "
                "active_from, active_until, granted_by) "
                "VALUES (:uid, :rid, 'course', :org, :cid, "
                "NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day', :gb)"
            ),
            {
                "uid": stale_teacher_id,
                "rid": teacher_role_id,
                "org": seeded_users.organization_id,
                "cid": course_a,
                "gb": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments "
                "(course_id, student_id, status, source) "
                "VALUES (:cid, :sid, 'active', 'self_enroll')"
            ),
            {"cid": course_a, "sid": enrolled_student_id},
        )

    data: dict[str, uuid.UUID] = {
        "course_a": course_a,
        "course_b": course_b,
        "course_other_org": course_other_org,
        "org_unit_a": seeded_users.org_unit_id,
        "org_unit_b": org_unit_b,
        "other_org": other_org,
        "other_owner": other_owner,
        "bob_id": bob_id,
        "stale_teacher_id": stale_teacher_id,
        "enrolled_student_id": enrolled_student_id,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :cid"),
            {"cid": course_a},
        )
        await conn.execute(
            text(
                "DELETE FROM user_role_assignments "
                "WHERE course_id = ANY(:cids) OR user_id = ANY(:uids)"
            ),
            {
                "cids": [course_a, course_b, course_other_org],
                "uids": [bob_id, stale_teacher_id, enrolled_student_id, other_owner],
            },
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [course_a, course_b, course_other_org]},
        )
        await conn.execute(
            text("DELETE FROM org_units WHERE id = :id"),
            {"id": org_unit_b},
        )
        await conn.execute(
            text("DELETE FROM user_profiles WHERE user_id = ANY(:ids)"),
            {"ids": [bob_id, stale_teacher_id, enrolled_student_id]},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [bob_id, stale_teacher_id, enrolled_student_id, other_owner]},
        )
        await conn.execute(
            text("DELETE FROM users WHERE primary_email LIKE :pat"),
            {"pat": f"placeholder-{suffix}@abridgeai.local"},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"),
            {"id": other_org},
        )


def test_router_metadata() -> None:
    paths = {(r.path, tuple(sorted(r.methods))) for r in assignment_router.routes}  # type: ignore[attr-defined]
    expected = {
        ("/dept/courses", ("GET",)),
        ("/dept/courses/{course_id}/teachers", ("GET",)),
        ("/dept/courses/{course_id}/teachers", ("POST",)),
        ("/dept/courses/{course_id}/teachers/{user_id}", ("DELETE",)),
        ("/dept/courses/{course_id}/roster", ("GET",)),
        ("/dept/org-units/{org_unit_id}/courses", ("GET",)),
    }
    assert expected.issubset(paths)
    assert assignment_router.prefix == "/dept"


async def test_unauthenticated_returns_401(
    client: httpx.AsyncClient, scenario: dict[str, uuid.UUID]
) -> None:
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
    )
    assert response.status_code == 401
    response = await client.delete(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers/{scenario['bob_id']}"
    )
    assert response.status_code == 401


async def test_student_403_on_assignment(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    headers = {"Authorization": f"Bearer {student_bearer}"}
    response = await client.get("/api/v1/dept/courses", headers=headers)
    assert response.status_code == 403
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert response.status_code == 403
    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/roster",
        headers=headers,
    )
    assert response.status_code == 403
    response = await client.get(
        f"/api/v1/dept/org-units/{scenario['org_unit_a']}/courses",
        headers=headers,
    )
    assert response.status_code == 403


async def test_hod_scope_bound_can_assign_in_dept(
    client: httpx.AsyncClient,
    hod_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["course_id"] == str(scenario["course_a"])
    assert body["user_id"] == str(scenario["bob_id"])
    assert body["role_code"] == "teacher"
    assert body["scope_kind"] == "course"
    assert body["granted_by"] == str(seeded_users.hod_id)

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT ura.scope_kind, ura.course_id, ura.granted_by, r.code "
                    "FROM user_role_assignments ura JOIN roles r ON r.id = ura.role_id "
                    "WHERE ura.user_id = :uid AND ura.course_id = :cid "
                    "AND ura.deleted_at IS NULL "
                    "AND (ura.active_until IS NULL OR ura.active_until > NOW())"
                ),
                {"uid": scenario["bob_id"], "cid": scenario["course_a"]},
            )
        ).one()
    assert row.scope_kind == "course"
    assert row.code == "teacher"
    assert row.granted_by == seeded_users.hod_id


async def test_hod_scope_bound_blocks_outside_dept(
    client: httpx.AsyncClient,
    hod_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_b']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["error"] == "permission_denied"
    assert body["detail"]["scope"] == "course"


async def test_manager_can_assign_org_wide(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_b']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_other_org']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert response.status_code == 403


async def test_admin_can_assign_globally(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_other_org']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 201, response.text


async def test_remove_sets_active_until(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    headers = {"Authorization": f"Bearer {admin_bearer}"}
    create = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert create.status_code == 201, create.text

    delete = await client.delete(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers/{scenario['bob_id']}",
        headers=headers,
    )
    assert delete.status_code == 204

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT active_until FROM user_role_assignments "
                    "WHERE user_id = :uid AND course_id = :cid "
                    "ORDER BY active_until DESC NULLS LAST LIMIT 1"
                ),
                {"uid": scenario["bob_id"], "cid": scenario["course_a"]},
            )
        ).one()
    assert row.active_until is not None
    delta = datetime.now(tz=UTC) - row.active_until
    assert abs(delta.total_seconds()) < 5


async def test_list_courses_filters_by_caller_scope(
    client: httpx.AsyncClient,
    hod_bearer: str,
    manager_bearer: str,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    hod_resp = await client.get(
        "/api/v1/dept/courses",
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert hod_resp.status_code == 200, hod_resp.text
    hod_ids = {c["id"] for c in hod_resp.json()}
    assert str(scenario["course_a"]) in hod_ids
    assert str(scenario["course_b"]) not in hod_ids
    assert str(scenario["course_other_org"]) not in hod_ids

    mgr_resp = await client.get(
        "/api/v1/dept/courses",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert mgr_resp.status_code == 200
    mgr_ids = {c["id"] for c in mgr_resp.json()}
    assert str(scenario["course_a"]) in mgr_ids
    assert str(scenario["course_b"]) in mgr_ids
    assert str(scenario["course_other_org"]) not in mgr_ids

    admin_resp = await client.get(
        "/api/v1/dept/courses",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert admin_resp.status_code == 200
    admin_ids = {c["id"] for c in admin_resp.json()}
    assert str(scenario["course_a"]) in admin_ids
    assert str(scenario["course_b"]) in admin_ids
    assert str(scenario["course_other_org"]) in admin_ids


async def test_get_teachers_for_course_returns_active_only(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    create = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert create.status_code == 201, create.text

    listing = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert listing.status_code == 200, listing.text
    rows = listing.json()
    user_ids = {r["user_id"] for r in rows}
    assert str(scenario["bob_id"]) in user_ids
    assert str(scenario["stale_teacher_id"]) not in user_ids


async def test_get_org_unit_courses_hod_blocked_outside(
    client: httpx.AsyncClient,
    hod_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    same = await client.get(
        f"/api/v1/dept/org-units/{scenario['org_unit_a']}/courses",
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert same.status_code == 200, same.text
    other = await client.get(
        f"/api/v1/dept/org-units/{scenario['org_unit_b']}/courses",
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert other.status_code == 403


async def test_get_roster_returns_enrolled_students(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/roster",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    rows = response.json()
    student_ids = {r["student_id"] for r in rows}
    assert str(scenario["enrolled_student_id"]) in student_ids
    by_id = {r["student_id"]: r for r in rows}
    entry = by_id[str(scenario["enrolled_student_id"])]
    assert entry["status"] == "active"
    assert entry["display_name"] == "Enrolled Student"


def test_no_bare_get_current_user() -> None:
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "courses"
        / "routers"
        / "assignment.py"
    ).read_text(encoding="utf-8")
    bare = re.findall(r"Depends\(get_current_user\)", src)
    assert bare == [], f"assignment.py uses bare Depends(get_current_user): {bare}"


async def test_manager_can_edit_course_identity_via_dept(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """The dept surface is where course identity edits live: a manager
    holding ``course.delete`` at org scope can PATCH title/slug (identity)
    via ``/dept/courses/{id}``."""
    auth = {"Authorization": f"Bearer {manager_bearer}"}
    course_a = scenario["course_a"]

    async with engine.begin() as conn:
        original_slug = (
            await conn.execute(
                text("SELECT slug FROM courses WHERE id = :cid"),
                {"cid": course_a},
            )
        ).scalar_one()

    resp = await client.patch(
        f"/api/v1/dept/courses/{course_a}",
        json={"title": "Manager Renamed", "slug": f"renamed-{course_a.hex[:6]}"},
        headers=auth,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "Manager Renamed"

    # Restore original identity for suite isolation.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET title = 'Course A', slug = :slug WHERE id = :cid"),
            {"cid": course_a, "slug": original_slug},
        )


async def test_hod_cannot_edit_course_identity_via_dept(
    client: httpx.AsyncClient,
    hod_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """HOD holds staffing/roster codes but NOT ``course.delete`` — identity
    edits on the dept surface are manager-owned and 403 for them."""
    resp = await client.patch(
        f"/api/v1/dept/courses/{scenario['course_a']}",
        json={"title": "HOD Renamed"},
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["error"] == "permission_denied"


async def test_teacher_owner_cannot_edit_identity_via_dept(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Even the course owner cannot edit identity via the dept surface:
    ``allow_owner=False`` on the gate means only an explicit ``course.delete``
    grant passes (manager/admin)."""
    resp = await client.patch(
        f"/api/v1/dept/courses/{scenario['course_a']}",
        json={"title": "Owner Renamed"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert resp.status_code == 403, resp.text
