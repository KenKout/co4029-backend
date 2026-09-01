"""Integration tests for ``features.courses.routers.assignment`` (T3.8).

Covers the HOD/Manager teacher-staffing surface at ``/api/v1/dept``.
The acceptance criteria from plan §4485-4489 + §4491-4505 map to:

* ``test_router_metadata`` -- 6 endpoints registered under ``/dept``.
* ``test_unauthenticated_returns_401`` -- bearer-less requests rejected.
* ``test_student_403_on_assignment`` -- seeded student lacks
  ``course.assign_teacher`` / ``user.role_assign`` / ``system.administer``;
  every endpoint returns 403.
* ``test_faculty_dean_can_assign_in_faculty`` -- Faculty Dean assigns Teacher-Bob
  to a course in their Faculty -> 201, ``user_role_assignments`` row
  created with role=teacher, scope_kind=course, granted_by=HOD.
* ``test_faculty_dean_blocks_outside_faculty`` -- Faculty Dean assigning to a
  course in a sibling Faculty -> 403.
* ``test_manager_is_bound_to_assigned_faculty`` -- Faculty-scoped Manager can
  assign inside their Faculty, but not in another Faculty or Organization.
* ``test_admin_can_assign_globally`` -- Admin -> 201 across org boundary.
* ``test_remove_sets_active_until`` -- DELETE flips ``active_until`` to
  NOW (within 5s); preserves audit trail.
* ``test_list_courses_filters_by_caller_scope`` -- Faculty staff see only
  courses in their Faculty; Admin sees all.
* ``test_get_teachers_for_course_returns_active_only`` -- teachers with
  ``active_until`` in the past are excluded.
* ``test_get_faculty_courses_dean_blocked_outside`` -- Dean passing a
  sibling Faculty id -> 403.
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
import pytest
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
from abridgeai.core.exceptions import ConflictError
from abridgeai.core.security import CurrentUser, create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import assignment_router
from abridgeai.features.courses.services import authoring as authoring_service

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
    """Two-organization / two-Faculty staffing layout.

    * ``course_a`` -- in the Dean's seeded Faculty.
    * ``org_unit_b`` -- sibling Faculty in the same Organization.
    * ``course_b`` -- in ``org_unit_b``; the Dean has NO scope here.
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
                "VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": org_unit_b,
                "org": seeded_users.organization_id,
                "name": f"Other Faculty {suffix}",
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
        # Org membership. Assignment requires the assignee to be a member of
        # the course's organization (server-side, not a UI convention), so the
        # fixture has to model what real users have: every teacher in the live
        # database carries an active organization_memberships row.
        await conn.execute(
            text(
                "INSERT INTO organization_memberships (user_id, organization_id, status) "
                "VALUES (:bid, :org, 'active'), (:sid, :org, 'active'), "
                "(:eid, :org, 'active')"
            ),
            {
                "bid": bob_id,
                "sid": stale_teacher_id,
                "eid": enrolled_student_id,
                "org": seeded_users.organization_id,
            },
        )
        teacher_role_id = (
            await conn.execute(text("SELECT id FROM roles WHERE code = 'teacher'"))
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id) "
                "VALUES (:uid, :rid, 'organization', :org)"
            ),
            {
                "uid": bob_id,
                "rid": teacher_role_id,
                "org": seeded_users.organization_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_faculty_assignments "
                "(user_id, organization_id, faculty_id, status) VALUES "
                "(:uid, :org, :faculty_a, 'active'), "
                "(:uid, :org, :faculty_b, 'active')"
            ),
            {
                "uid": bob_id,
                "org": seeded_users.organization_id,
                "faculty_a": seeded_users.org_unit_id,
                "faculty_b": org_unit_b,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, faculty_id, owner_user_id, "
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
                # NOT the manager: the course owner short-circuits the
                # permission lookup, so a manager owning course_b would be
                # granted by OWNERSHIP and the faculty-scope assertion below
                # would pass for the wrong reason (it returned 201, not 403).
                "owner_b": seeded_users.admin_id,
                "slug_b": f"course-b-{suffix}",
                "c": course_other_org,
                "org_c": other_org,
                "owner_c": other_owner,
                "slug_c": f"other-course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id, course_id, "
                "active_from, active_until, granted_by, is_instructor, is_assistant) "
                "VALUES (:uid, :rid, 'course', :org, :cid, "
                "NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day', :gb, true, false)"
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
            text("DELETE FROM user_faculty_assignments WHERE user_id = :uid"),
            {"uid": bob_id},
        )
        await conn.execute(
            text("DELETE FROM org_units WHERE id = :id"),
            {"id": org_unit_b},
        )
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE user_id = ANY(:ids)"),
            {"ids": [bob_id, stale_teacher_id, enrolled_student_id]},
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
        ("/dept/faculties/{faculty_id}/courses", ("GET",)),
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
        f"/api/v1/dept/faculties/{scenario['org_unit_a']}/courses",
        headers=headers,
    )
    assert response.status_code == 403


async def test_faculty_dean_can_assign_in_faculty(
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


async def test_faculty_dean_blocks_outside_faculty(
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


async def test_manager_is_bound_to_assigned_faculty(
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
    assert response.status_code == 403, response.text
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_other_org']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert response.status_code == 403


async def test_admin_assign_still_requires_the_teacher_to_be_in_the_course_org(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Admin's global reach is about WHICH COURSES they may staff, not about
    waiving org membership for the assignee.

    This test previously asserted 201 for "admin assigns Bob (org A) to a
    course in another org". Bob would have received course.update on a course
    of an organization he is not a member of — cross-tenant access created by
    a staffing action. Admin can still staff any course; they just have to
    pick someone who belongs to that course's organization.
    """
    response = await client.post(
        f"/api/v1/dept/courses/{scenario['course_other_org']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 403, response.text
    assert "teacher_not_in_course_org" in response.json()["detail"]["message"]


async def test_admin_can_assign_a_teacher_of_that_org(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """The reach itself is intact: give the assignee membership in the other
    org and the same admin assignment goes through."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organization_memberships (user_id, organization_id, status) "
                "VALUES (:uid, :org, 'active')"
            ),
            {"uid": scenario["bob_id"], "org": scenario["other_org"]},
        )
    try:
        response = await client.post(
            f"/api/v1/dept/courses/{scenario['course_other_org']}/teachers",
            json={"user_id": str(scenario["bob_id"])},
            headers={"Authorization": f"Bearer {admin_bearer}"},
        )
        assert response.status_code == 201, response.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM organization_memberships "
                    "WHERE user_id = :uid AND organization_id = :org"
                ),
                {"uid": scenario["bob_id"], "org": scenario["other_org"]},
            )


async def test_assignable_teachers_lists_only_in_org_teachers(
    client: httpx.AsyncClient,
    admin_bearer: str,
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The picker's source of truth. Org comes from the COURSE, not the client.

    Replaces "paste a user UUID" on the manager surface. The seeded teacher
    holds the teacher role in org A, so they must appear for a course in org A
    and NOT for the other-org course — otherwise the picker would offer an
    assignment that the POST then 403s on.
    """
    headers = {"Authorization": f"Bearer {admin_bearer}"}
    same_org = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/assignable-teachers",
        headers=headers,
    )
    assert same_org.status_code == 200, same_org.text
    ids = {row["user_id"] for row in same_org.json()}
    assert str(seeded_users.teacher_id) in ids

    other_org = await client.get(
        f"/api/v1/dept/courses/{scenario['course_other_org']}/assignable-teachers",
        headers=headers,
    )
    assert other_org.status_code == 200, other_org.text
    other_ids = {row["user_id"] for row in other_org.json()}
    assert str(seeded_users.teacher_id) not in other_ids, (
        "a teacher of another organization must not be offered"
    )


async def test_assignable_teachers_excludes_users_without_the_teacher_role(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    admin_bearer: str,
    seeded_users: SeededUsers,
) -> None:
    """Org membership alone is not enough — the picker offers TEACHERS.

    The enrolled student and Faculty Manager are active members of the
    Organization, but neither holds the Teacher role.
    """
    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/assignable-teachers",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    ids = {row["user_id"] for row in response.json()}
    assert str(scenario["enrolled_student_id"]) not in ids
    assert str(seeded_users.manager_id) not in ids


async def test_assignable_teachers_flags_already_assigned(
    client: httpx.AsyncClient,
    admin_bearer: str,
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """So the picker can show current teachers as chosen instead of offering a
    no-op assignment."""
    headers = {"Authorization": f"Bearer {admin_bearer}"}
    assign = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/teachers",
        json={"user_id": str(seeded_users.teacher_id)},
        headers=headers,
    )
    assert assign.status_code == 201, assign.text

    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/assignable-teachers",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    row = next(r for r in response.json() if r["user_id"] == str(seeded_users.teacher_id))
    assert row["already_assigned"] is True
    # And carries something human-readable — the whole point of the selector.
    assert row["primary_email"]


async def test_assignable_teachers_requires_staffing_permission(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """A teacher cannot enumerate who else could be staffed."""
    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_a']}/assignable-teachers",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403, response.text


async def test_assignable_teachers_for_a_new_course_uses_the_callers_org(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
) -> None:
    """The create wizard picks teachers before any course exists.

    Org comes from the caller's token, so the list must match what the
    per-course endpoint would return for a course this manager creates.
    """
    response = await client.get(
        "/api/v1/dept/assignable-teachers",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    rows = response.json()
    ids = {row["user_id"] for row in rows}
    assert str(seeded_users.teacher_id) in ids
    # Nothing to be assigned to yet, so the flag must be uniformly false rather
    # than leaking state from some other course.
    assert all(row["already_assigned"] is False for row in rows)


async def test_assignable_teachers_for_a_new_course_excludes_other_orgs(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The org restriction has to hold on this route too, not just the scoped one.

    ``other_owner`` lives in a different organization, so it must not appear
    regardless of any role it holds.
    """
    response = await client.get(
        "/api/v1/dept/assignable-teachers",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    ids = {row["user_id"] for row in response.json()}
    assert str(scenario["other_owner"]) not in ids


async def test_assignable_teachers_for_a_new_course_requires_staffing_permission(
    client: httpx.AsyncClient,
    teacher_bearer: str,
) -> None:
    response = await client.get(
        "/api/v1/dept/assignable-teachers",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403, response.text


async def test_readiness_reports_an_empty_course_as_not_publishable(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The checklist's whole purpose: say it BEFORE publish, not as a 409 after.

    The scenario's course_b has no published lesson, no teacher and no career
    path — the state a freshly created course is in.
    """
    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_b']}/readiness",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["gradeable_unit_count"] == 0
    assert body["can_publish"] is False
    assert body["career_paths"] == []
    assert body["status"] == "draft"


def _service_actor(user_id: uuid.UUID) -> CurrentUser:
    """Service-level actor for the one assertion that bypasses HTTP."""
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def test_readiness_can_publish_matches_the_publish_gate(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
) -> None:
    """`can_publish` must agree with the gate, or the checklist lies.

    Publish the course through the API right after the checklist says it can be
    published: a green row followed by a 409 would be worse than no row at all.
    """
    headers = {"Authorization": f"Bearer {admin_bearer}"}
    # course_a, not course_b: the publish route resolves through the course's
    # org_unit, and course_b sits in a unit the admin has no path to (404).
    course_id = scenario["course_a"]
    # This fixture seeds courses with no curriculum at all, so create the one
    # gradeable unit rather than promoting a lesson that does not exist.
    module_id, lesson_id, outcome_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, 'M1', 1, 'published')"
            ),
            {"id": module_id, "cid": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:id, :mid, :slug, 'L1', 'published', 'video')"
            ),
            {
                "id": lesson_id,
                "mid": module_id,
                "slug": f"rl-{uuid.uuid4().hex[:8]}",
            },
        )

    try:
        # Content but no outcomes: the second gate is unmet, so the checklist
        # must NOT show green. Asserted before satisfying it — a checklist that
        # only ever gets checked in the ready state cannot catch disagreement.
        partial = await client.get(f"/api/v1/dept/courses/{course_id}/readiness", headers=headers)
        assert partial.status_code == 200, partial.text
        assert partial.json()["gradeable_unit_count"] == 1
        assert partial.json()["learning_outcome_count"] == 0
        assert partial.json()["can_publish"] is False

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO course_learning_outcomes "
                    "(id, course_id, position, outcome_text) "
                    "VALUES (:id, :cid, 1, 'State the outcome')"
                ),
                {"id": outcome_id, "cid": course_id},
            )

        readiness = await client.get(f"/api/v1/dept/courses/{course_id}/readiness", headers=headers)
        assert readiness.status_code == 200, readiness.text
        assert readiness.json()["gradeable_unit_count"] == 1
        assert readiness.json()["learning_outcome_count"] == 1
        # Staffing gate (user decision 2026-08-18): a draft must meet the
        # default min (2) before it can publish, so course_a — which has NO
        # active teachers — must stay red even with content + outcomes.
        assert readiness.json()["teacher_count"] == 0
        assert readiness.json()["min_teachers_per_course"] == 2
        assert readiness.json()["can_publish"] is False

        # One teacher (the Course Instructor) is still below the min of 2.
        assign_ci = await client.post(
            f"/api/v1/dept/courses/{course_id}/teachers",
            json={
                "user_id": str(seeded_users.teacher_id),
                "is_instructor": True,
            },
            headers=headers,
        )
        assert assign_ci.status_code == 201, assign_ci.text
        one = await client.get(f"/api/v1/dept/courses/{course_id}/readiness", headers=headers)
        assert one.json()["teacher_count"] == 1
        assert one.json()["can_publish"] is False

        # A second teacher (Teacher Assistant) satisfies the min.
        assign_ta = await client.post(
            f"/api/v1/dept/courses/{course_id}/teachers",
            json={
                "user_id": str(scenario["bob_id"]),
                "is_assistant": True,
            },
            headers=headers,
        )
        assert assign_ta.status_code == 201, assign_ta.text
        ready = await client.get(f"/api/v1/dept/courses/{course_id}/readiness", headers=headers)
        assert ready.json()["teacher_count"] == 2
        assert ready.json()["course_instructor_count"] == 1
        assert ready.json()["can_publish"] is True

        # Cross-check against the gate itself. The publish ROUTE lives on the
        # teacher router, which this test app does not mount (it mounts only
        # the dept router), so call the gate directly: the point is that
        # can_publish and the gate agree, not which URL carries it.
        async with session_factory() as db:
            published = await authoring_service.publish_course(
                db, course_id, _service_actor(seeded_users.admin_id)
            )
            await db.commit()
        assert published.status == "published"
    finally:
        async with engine.begin() as conn:
            # Restore the shared seeded course to draft: it belongs to the
            # session-scoped seed and LATER suites (authoring_router) assume it
            # is a fresh draft.
            await conn.execute(
                text("UPDATE courses SET status = 'draft' WHERE id = :id"),
                {"id": course_id},
            )
            await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
            await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
            await conn.execute(
                text("DELETE FROM course_learning_outcomes WHERE id = :id"),
                {"id": outcome_id},
            )


async def test_readiness_counts_assigned_teachers(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    headers = {"Authorization": f"Bearer {admin_bearer}"}
    before = await client.get(
        f"/api/v1/dept/courses/{scenario['course_b']}/readiness", headers=headers
    )
    assert before.json()["teacher_count"] == 0

    assign = await client.post(
        f"/api/v1/dept/courses/{scenario['course_b']}/teachers",
        json={"user_id": str(scenario["bob_id"])},
        headers=headers,
    )
    assert assign.status_code == 201, assign.text

    after = await client.get(
        f"/api/v1/dept/courses/{scenario['course_b']}/readiness", headers=headers
    )
    assert after.json()["teacher_count"] == 1


async def test_readiness_flags_a_required_course_that_locks_its_stage(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """The urgent case, separated from plain "no content".

    A REQUIRED course with nothing to grade does not merely fail to complete: no
    student can ever satisfy it, so its stage and every stage behind it stay
    locked forever. That is worth shouting about; an optional empty course is
    not.
    """
    path_id, stage_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        org_id = (
            await conn.execute(
                text("SELECT organization_id FROM courses WHERE id = :c"),
                {"c": scenario["course_b"]},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, 'Readiness Path', 'draft')"
            ),
            {
                "id": path_id,
                "org": org_id,
                "slug": f"rp-{uuid.uuid4().hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO career_path_versions "
                "(id, career_path_id, version_no, status) "
                "VALUES (gen_random_uuid(), :pid, 1, 'draft')"
            ),
            {"pid": path_id},
        )
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, version_id, position, title, unlock_policy, enforcement) "
                "VALUES (:id, (SELECT id FROM career_path_versions "
                "WHERE career_path_id = :pid AND version_no = 1), 1, 'Stage 1', 'always', 'advisory')"
            ),
            {"id": stage_id, "pid": path_id},
        )
        await conn.execute(
            text(
                "INSERT INTO career_course_items "
                "(version_id, course_id, stage_id, position, is_required) "
                "VALUES ((SELECT id FROM career_path_versions "
                "WHERE career_path_id = :pid AND version_no = 1), :cid, :sid, 1, TRUE)"
            ),
            {"pid": path_id, "cid": scenario["course_b"], "sid": stage_id},
        )
    try:
        response = await client.get(
            f"/api/v1/dept/courses/{scenario['course_b']}/readiness",
            headers={"Authorization": f"Bearer {admin_bearer}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["blocks_required_stage"] is True
        assert len(body["career_paths"]) == 1
        placement = body["career_paths"][0]
        assert placement["career_path_name"] == "Readiness Path"
        assert placement["is_required"] is True
        assert placement["stage_position"] == 1
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM career_course_items WHERE version_id IN "
                    "(SELECT id FROM career_path_versions WHERE career_path_id = :p)"
                ),
                {"p": path_id},
            )
            await conn.execute(
                text(
                    "DELETE FROM career_path_stages WHERE version_id IN "
                    "(SELECT id FROM career_path_versions WHERE career_path_id = :p)"
                ),
                {"p": path_id},
            )
            await conn.execute(text("DELETE FROM career_paths WHERE id = :p"), {"p": path_id})


async def test_readiness_requires_staffing_permission(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        f"/api/v1/dept/courses/{scenario['course_b']}/readiness",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403, response.text


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
    assert str(scenario["course_b"]) not in mgr_ids
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


async def test_get_faculty_courses_dean_blocked_outside(
    client: httpx.AsyncClient,
    hod_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    same = await client.get(
        f"/api/v1/dept/faculties/{scenario['org_unit_a']}/courses",
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert same.status_code == 200, same.text
    other = await client.get(
        f"/api/v1/dept/faculties/{scenario['org_unit_b']}/courses",
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


async def test_hod_can_edit_course_identity_via_dept(
    client: httpx.AsyncClient,
    hod_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """Dean (``hod``) subsumes Manager (role change 2026-08-25) — including
    ``course.delete``, so identity edits on the dept surface are NOT 403 for
    a dean; a plain TEACHER is the role that must be refused (see
    ``test_teacher_owner_cannot_edit_identity_via_dept``).
    """
    resp = await client.patch(
        f"/api/v1/dept/courses/{scenario['course_a']}",
        json={"title": "HOD Renamed"},
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert resp.status_code == 200, resp.text
    # Restore the title for suite isolation (same pattern as the manager test).
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET title = 'Course A' WHERE id = :cid"),
            {"cid": scenario["course_a"]},
        )


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


async def test_manager_can_clone_course_via_dept(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """Manager-only course clone: a manager holding ``course.delete`` clones
    an org course at the requested depth (201 + fresh draft course)."""
    auth = {"Authorization": f"Bearer {manager_bearer}"}
    course_a = scenario["course_a"]

    async with engine.begin() as conn:
        source_slug = (
            await conn.execute(
                text("SELECT slug FROM courses WHERE id = :cid"),
                {"cid": course_a},
            )
        ).scalar_one()

    resp = await client.post(
        f"/api/v1/dept/courses/{course_a}/clone",
        json={"depth": "shell"},
        headers=auth,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"] != str(course_a)
    assert body["status"] == "draft"
    assert body["slug"] == f"{source_slug}-copy"
    assert body["title"] == "Course A (Copy)"

    # The clone lives in the same org as the source but is owned by the actor
    # (the manager), not the source owner.
    async with engine.begin() as conn:
        source_org, source_owner = (
            await conn.execute(
                text("SELECT organization_id, owner_user_id FROM courses WHERE id = :cid"),
                {"cid": course_a},
            )
        ).one()
        clone_org, clone_owner, clone_status = (
            await conn.execute(
                text("SELECT organization_id, owner_user_id, status FROM courses WHERE id = :cid"),
                {"cid": uuid.UUID(body["id"])},
            )
        ).one()
        await conn.execute(
            text("DELETE FROM courses WHERE id = :cid"),
            {"cid": uuid.UUID(body["id"])},
        )

    assert clone_org == source_org
    assert clone_owner != source_owner
    assert clone_status == "draft"


async def test_teacher_cannot_clone_course_via_dept(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Even the course owner cannot clone via the dept surface —
    ``course.delete`` (manager-only) gates the clone endpoint."""
    resp = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/clone",
        json={"depth": "full"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["error"] == "permission_denied"


async def test_clone_course_rejects_missing_depth(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Depth is required — no hidden default decides how much to copy."""
    resp = await client.post(
        f"/api/v1/dept/courses/{scenario['course_a']}/clone",
        json={},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert resp.status_code == 422, resp.text


# --- Course teacher titles + min/max staffing (user decision 2026-08-18) -----


@pytest_asyncio.fixture
async def staffing_course(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, str]]:
    """A fresh draft course in the seeded org + three assignable teachers."""
    course_id = uuid.uuid4()
    ta1, ta2 = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Staffing Course', 'draft')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.manager_id,
                "slug": f"staff-{uuid.uuid4().hex[:8]}",
            },
        )
        for uid, email in ((ta1, "ta-1"), (ta2, "ta-2")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
                {"id": uid, "e": f"{email}-{uuid.uuid4().hex[:6]}@abridgeai.local"},
            )
            await conn.execute(
                text(
                    "INSERT INTO organization_memberships (id, user_id, organization_id, status) "
                    "VALUES (gen_random_uuid(), :u, :org, 'active')"
                ),
                {"u": uid, "org": seeded_users.organization_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments (id, user_id, role_id, scope_kind, "
                    "organization_id, granted_by) "
                    "SELECT gen_random_uuid(), :u, r.id, 'organization', :org, :g "
                    "FROM roles r WHERE r.code = 'teacher'"
                ),
                {"u": uid, "org": seeded_users.organization_id, "g": seeded_users.admin_id},
            )
    yield {
        "course_id": str(course_id),
        "ci_id": str(seeded_users.teacher_id),
        "ta1_id": str(ta1),
        "ta2_id": str(ta2),
        "org_id": str(seeded_users.organization_id),
    }
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE course_id = :c"), {"c": course_id}
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        for uid in (ta1, ta2):
            await conn.execute(
                text(
                    "DELETE FROM user_role_assignments WHERE user_id = :u "
                    "AND scope_kind = 'organization'"
                ),
                {"u": uid},
            )
            await conn.execute(
                text("DELETE FROM organization_memberships WHERE user_id = :u"), {"u": uid}
            )
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})


async def test_first_teacher_is_instructor_rest_are_assistants(
    client: httpx.AsyncClient, manager_bearer: str, staffing_course: dict[str, str]
) -> None:
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    cid = staffing_course["course_id"]

    r1 = await client.post(
        f"/api/v1/dept/courses/{cid}/teachers",
        json={"user_id": staffing_course["ci_id"]},
        headers=headers,
    )
    assert r1.status_code == 201, r1.text
    assert r1.json()["is_instructor"] is True
    assert r1.json()["is_assistant"] is False

    r2 = await client.post(
        f"/api/v1/dept/courses/{cid}/teachers",
        json={"user_id": staffing_course["ta1_id"], "is_assistant": True},
        headers=headers,
    )
    assert r2.status_code == 201, r2.text
    assert r2.json()["is_instructor"] is False
    assert r2.json()["is_assistant"] is True


async def test_assigning_beyond_max_is_rejected(
    client: httpx.AsyncClient,
    manager_bearer: str,
    staffing_course: dict[str, str],
    engine: AsyncEngine,
) -> None:
    """Assigning a teacher past courses.max_teachers_per_course -> 409."""
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    cid = staffing_course["course_id"]
    # Cap this course's org at 2 teachers.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO system_settings (organization_id, setting_key, "
                "setting_value_json) VALUES (:o, 'courses.max_teachers_per_course', '2')"
            ),
            {"o": uuid.UUID(staffing_course["org_id"])},
        )
    from abridgeai.core.runtime_settings import invalidate_settings_cache

    invalidate_settings_cache()

    for uid in (staffing_course["ci_id"], staffing_course["ta1_id"]):
        resp = await client.post(
            f"/api/v1/dept/courses/{cid}/teachers", json={"user_id": uid}, headers=headers
        )
        assert resp.status_code == 201, resp.text

    over = await client.post(
        f"/api/v1/dept/courses/{cid}/teachers",
        json={"user_id": staffing_course["ta2_id"]},
        headers=headers,
    )
    assert over.status_code == 409, over.text
    assert over.json()["detail"]["error"] == "conflict"

    # Restore the org cap (affects the shared seeded org).
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM system_settings WHERE organization_id = :o "
                "AND setting_key = 'courses.max_teachers_per_course'"
            ),
            {"o": uuid.UUID(staffing_course["org_id"])},
        )
    invalidate_settings_cache()


async def test_a_second_instructor_can_be_promoted(
    client: httpx.AsyncClient, manager_bearer: str, staffing_course: dict[str, str]
) -> None:
    """Multiple Course Instructors are legal (user decision 2026-08-30)."""
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    cid = staffing_course["course_id"]
    for uid in (staffing_course["ci_id"], staffing_course["ta1_id"]):
        await client.post(
            f"/api/v1/dept/courses/{cid}/teachers", json={"user_id": uid}, headers=headers
        )

    promote = await client.put(
        f"/api/v1/dept/courses/{cid}/teachers/{staffing_course['ta1_id']}/role",
        json={"is_instructor": True, "is_assistant": True},
        headers=headers,
    )
    assert promote.status_code == 200, promote.text
    assert promote.json()["is_instructor"] is True
    assert promote.json()["is_assistant"] is True


async def test_removing_the_sole_instructor_is_rejected_when_tas_exist(
    client: httpx.AsyncClient, manager_bearer: str, staffing_course: dict[str, str]
) -> None:
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    cid = staffing_course["course_id"]
    await client.post(
        f"/api/v1/dept/courses/{cid}/teachers",
        json={"user_id": staffing_course["ci_id"]},
        headers=headers,
    )
    await client.post(
        f"/api/v1/dept/courses/{cid}/teachers",
        json={"user_id": staffing_course["ta1_id"]},
        headers=headers,
    )

    remove = await client.delete(
        f"/api/v1/dept/courses/{cid}/teachers/{staffing_course['ci_id']}",
        headers=headers,
    )
    assert remove.status_code == 409, remove.text


async def test_publish_below_min_teachers_is_rejected(
    client: httpx.AsyncClient,
    manager_bearer: str,
    staffing_course: dict[str, str],
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """First publish of a draft with fewer than min teachers -> 409 (min=2)."""
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    cid = uuid.UUID(staffing_course["course_id"])
    # One gradeable unit + one outcome, so only the staffing gate can block.
    module_id, lesson_id, outcome_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :c, 'M1', 1, 'published')"
            ),
            {"id": module_id, "c": cid},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:id, :m, :s, 'L1', 'published', 'video')"
            ),
            {"id": lesson_id, "m": module_id, "s": f"pl-{uuid.uuid4().hex[:6]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO course_learning_outcomes (id, course_id, position, outcome_text) "
                "VALUES (:id, :c, 1, 'Outcome')"
            ),
            {"id": outcome_id, "c": cid},
        )
    # One teacher only: below the default min of 2.
    await client.post(
        f"/api/v1/dept/courses/{cid}/teachers",
        json={"user_id": staffing_course["ci_id"]},
        headers=headers,
    )

    from abridgeai.features.courses.services import authoring as authoring_service

    async with session_factory() as db:
        # publish_course ignores the actor's identity, so a throwaway actor is fine.
        with pytest.raises(ConflictError, match="course_teacher_min_not_met"):
            await authoring_service.publish_course(db, cid, _service_actor(uuid.uuid4()))
        await db.rollback()

    # Tidy the content added above so the fixture teardown can drop the course.
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :c"), {"c": cid}
        )
        await conn.execute(
            text("DELETE FROM lesson_resources WHERE lesson_id = :l"), {"l": lesson_id}
        )
        await conn.execute(text("DELETE FROM module_items WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM lessons WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
