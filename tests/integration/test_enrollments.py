"""Integration tests for ``features.enrollments`` (T7.1).

Covers the Manager-assigned enrollment surface and the locked
"no self-enroll" + "no student redemption" invariants.
"""

from __future__ import annotations

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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses + organizations
import abridgeai.features.enrollments.models  # noqa: F401  -- register enrollments tables
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.enrollments.routers import (
    assignment_dept_router,
    assignment_management_router,
    me_enrollments_router,
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
    fastapi_app.include_router(me_enrollments_router, prefix="/api/v1")
    fastapi_app.include_router(assignment_dept_router, prefix="/api/v1")
    fastapi_app.include_router(assignment_management_router, prefix="/api/v1")
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
async def manager_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.manager_id)
    yield create_access_token(user_id=seeded_users.manager_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def hod_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.hod_id)
    yield create_access_token(user_id=seeded_users.hod_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, object]]:
    """Seed a fresh course (in the seeded org/unit) plus 5 students for bulk-enroll."""
    suffix = uuid.uuid4().hex[:8]
    course_id = uuid.uuid4()
    student_ids = [uuid.uuid4() for _ in range(5)]
    student_emails = [f"enr-stu-{i}-{suffix}@abridgeai.local" for i in range(5)]

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, org_unit_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :ou, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "ou": seeded_users.org_unit_id,
                "owner": seeded_users.admin_id,
                "slug": f"enr-course-{suffix}",
                "title": f"Enrollment Course {suffix}",
            },
        )
        for sid_, email in zip(student_ids, student_emails, strict=True):
            await conn.execute(
                text(
                    "INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"
                ),
                {"id": sid_, "email": email},
            )
            await conn.execute(
                text(
                    "INSERT INTO user_profiles (user_id, given_name, family_name, display_name) "
                    "VALUES (:uid, 'Enr', 'Student', :dn)"
                ),
                {"uid": sid_, "dn": f"Enr Student {sid_.hex[:6]}"},
            )

    yield {
        "course_id": course_id,
        "student_ids": student_ids,
        "student_emails": student_emails,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(
            text("DELETE FROM course_invitation_codes WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = :id"),
            {"id": course_id},
        )
        await conn.execute(
            text("DELETE FROM user_profiles WHERE user_id = ANY(:ids)"),
            {"ids": student_ids},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": student_ids},
        )
        await conn.execute(
            text("DELETE FROM users WHERE primary_email LIKE :pat"),
            {"pat": f"%-{suffix}@abridgeai.local"},
        )


def test_router_metadata() -> None:
    learner_paths = {(r.path, tuple(sorted(r.methods))) for r in me_enrollments_router.routes}  # type: ignore[attr-defined]
    assert ("/me/enrollments", ("GET",)) in learner_paths
    assert ("/me/enrollments/{course_id}", ("GET",)) in learner_paths
    learner_methods = {m for _, methods in learner_paths for m in methods}
    assert "POST" not in learner_methods


def test_no_self_enroll_route_exists() -> None:
    """`POST /me/enrollments` MUST NOT exist (T7.1 locked decision)."""
    learner_paths = {
        (r.path, tuple(sorted(r.methods)))  # type: ignore[attr-defined]
        for r in me_enrollments_router.routes  # type: ignore[attr-defined]
    }
    forbidden = {
        ("/me/enrollments", ("POST",)),
        ("/me/enrollments/", ("POST",)),
    }
    assert not (forbidden & learner_paths)


def test_invitation_code_no_student_redemption() -> None:
    """No student-facing redemption endpoint for invitation codes."""
    learner_paths = {r.path for r in me_enrollments_router.routes}  # type: ignore[attr-defined]
    redemption_shapes = {
        "/me/enrollments/redeem",
        "/me/enrollments/redeem-code",
        "/me/invitation-codes/redeem",
        "/me/enrollments/invitation-codes",
    }
    assert not (redemption_shapes & learner_paths)


async def test_post_me_enrollments_returns_404_or_405(
    client: httpx.AsyncClient, student_bearer: str
) -> None:
    response = await client.post(
        "/api/v1/me/enrollments",
        json={"course_id": str(uuid.uuid4())},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code in (404, 405)


async def test_manager_bulk_enroll(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    course_id = scenario["course_id"]
    student_ids = scenario["student_ids"]  # type: ignore[assignment]
    response = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(sid) for sid in student_ids]},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["enrolled"]) == 5
    assert body["failures"] == []

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT student_id, status, source FROM course_enrollments "
                    "WHERE course_id = :cid"
                ),
                {"cid": course_id},
            )
        ).all()
    assert len(rows) == 5
    assert all(r.status == "active" for r in rows)
    assert all(r.source == "manager_bulk" for r in rows)


async def test_dept_course_enrollments_includes_student_identity(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
) -> None:
    """``GET /dept/courses/{id}/enrollments`` must surface the enrolled
    student's name/email, not just their raw UUID.

    Regression: the roster tab on the Manager enrollments page used to
    render a bare ``student_id`` because ``EnrollmentAuthoring`` had no
    identity fields and the query never joined ``users``/``user_profiles``.
    """
    course_id = scenario["course_id"]
    student_ids = scenario["student_ids"]  # type: ignore[assignment]
    student_emails = scenario["student_emails"]  # type: ignore[assignment]
    await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(sid) for sid in student_ids]},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )

    response = await client.get(
        f"/api/v1/dept/courses/{course_id}/enrollments",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 5
    by_student = {row["student_id"]: row for row in body}
    for sid_, email in zip(student_ids, student_emails, strict=True):
        row = by_student[str(sid_)]
        assert row["primary_email"] == email
        assert row["display_name"] == f"Enr Student {sid_.hex[:6]}"


async def test_already_enrolled_rejected(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
) -> None:
    course_id = scenario["course_id"]
    student_ids = scenario["student_ids"]  # type: ignore[assignment]
    headers = {"Authorization": f"Bearer {manager_bearer}"}

    first = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(student_ids[0])]},
        headers=headers,
    )
    assert first.status_code == 200

    second = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(sid) for sid in student_ids]},
        headers=headers,
    )
    assert second.status_code == 200
    body = second.json()
    assert len(body["enrolled"]) == 4
    assert len(body["failures"]) == 1
    assert body["failures"][0]["identifier"] == str(student_ids[0])
    assert body["failures"][0]["reason"] == "already_enrolled"


async def test_teacher_cannot_enroll(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, object],
) -> None:
    course_id = scenario["course_id"]
    student_ids = scenario["student_ids"]  # type: ignore[assignment]
    response = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(student_ids[0])]},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403


async def test_unenroll_sets_dropped_status(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    course_id = scenario["course_id"]
    student_ids = scenario["student_ids"]  # type: ignore[assignment]
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    enroll_resp = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(student_ids[0])]},
        headers=headers,
    )
    assert enroll_resp.status_code == 200

    drop_resp = await client.delete(
        f"/api/v1/management/courses/{course_id}/enrollments/{student_ids[0]}",
        headers=headers,
    )
    assert drop_resp.status_code == 204

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT status, dropped_at FROM course_enrollments "
                    "WHERE course_id = :cid AND student_id = :sid"
                ),
                {"cid": course_id, "sid": student_ids[0]},
            )
        ).one_or_none()
    assert row is not None
    assert row.status == "dropped"
    assert row.dropped_at is not None


async def test_csv_import_partial_failure(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    course_id = scenario["course_id"]
    suffix = uuid.uuid4().hex[:6]
    valid_lines = [
        "email,given_name,family_name",
        f"valid-1-{suffix}@abridgeai.local,Alpha,Tester",
        f"valid-2-{suffix}@abridgeai.local,Bravo,Tester",
        f"valid-3-{suffix}@abridgeai.local,Charlie,Tester",
        f"valid-4-{suffix}@abridgeai.local,Delta,Tester",
        f"valid-5-{suffix}@abridgeai.local,Echo,Tester",
        "not-an-email,Foxtrot,Tester",
        "another-bad-row@,Golf,Tester",
    ]
    csv_text = "\n".join(valid_lines)

    response = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/import-csv",
        json={"csv_text": csv_text},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["enrolled"]) == 5
    assert len(body["failures"]) == 2
    assert len(body["created_users"]) == 5

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text("SELECT COUNT(*) AS c FROM course_enrollments WHERE course_id = :cid"),
                {"cid": course_id},
            )
        ).one()
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(
            text("DELETE FROM user_profiles WHERE user_id = ANY(:ids)"),
            {"ids": body["created_users"]},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": body["created_users"]},
        )
    assert rows.c == 5


async def test_invitation_code_create_and_expire(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    course_id = scenario["course_id"]
    suffix = uuid.uuid4().hex[:6]
    code_str = f"INV-{suffix}"
    headers = {"Authorization": f"Bearer {manager_bearer}"}

    create_resp = await client.post(
        f"/api/v1/management/courses/{course_id}/invitation-codes",
        json={"code": code_str, "max_uses": 10},
        headers=headers,
    )
    assert create_resp.status_code == 201, create_resp.text
    code_body = create_resp.json()
    code_id = code_body["id"]
    assert code_body["code"] == code_str
    assert code_body["current_uses"] == 0
    assert code_body["is_active"] is True

    list_resp = await client.get(
        f"/api/v1/management/courses/{course_id}/invitation-codes",
        headers=headers,
    )
    assert list_resp.status_code == 200
    assert any(c["id"] == code_id for c in list_resp.json())

    expire_resp = await client.delete(
        f"/api/v1/management/invitation-codes/{code_id}",
        headers=headers,
    )
    assert expire_resp.status_code == 204

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT is_active, deleted_at FROM course_invitation_codes WHERE id = :id"),
                {"id": code_id},
            )
        ).one_or_none()
    assert row is not None
    assert row.is_active is False
    assert row.deleted_at is not None

    list_after = await client.get(
        f"/api/v1/management/courses/{course_id}/invitation-codes",
        headers=headers,
    )
    assert list_after.status_code == 200
    assert all(c["id"] != code_id for c in list_after.json())


async def test_invitation_code_duplicate_returns_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    """Re-submitting an invitation code that already exists must return 409.

    Regression: ``create_invitation_code`` used to raise ``ValueError`` for
    a known-duplicate, which the router mapped to 409, but a true UNIQUE
    race (e.g. two managers POSTing the same code at the same time) hit
    Postgres's ``course_invitation_codes_code_key`` and bubbled up as a
    raw IntegrityError -> 500. The conflict mapper now turns both paths
    into the same stable 409.
    """
    course_id = scenario["course_id"]
    suffix = uuid.uuid4().hex[:6]
    code_str = f"DUP-{suffix}"
    headers = {"Authorization": f"Bearer {manager_bearer}"}

    first = await client.post(
        f"/api/v1/management/courses/{course_id}/invitation-codes",
        json={"code": code_str, "max_uses": 1},
        headers=headers,
    )
    assert first.status_code == 201, first.text
    created_id = first.json()["id"]
    try:
        second = await client.post(
            f"/api/v1/management/courses/{course_id}/invitation-codes",
            json={"code": code_str, "max_uses": 1},
            headers=headers,
        )
        assert second.status_code == 409, second.text
        detail = second.json()["detail"]
        assert detail["error"] == "conflict"
        assert "invitation_code_taken" in detail["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM course_invitation_codes WHERE id = :id"),
                {"id": created_id},
            )


async def test_me_enrollments_returns_my_courses(
    client: httpx.AsyncClient,
    manager_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    scenario: dict[str, object],
) -> None:
    course_id = scenario["course_id"]
    enroll_resp = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(seeded_users.student_id)]},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert enroll_resp.status_code == 200

    me_resp = await client.get(
        "/api/v1/me/enrollments",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert me_resp.status_code == 200
    body = me_resp.json()
    assert any(item["course_id"] == str(course_id) for item in body)


@pytest.mark.parametrize("path", ["/api/v1/me/enrollments"])
async def test_me_enrollments_requires_auth(client: httpx.AsyncClient, path: str) -> None:
    response = await client.get(path)
    assert response.status_code == 401
