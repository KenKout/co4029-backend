from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest_asyncio
from conftest import SeededUsers
from fastapi import FastAPI
from fastapi.routing import APIRoute
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
from abridgeai.features.identity.routers.users import router as users_router


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def test_engine_local() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def app(test_engine_local: AsyncEngine) -> AsyncIterator[FastAPI]:
    sm = async_sessionmaker(test_engine_local, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with sm() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(users_router, prefix="/api/v1")
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


@pytest_asyncio.fixture
async def admin_auth(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    session_id, token = await _open_session(test_engine_local, seeded_users.admin_id)
    try:
        yield session_id, token
    finally:
        await _close_session(test_engine_local, session_id)


@pytest_asyncio.fixture
async def student_auth(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    session_id, token = await _open_session(test_engine_local, seeded_users.student_id)
    try:
        yield session_id, token
    finally:
        await _close_session(test_engine_local, session_id)


@pytest_asyncio.fixture
async def manager_auth(
    test_engine_local: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, str]]:
    session_id, token = await _open_session(test_engine_local, seeded_users.manager_id)
    try:
        yield session_id, token
    finally:
        await _close_session(test_engine_local, session_id)


def test_router_metadata() -> None:
    assert users_router.prefix == "/users"
    assert "users" in users_router.tags
    assert "admin" in users_router.tags
    paths = {(r.path, tuple(sorted(r.methods))) for r in users_router.routes}  # type: ignore[attr-defined]
    assert ("/users", ("GET",)) in paths
    assert ("/users/{user_id}", ("GET",)) in paths


def test_every_endpoint_has_permission_dependency() -> None:
    """FIX-CRIT-4: every route in the admin /users router must have a require_* dep."""
    offending: list[str] = []
    for route in users_router.routes:
        if not isinstance(route, APIRoute):
            continue
        names = _collect_dep_names(route)
        if not any(n.startswith("require_") or n == "dependency" for n in names):
            offending.append(f"{route.path} -> {names}")
    assert offending == [], f"Routes missing permission dependency: {offending}"


def _collect_dep_names(route: APIRoute) -> list[str]:
    names: list[str] = []
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is not None:
            names.append(getattr(dep.call, "__name__", repr(dep.call)))
        stack.extend(dep.dependencies)
    return names


async def test_get_users_without_token_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/users")
    assert response.status_code == 401


async def test_get_users_with_student_token_returns_403(
    client: httpx.AsyncClient,
    student_auth: tuple[uuid.UUID, str],
) -> None:
    _, token = student_auth
    response = await client.get(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "permission_denied"
    assert "user.read" in detail["required"]


async def test_get_user_by_id_with_student_token_returns_403(
    client: httpx.AsyncClient,
    student_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    _, token = student_auth
    response = await client.get(
        f"/api/v1/users/{seeded_users.student_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403, response.text


async def test_get_users_with_admin_token_returns_200(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    _, token = admin_auth
    response = await client.get(
        "/api/v1/users?limit=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "items" in body
    assert "next_cursor" in body
    assert isinstance(body["items"], list)
    assert len(body["items"]) >= 5
    for item in body["items"]:
        assert "primary_email" in item
        assert "password" not in item


async def test_get_user_by_id_with_admin_token_returns_200(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    _, token = admin_auth
    response = await client.get(
        f"/api/v1/users/{seeded_users.teacher_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(seeded_users.teacher_id)
    assert "password" not in body


async def test_get_user_by_id_404_for_unknown(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    _, token = admin_auth
    response = await client.get(
        f"/api/v1/users/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


async def test_get_users_by_ids_batch(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """Batch lookup resolves a comma-separated id list; missing ids skipped."""
    _, token = admin_auth
    ids = f"{seeded_users.teacher_id},{seeded_users.hod_id},{uuid.uuid4()}"
    response = await client.get(
        f"/api/v1/users/by-ids?ids={ids}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    found = {u["id"] for u in body}
    assert str(seeded_users.teacher_id) in found
    assert str(seeded_users.hod_id) in found
    # The unknown uuid is simply absent (no 404 for the batch).
    assert len(body) == 2


async def test_get_users_by_ids_rejects_garbage(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    """Non-UUID segments 422 instead of crashing."""
    _, token = admin_auth
    response = await client.get(
        "/api/v1/users/by-ids?ids=not-a-uuid,also-bad",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


async def test_users_pagination_cursor_works(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    _, token = admin_auth
    headers = {"Authorization": f"Bearer {token}"}

    page_one = (await client.get("/api/v1/users?limit=2", headers=headers)).json()
    assert len(page_one["items"]) == 2
    assert page_one["next_cursor"] is not None

    page_two = (
        await client.get(
            f"/api/v1/users?limit=2&cursor={page_one['next_cursor']}",
            headers=headers,
        )
    ).json()
    assert len(page_two["items"]) >= 1

    ids_one = {item["id"] for item in page_one["items"]}
    ids_two = {item["id"] for item in page_two["items"]}
    assert ids_one.isdisjoint(ids_two), "pages must not overlap"


async def test_invalid_cursor_returns_422(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    _, token = admin_auth
    response = await client.get(
        "/api/v1/users?cursor=not-a-valid-cursor",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


async def test_search_users_includes_role_codes(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """The search payload carries each user's active role codes (Role column)."""
    _, token = admin_auth
    response = await client.get(
        "/api/v1/users/search?page_size=200",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    items = {item["id"]: item for item in response.json()["items"]}

    teacher = items.get(str(seeded_users.teacher_id))
    assert teacher is not None
    assert "roles" in teacher
    assert "teacher" in teacher["roles"]


async def test_search_users_role_filter_narrows_results(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """``?role=teacher`` returns only teachers; the seeded student is excluded."""
    _, token = admin_auth
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.get(
        "/api/v1/users/search?role=teacher&page_size=200", headers=headers
    )
    assert response.status_code == 200, response.text
    ids = {item["id"] for item in response.json()["items"]}
    assert str(seeded_users.teacher_id) in ids
    assert str(seeded_users.student_id) not in ids
    # Every returned user actually holds the teacher role.
    for item in response.json()["items"]:
        assert "teacher" in item["roles"]


async def test_search_users_unknown_role_returns_empty(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    """A role code nobody holds yields an empty page, not an error."""
    _, token = admin_auth
    response = await client.get(
        "/api/v1/users/search?role=nonexistent_role_xyz",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


async def test_search_users_includes_primary_organization(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """The search payload carries each user's primary org id + name."""
    _, token = admin_auth
    response = await client.get(
        "/api/v1/users/search?page_size=200",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    items = {item["id"]: item for item in response.json()["items"]}
    teacher = items.get(str(seeded_users.teacher_id))
    assert teacher is not None
    # Field is present (may be null for users with no membership).
    assert "organization_id" in teacher
    assert "organization_name" in teacher


async def test_search_users_org_filter_narrows_results(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """``?organization=<seeded org>`` returns only members of that org."""
    _, token = admin_auth
    headers = {"Authorization": f"Bearer {token}"}
    org_id = seeded_users.organization_id

    response = await client.get(
        f"/api/v1/users/search?organization={org_id}&page_size=200",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    # The seeded teacher belongs to the test org; every returned row reports it.
    ids = {item["id"] for item in response.json()["items"]}
    assert str(seeded_users.teacher_id) in ids
    for item in response.json()["items"]:
        assert item["organization_id"] == str(org_id)


async def test_search_users_role_and_org_filter_intersect(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """role + organization filters intersect (teacher in the seeded org)."""
    _, token = admin_auth
    headers = {"Authorization": f"Bearer {token}"}
    org_id = seeded_users.organization_id
    response = await client.get(
        f"/api/v1/users/search?role=teacher&organization={org_id}&page_size=200",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    for item in response.json()["items"]:
        assert "teacher" in item["roles"]
        assert item["organization_id"] == str(org_id)


async def test_create_user_as_admin_201(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """Admin invite: creates active user + profile + org membership + role."""
    _, token = admin_auth
    email = f"invite-{uuid.uuid4().hex[:8]}@test.local"
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": email,
            "given_name": "Invited",
            "family_name": "User",
            "display_name": "Invited User",
            "organization_id": str(seeded_users.organization_id),
            "role_code": "student",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["primary_email"] == email
    assert body["status"] == "active"
    assert body["profile"]["display_name"] == "Invited User"

    # The new user shows up in search with the invited role + org.
    search = await client.get(
        f"/api/v1/users/search?search={email}&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert search.status_code == 200, search.text
    items = search.json()["items"]
    assert len(items) == 1
    assert "student" in items[0]["roles"]
    assert items[0]["organization_id"] == str(seeded_users.organization_id)


async def test_create_user_with_manager_role_201(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """Admin may invite with a non-student role (manager) in one shot."""
    _, token = admin_auth
    email = f"invite-mgr-{uuid.uuid4().hex[:8]}@test.local"
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": email,
            "display_name": "Manager Invite",
            "organization_id": str(seeded_users.organization_id),
            "role_code": "manager",
        },
    )
    assert response.status_code == 201, response.text
    search = await client.get(
        f"/api/v1/users/search?search={email}&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert search.status_code == 200, search.text
    items = search.json()["items"]
    assert len(items) == 1
    assert "manager" in items[0]["roles"]


async def test_create_user_duplicate_email_409(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """Inviting an email that already exists conflicts (409)."""
    _, token = admin_auth
    headers = {"Authorization": f"Bearer {token}"}
    email = f"duplicate-{uuid.uuid4().hex[:8]}@test.local"
    payload = {
        "primary_email": email,
        "display_name": "Dup",
        "organization_id": str(seeded_users.organization_id),
    }
    first = await client.post("/api/v1/users", headers=headers, json=payload)
    assert first.status_code == 201, first.text
    second = await client.post("/api/v1/users", headers=headers, json=payload)
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["error"] == "conflict"


async def test_create_user_student_token_403(
    client: httpx.AsyncClient,
    student_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """Inviting is admin-only — a student is rejected (403)."""
    _, token = student_auth
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": f"nope-{uuid.uuid4().hex[:8]}@test.local",
            "display_name": "Nope",
            "organization_id": str(seeded_users.organization_id),
        },
    )
    assert response.status_code == 403, response.text


async def test_create_user_unknown_role_404(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """A role code that does not exist is rejected cleanly — not a 500.

    Regression: ``grant_org_role_access`` used ``scalar_one()`` so an unknown
    ``role_code`` surfaced as ``sqlalchemy.exc.NoResultFound`` -> HTTP 500.
    """
    _, token = admin_auth
    email = f"invite-unknown-role-{uuid.uuid4().hex[:8]}@test.local"
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": email,
            "display_name": "Unknown Role",
            "organization_id": str(seeded_users.organization_id),
            "role_code": "astronaut",
        },
    )
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["error"] == "role_not_found"

    # The failed invite must not leave a half-created user behind.
    search = await client.get(
        f"/api/v1/users/search?search={email}&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert search.status_code == 200, search.text
    assert search.json()["items"] == []


async def test_create_user_as_manager_201(
    client: httpx.AsyncClient,
    manager_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """A manager may invite a teacher into their OWN org.

    The payload's ``organization_id`` is ignored and replaced with the
    caller's primary org (same forcing precedent as GET /users/search), so
    even a foreign org id in the body cannot leak a user into another
    tenant.
    """
    _, token = manager_auth
    email = f"invite-mgr-teacher-{uuid.uuid4().hex[:8]}@test.local"
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": email,
            "display_name": "Manager-Invited Teacher",
            # Deliberately a foreign org: must be ignored, not honored.
            "organization_id": str(uuid.uuid4()),
            "role_code": "teacher",
        },
    )
    assert response.status_code == 201, response.text

    search = await client.get(
        f"/api/v1/users/search?search={email}&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert search.status_code == 200, search.text
    items = search.json()["items"]
    assert len(items) == 1
    assert "teacher" in items[0]["roles"]
    # Landed in the manager's org, NOT the foreign id from the payload.
    assert items[0]["organization_id"] == str(seeded_users.organization_id)


async def test_create_user_as_manager_without_org_201(
    client: httpx.AsyncClient,
    manager_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """A manager invite may omit organization_id entirely (schema-optional;
    the router still forces the caller's org)."""
    _, token = manager_auth
    email = f"invite-mgr-no-org-{uuid.uuid4().hex[:8]}@test.local"
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": email,
            "display_name": "No Org Field",
            "role_code": "student",
        },
    )
    assert response.status_code == 201, response.text

    search = await client.get(
        f"/api/v1/users/search?search={email}&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert search.status_code == 200, search.text
    items = search.json()["items"]
    assert len(items) == 1
    assert items[0]["organization_id"] == str(seeded_users.organization_id)


async def test_create_user_manager_peer_role_403(
    client: httpx.AsyncClient,
    manager_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """A manager cannot invite peer roles (hod / manager) — same peer guard
    as disable/enable; only student and teacher are creatable."""
    _, token = manager_auth
    for role_code in ("hod", "manager"):
        email = f"invite-mgr-peer-{role_code}-{uuid.uuid4().hex[:8]}@test.local"
        response = await client.post(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "primary_email": email,
                "organization_id": str(seeded_users.organization_id),
                "role_code": role_code,
            },
        )
        assert response.status_code == 403, (role_code, response.text)
        assert response.json()["detail"]["error"] == "forbidden"


async def test_create_user_admin_without_org_422(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
) -> None:
    """For a platform admin the org is still mandatory: the schema is
    optional only so managers can omit it."""
    _, token = admin_auth
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": f"admin-no-org-{uuid.uuid4().hex[:8]}@test.local",
            "role_code": "student",
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"] == "organization_required"


async def test_create_user_admin_role_org_scope_403(
    client: httpx.AsyncClient,
    admin_auth: tuple[uuid.UUID, str],
    seeded_users: SeededUsers,
) -> None:
    """The ``admin`` role (permissions: ALL) is global-only and cannot be
    granted through the org-scoped invite path.

    Permission resolution ignores assignment scope, so an org-scoped admin
    assignment would mint a full platform admin — reject it here.
    """
    _, token = admin_auth
    email = f"invite-org-admin-{uuid.uuid4().hex[:8]}@test.local"
    response = await client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "primary_email": email,
            "display_name": "Org Admin",
            "organization_id": str(seeded_users.organization_id),
            "role_code": "admin",
        },
    )
    assert response.status_code == 403, response.text

    search = await client.get(
        f"/api/v1/users/search?search={email}&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert search.status_code == 200, search.text
    assert search.json()["items"] == []
