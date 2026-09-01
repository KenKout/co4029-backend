"""End-to-end identity-feature integration suite (T1.13).

Wires every identity router (``auth``, ``mfa``, ``me``, ``users``) plus the
access-control admin router into a single in-process FastAPI app and drives
real HTTPX requests through it against the docker postgres at port 5433.
Only the Google OAuth API is mocked (via ``respx``) -- everything else hits
the live test database (created from baseline + alembic upgrade head, with
catalog seed from migration 0004).

This file is the canonical Phase 1 OAuth + session + lookup test surface.
The TestClient setup below replicates what ``abridgeai/api.py`` (Phase 1
integration, post-T1.13) will do at app startup time.

Self-contained module-scoped fixtures (see ``scenario`` below) avoid
depending on the session-scoped ``seeded_users`` from ``tests/conftest.py``
because the destructive ``test_catalog_seed_migration`` round-trip can
invalidate that data when it runs in the same suite.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import jwt
import pytest
import pytest_asyncio
import respx
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
from abridgeai.core.security import (
    ALGORITHM,
    create_access_token,
    generate_token,
    hash_secret,
)
from abridgeai.features.access_control.routers.admin import router as admin_router
from abridgeai.features.identity.routers import (
    auth_router,
    me_router,
    mfa_router,
    users_router,
)

AUTH_ROUTER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "abridgeai"
    / "features"
    / "identity"
    / "routers"
    / "auth.py"
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


@dataclass(frozen=True)
class _Scenario:
    organization_id: uuid.UUID
    org_unit_id: uuid.UUID
    course_id: uuid.UUID
    student_id: uuid.UUID
    teacher_id: uuid.UUID
    admin_id: uuid.UUID


@pytest_asyncio.fixture(scope="module")
async def scenario(engine: AsyncEngine) -> AsyncIterator[_Scenario]:
    organization_id = uuid.uuid4()
    org_unit_id = uuid.uuid4()
    course_id = uuid.uuid4()
    student_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    admin_id = uuid.uuid4()

    role_assignments = (
        ("student", "organization", organization_id, None, None, student_id),
        ("teacher", "course", organization_id, None, course_id, teacher_id),
        ("admin", "global", None, None, None, admin_id),
    )

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": organization_id,
                "slug": f"t113-id-{organization_id.hex[:8]}",
                "name": "T1.13 Identity Suite Org",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": org_unit_id,
                "org": organization_id,
                "name": "T1.13 Identity Faculty",
                "code": f"T113-ID-{org_unit_id.hex[:6]}",
            },
        )
        for label, uid in (("student", student_id), ("teacher", teacher_id), ("admin", admin_id)):
            await conn.execute(
                text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
                {"id": uid, "em": f"t113-id-{label}-{uid.hex[:6]}@test.local"},
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
                "owner": admin_id,
                "slug": f"t113-id-course-{course_id.hex[:8]}",
                "title": "T1.13 Identity Course",
            },
        )
        for role_code, scope_kind, org_id, ou_id, c_id, uid in role_assignments:
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments "
                    "(user_id, role_id, scope_kind, organization_id, "
                    "org_unit_id, course_id, is_instructor) "
                    # 0093_teacher_title_flags: a COURSE-scoped assignment
                    # must carry at least one title
                    # (ck_user_role_assignments_course_title). Other scopes
                    # keep both flags false, which the CHECK allows.
                    "SELECT :uid, r.id, :scope_kind, :organization_id, "
                    ":org_unit_id, :course_id, :is_instructor "
                    "FROM roles r WHERE r.code = :role_code"
                ),
                {
                    "uid": uid,
                    "role_code": role_code,
                    "scope_kind": scope_kind,
                    "organization_id": org_id,
                    "org_unit_id": ou_id,
                    "course_id": c_id,
                    # Computed here rather than as `:scope_kind = 'course'`
                    # in SQL: reusing one bind param as both a value and a
                    # comparison leaves Postgres unable to deduce its type.
                    "is_instructor": scope_kind == "course",
                },
            )

    yield _Scenario(
        organization_id=organization_id,
        org_unit_id=org_unit_id,
        course_id=course_id,
        student_id=student_id,
        teacher_id=teacher_id,
        admin_id=admin_id,
    )

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
            {"ids": [student_id, teacher_id, admin_id]},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM org_units WHERE id = :id"), {"id": org_unit_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [student_id, teacher_id, admin_id]},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"),
            {"id": organization_id},
        )


@pytest_asyncio.fixture
async def app(engine: AsyncEngine) -> AsyncIterator[FastAPI]:
    """Phase 1 integration FastAPI app: every identity router + admin router.

    Mounts under ``/api/v1`` so paths match what ``abridgeai/api.py`` will
    use post-T1.13. Overrides ``get_db`` to bind to the module-scoped engine.
    """
    sm = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with sm() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(auth_router, prefix="/api/v1")
    fastapi_app.include_router(mfa_router, prefix="/api/v1")
    fastapi_app.include_router(me_router, prefix="/api/v1")
    fastapi_app.include_router(users_router, prefix="/api/v1")
    fastapi_app.include_router(admin_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def google_oauth_settings(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _mock_google(email: str, subject: str) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False)
    router.post("https://oauth2.googleapis.com/token").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "ya29.fake",
                "id_token": "fake.jwt",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )
    )
    router.get("https://openidconnect.googleapis.com/v1/userinfo").mock(
        return_value=httpx.Response(
            200,
            json={
                "sub": subject,
                "email": email,
                "given_name": "Oauth",
                "family_name": "Tester",
                "name": "Oauth Tester",
            },
        )
    )
    return router


async def _purge_user(engine: AsyncEngine, *, email: str, subject: str | None = None) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM auth_sessions WHERE user_id IN "
                "(SELECT id FROM users WHERE primary_email = :email)"
            ),
            {"email": email},
        )
        if subject is not None:
            await conn.execute(
                text("DELETE FROM auth_identities WHERE provider_subject = :sub"),
                {"sub": subject},
            )
        await conn.execute(
            text(
                "DELETE FROM user_profiles WHERE user_id IN "
                "(SELECT id FROM users WHERE primary_email = :email)"
            ),
            {"email": email},
        )
        await conn.execute(
            text("DELETE FROM users WHERE primary_email = :email"),
            {"email": email},
        )


async def _open_session(engine: AsyncEngine, user_id: uuid.UUID) -> tuple[uuid.UUID, str]:
    """Insert a real ``auth_sessions`` row and mint a matching bearer token.

    ``get_current_user`` joins ``users`` ⨝ ``auth_sessions`` so any test that
    needs a working bearer flow must seed a live session row first.
    """
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


async def test_oauth_callback_rejects_unprovisioned_email_full_flow(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    google_oauth_settings: None,
) -> None:
    """Invite-only OAuth: brand-new email returns 403, no User row created."""
    fresh_email = f"oauth-full-{uuid.uuid4().hex[:8]}@abridgeai.local"
    google_subject = f"google-uid-{uuid.uuid4().hex[:12]}"

    try:
        with _mock_google(fresh_email, google_subject):
            response = await client.get(
                "/api/v1/auth/google/callback",
                params={"code": "fakeCode"},
            )

        assert response.status_code == 403, response.text
        body = response.json()
        assert body["detail"]["error"] == "oauth_account_not_provisioned"

        async with engine.begin() as conn:
            user_row = (
                await conn.execute(
                    text("SELECT id FROM users WHERE primary_email = :email"),
                    {"email": fresh_email},
                )
            ).one_or_none()
            assert user_row is None, "Unprovisioned email must NOT create a users row"
    finally:
        await _purge_user(engine, email=fresh_email, subject=google_subject)


async def test_oauth_callback_links_identity_for_preprovisioned_user(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    google_oauth_settings: None,
) -> None:
    """Pre-provisioned user without identity: link AuthIdentity + UserProfile, issue tokens."""
    fresh_email = f"oauth-pre-{uuid.uuid4().hex[:8]}@abridgeai.local"
    google_subject = f"google-uid-{uuid.uuid4().hex[:12]}"
    pre_user_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": pre_user_id, "em": fresh_email},
        )

    try:
        with _mock_google(fresh_email, google_subject):
            response = await client.get(
                "/api/v1/auth/google/callback",
                params={"code": "fakeCode"},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["user"]["id"] == str(pre_user_id)

        decoded = jwt.decode(
            body["access_token"],
            get_settings().jwt_secret_key,
            algorithms=[ALGORITHM],
        )
        assert "sub" in decoded
        assert "sid" in decoded

        async with engine.begin() as conn:
            profile_row = (
                await conn.execute(
                    text("SELECT display_name FROM user_profiles WHERE user_id = :uid"),
                    {"uid": pre_user_id},
                )
            ).one_or_none()
            assert profile_row is not None, "UserProfile was not created on first OAuth login"

            identity_row = (
                await conn.execute(
                    text(
                        "SELECT id FROM auth_identities "
                        "WHERE provider = 'google' AND provider_subject = :sub"
                    ),
                    {"sub": google_subject},
                )
            ).one_or_none()
            assert identity_row is not None, "AuthIdentity was not linked"
    finally:
        await _purge_user(engine, email=fresh_email, subject=google_subject)


async def test_oauth_callback_existing_user_returns_token(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    google_oauth_settings: None,
) -> None:
    existing_email = f"oauth-existing-{uuid.uuid4().hex[:8]}@abridgeai.local"
    google_subject = f"google-uid-{uuid.uuid4().hex[:12]}"
    pre_user_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": pre_user_id, "em": existing_email},
        )

    try:
        with _mock_google(existing_email, google_subject):
            response = await client.get(
                "/api/v1/auth/google/callback",
                params={"code": "fakeCode"},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["user"]["id"] == str(pre_user_id), "Existing User row must be reused"

        async with engine.begin() as conn:
            count = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM users WHERE primary_email = :em"),
                    {"em": existing_email},
                )
            ).scalar_one()
            assert count == 1, "OAuth callback must not duplicate User rows"
    finally:
        await _purge_user(engine, email=existing_email, subject=google_subject)


async def test_full_login_then_me_then_logout(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    google_oauth_settings: None,
) -> None:
    email = f"lifecycle-{uuid.uuid4().hex[:8]}@abridgeai.local"
    subject = f"google-uid-{uuid.uuid4().hex[:12]}"
    pre_user_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": pre_user_id, "em": email},
        )

    try:
        with _mock_google(email, subject):
            login = await client.get(
                "/api/v1/auth/google/callback",
                params={"code": "fakeCode"},
            )
        assert login.status_code == 200, login.text
        access_token = login.json()["access_token"]

        me_response = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_response.status_code == 200, me_response.text
        assert me_response.json()["primary_email"] == email

        logout_response = await client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert logout_response.status_code == 204, logout_response.text

        post_logout = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert post_logout.status_code == 401, post_logout.text
    finally:
        await _purge_user(engine, email=email, subject=subject)


async def test_refresh_token_rotation(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    google_oauth_settings: None,
) -> None:
    email = f"refresh-{uuid.uuid4().hex[:8]}@abridgeai.local"
    subject = f"google-uid-{uuid.uuid4().hex[:12]}"
    pre_user_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :em, 'active')"),
            {"id": pre_user_id, "em": email},
        )

    try:
        with _mock_google(email, subject):
            login = await client.get(
                "/api/v1/auth/google/callback",
                params={"code": "fakeCode"},
            )
        first_refresh = login.json()["refresh_token"]

        rotation = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": first_refresh},
        )
        assert rotation.status_code == 200, rotation.text
        rotated_refresh = rotation.json()["refresh_token"]
        assert rotated_refresh != first_refresh, "Refresh token must rotate on use"

        replay = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": first_refresh},
        )
        assert replay.status_code == 401, replay.text

        followup = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": rotated_refresh},
        )
        assert followup.status_code == 200, followup.text
    finally:
        await _purge_user(engine, email=email, subject=subject)


async def test_get_users_me_permissions_returns_role_permissions(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: _Scenario,
) -> None:
    session_id, token = await _open_session(engine, scenario.student_id)
    try:
        response = await client.get(
            "/api/v1/users/me/permissions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        perms = set(response.json()["permissions"])

        assert {"course.read", "quiz.take", "interview.take", "progress.read.self"} <= perms, (
            f"Student token must resolve to T1.3 catalog permission set; got {sorted(perms)}"
        )
    finally:
        await _close_session(engine, session_id)


async def test_admin_user_lookup_via_users_id(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: _Scenario,
) -> None:
    session_id, token = await _open_session(engine, scenario.admin_id)
    try:
        response = await client.get(
            f"/api/v1/users/{scenario.teacher_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == str(scenario.teacher_id)
        assert "password" not in body
        assert "password_hash" not in body
    finally:
        await _close_session(engine, session_id)


async def test_password_endpoints_do_not_exist(client: httpx.AsyncClient) -> None:
    for path in (
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/forgot-password",
        "/api/v1/auth/reset-password",
    ):
        get_response = await client.get(path)
        assert get_response.status_code == 404, (
            f"GET {path} should not exist; got {get_response.status_code}"
        )
        post_response = await client.post(path, json={})
        assert post_response.status_code == 404, (
            f"POST {path} should not exist; got {post_response.status_code}"
        )


def test_auth_router_source_has_no_password_tokens() -> None:
    source = AUTH_ROUTER_PATH.read_text()
    code_only = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    code_only = re.sub(r"#.*", "", code_only)
    forbidden = re.compile(
        r"\b(register|forgot[_-]?password|password[_-]?hash|password[_-]?reset)\b",
        re.IGNORECASE,
    )
    matches = forbidden.findall(code_only)
    assert matches == [], f"Forbidden password-auth tokens in auth router: {matches}"
    assert "_DEV_PERMISSION" not in code_only
