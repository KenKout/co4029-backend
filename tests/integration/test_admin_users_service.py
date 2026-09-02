"""Coverage for ``admin/services/users.py`` and ``admin/routers/users.py``.

``test_admin.py`` already covers a list smoke test and the disable/enable
happy path. The gaps this file fills are the ones with teeth:

* **Cursor pagination.** ``list_users`` orders by ``(created_at DESC, id
  DESC)`` and pages on both. The tiebreaker is the whole point: the seed
  inserts users inside one transaction, so a great many share a timestamp to
  the microsecond, and a cursor on ``created_at`` alone would either skip or
  repeat them. That is invisible on page one and only shows up in production.

* **Idempotence on disable.** Disabling an already-inactive user must still
  revoke sessions. Treating it as an error leaves a disabled account with a
  live session, which is the exact failure the operation exists to prevent.

* **The peer guard.** A manager may not disable another manager, and may not
  touch a user outside their organization -- reported as not-found so the
  existence of other tenants' accounts does not leak.

Isolation: the shared Postgres has other suites' users in it, so the listing
tests assert over a private cohort of their own rather than absolute counts.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
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

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.admin.routers import users_router
from abridgeai.features.admin.services import users as users_service


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
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
async def db(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
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


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> str:
    sid = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, NOW() + INTERVAL '1 hour')"
            ),
            {"id": sid, "uid": user_id, "h": hash_secret(generate_token())},
        )
    return create_access_token(user_id=user_id, session_id=sid)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, NOW() + INTERVAL '1 hour')"
            ),
            {"id": sid, "uid": user_id, "h": hash_secret(generate_token())},
        )
    return sid


class Cohort:
    """A private organization with its own users, so listings are exact."""

    def __init__(self) -> None:
        self.org_id = uuid.uuid4()
        self.user_ids: list[uuid.UUID] = []
        self.tag = uuid.uuid4().hex[:10]


@pytest_asyncio.fixture
async def cohort(engine: AsyncEngine) -> AsyncIterator[Cohort]:
    c = Cohort()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Users Coverage Org', 'active')"
            ),
            {"id": c.org_id, "slug": f"users-cov-{c.tag}"},
        )
        # Five users created in one statement so they share a created_at to
        # the microsecond -- the condition the composite cursor exists for.
        for i in range(5):
            uid = uuid.uuid4()
            c.user_ids.append(uid)
            await conn.execute(
                text(
                    "INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"
                ),
                {"id": uid, "email": f"cohort-{c.tag}-{i}@test.local"},
            )
            await conn.execute(
                text(
                    "INSERT INTO organization_memberships "
                    "(id, user_id, organization_id, status) "
                    "VALUES (gen_random_uuid(), :uid, :org, 'active')"
                ),
                {"uid": uid, "org": c.org_id},
            )
    yield c
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = ANY(:ids)"), {"ids": c.user_ids}
        )
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE organization_id = :org"),
            {"org": c.org_id},
        )
        # Profiles reference users; without this the users DELETE below hits
        # the FK and the whole teardown aborts, stranding the cohort and
        # erroring every later test in the file rather than just this one.
        await conn.execute(
            text("DELETE FROM user_profiles WHERE user_id = ANY(:ids)"), {"ids": c.user_ids}
        )
        await conn.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": c.user_ids})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": c.org_id})


# ---------------------------------------------------------------------------
# list_users -- cursor pagination
# ---------------------------------------------------------------------------


async def _page(db: AsyncSession, cohort: Cohort, *, limit: int, cursor: str | None = None):  # noqa: ANN202
    return await users_service.list_users(
        db,
        status_filter=None,
        role_code=None,
        organization_id=cohort.org_id,
        q=None,
        limit=limit,
        cursor=cursor,
    )


async def test_paging_visits_every_user_exactly_once(db: AsyncSession, cohort: Cohort) -> None:
    """The composite cursor must not skip or repeat rows on a timestamp tie.

    Five users share one ``created_at``. A cursor keyed on the timestamp alone
    would either re-serve the whole group on every page (an infinite list) or
    step past all five at once (four users that no admin can ever find).
    """
    seen: list[uuid.UUID] = []
    cursor: str | None = None
    for _ in range(10):  # generous bound; the loop breaks on exhaustion
        page = await _page(db, cohort, limit=2, cursor=cursor)
        seen.extend(row["user_id"] for row in page.items)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert cursor is None, "pagination did not terminate"
    assert sorted(seen, key=str) == sorted(cohort.user_ids, key=str)
    assert len(seen) == len(set(seen)), "a user was served on two pages"


async def test_next_cursor_is_none_on_a_partial_page(db: AsyncSession, cohort: Cohort) -> None:
    """A page that did not fill cannot have more behind it."""
    page = await _page(db, cohort, limit=50)
    assert len(page.items) == 5
    assert page.next_cursor is None


async def test_next_cursor_is_set_on_a_full_page(db: AsyncSession, cohort: Cohort) -> None:
    """A full page MAY have more behind it, so the cursor is offered.

    It is offered even when the next page turns out empty -- the query cannot
    know without looking, and an extra empty round trip is cheaper than
    truncating the list.
    """
    page = await _page(db, cohort, limit=5)
    assert len(page.items) == 5
    assert page.next_cursor is not None


async def test_a_malformed_cursor_is_rejected(db: AsyncSession, cohort: Cohort) -> None:
    """A hand-edited cursor must not silently return page one.

    Quietly ignoring it would make a corrupted token look like the end of the
    list rather than an error.
    """
    with pytest.raises(ValueError, match="cursor"):
        await _page(db, cohort, limit=2, cursor="not-a-real-cursor")


async def test_status_filter_narrows_the_list(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE users SET status = 'inactive' WHERE id = :id"),
            {"id": cohort.user_ids[0]},
        )

    active = await users_service.list_users(
        db,
        status_filter="active",
        role_code=None,
        organization_id=cohort.org_id,
        q=None,
        limit=50,
        cursor=None,
    )
    assert cohort.user_ids[0] not in {r["user_id"] for r in active.items}
    assert len(active.items) == 4


async def test_search_matches_on_email(db: AsyncSession, cohort: Cohort) -> None:
    page = await users_service.list_users(
        db,
        status_filter=None,
        role_code=None,
        organization_id=cohort.org_id,
        q=f"cohort-{cohort.tag}-3",
        limit=50,
        cursor=None,
    )
    assert [r["user_id"] for r in page.items] == [cohort.user_ids[3]]


async def test_org_filter_excludes_other_tenants(
    db: AsyncSession, cohort: Cohort, seeded_users: SeededUsers
) -> None:
    page = await _page(db, cohort, limit=50)
    listed = {r["user_id"] for r in page.items}
    assert seeded_users.admin_id not in listed
    assert listed == set(cohort.user_ids)


# ---------------------------------------------------------------------------
# user_detail
# ---------------------------------------------------------------------------


async def test_user_detail_assembles_the_four_sections(
    db: AsyncSession, seeded_users: SeededUsers
) -> None:
    detail = await users_service.user_detail(db, user_id=seeded_users.teacher_id)
    assert detail["user"]["id"] == seeded_users.teacher_id
    assert set(detail) == {"user", "role_assignments", "active_sessions", "role_history"}
    assert any(a["role_code"] == "teacher" for a in detail["role_assignments"])


async def test_user_detail_reports_no_profile_rather_than_a_hollow_one(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    """A user with no profile row must get ``profile: None``.

    The base query LEFT JOINs the profile, so the columns come back as a full
    set of nulls. Passing that through would give the UI an object whose every
    field is empty -- indistinguishable from a profile someone cleared.
    """
    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert detail["user"]["profile"] is None

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO user_profiles (user_id, display_name) VALUES (:uid, 'Named Person')"),
            {"uid": cohort.user_ids[0]},
        )
    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert detail["user"]["profile"]["display_name"] == "Named Person"


async def test_user_detail_of_a_missing_user_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await users_service.user_detail(db, user_id=uuid.uuid4())


async def test_user_detail_lists_only_live_sessions(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    live = await _seed_session(engine, cohort.user_ids[0])
    revoked = await _seed_session(engine, cohort.user_ids[0])
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE auth_sessions SET revoked_at = NOW() WHERE id = :id"),
            {"id": revoked},
        )

    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    ids = {s["id"] for s in detail["active_sessions"]}
    assert live in ids
    assert revoked not in ids


# ---------------------------------------------------------------------------
# revoke_session
# ---------------------------------------------------------------------------


async def test_revoke_session_marks_it_revoked(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    sid = await _seed_session(engine, cohort.user_ids[0])
    result = await users_service.revoke_session(db, user_id=cohort.user_ids[0], session_id=sid)
    assert result == {"session_id": sid, "revoked": True}

    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert sid not in {s["id"] for s in detail["active_sessions"]}


async def test_revoking_an_already_revoked_session_is_not_found(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    """Reported rather than silently succeeding, so a double-click is visible."""
    sid = await _seed_session(engine, cohort.user_ids[0])
    await users_service.revoke_session(db, user_id=cohort.user_ids[0], session_id=sid)
    with pytest.raises(NotFoundError):
        await users_service.revoke_session(db, user_id=cohort.user_ids[0], session_id=sid)


async def test_a_session_cannot_be_revoked_through_the_wrong_user(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    """The user id is part of the predicate, not just a URL segment.

    Without it, an admin allowed to manage user A could revoke user B's
    session by pasting B's session id into A's URL.
    """
    sid = await _seed_session(engine, cohort.user_ids[0])
    with pytest.raises(NotFoundError):
        await users_service.revoke_session(db, user_id=cohort.user_ids[1], session_id=sid)

    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert sid in {s["id"] for s in detail["active_sessions"]}, "session must survive"


# ---------------------------------------------------------------------------
# disable / enable
# ---------------------------------------------------------------------------


async def test_disable_sets_inactive_and_revokes_every_session(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    await _seed_session(engine, cohort.user_ids[0])
    await _seed_session(engine, cohort.user_ids[0])

    result = await users_service.disable_user(db, user_id=cohort.user_ids[0])
    assert result["status"] == "inactive"
    assert result["revoked_session_count"] == 2

    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert detail["active_sessions"] == []


async def test_disabling_an_already_inactive_user_still_revokes_sessions(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    """The dangerous case, and the reason this is not a plain no-op.

    A user disabled earlier who has since signed in again (or whose session
    survived a partial failure) is exactly who an operator is disabling a
    second time. Returning early on "already inactive" would leave that live
    session alone and report success.
    """
    await users_service.disable_user(db, user_id=cohort.user_ids[0])
    await _seed_session(engine, cohort.user_ids[0])

    result = await users_service.disable_user(db, user_id=cohort.user_ids[0])
    assert result["status"] == "inactive"
    assert result["revoked_session_count"] == 1

    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert detail["active_sessions"] == []


async def test_disable_of_a_missing_user_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await users_service.disable_user(db, user_id=uuid.uuid4())


async def test_enable_restores_active_but_leaves_sessions_revoked(
    db: AsyncSession, engine: AsyncEngine, cohort: Cohort
) -> None:
    """Re-enabling is not un-revoking. The user signs in again.

    Restoring the old sessions would hand back tokens that were deliberately
    killed, possibly to a device the account holder no longer controls.
    """
    await _seed_session(engine, cohort.user_ids[0])
    await users_service.disable_user(db, user_id=cohort.user_ids[0])

    result = await users_service.enable_user(db, user_id=cohort.user_ids[0])
    assert result["status"] == "active"

    detail = await users_service.user_detail(db, user_id=cohort.user_ids[0])
    assert detail["user"]["status"] == "active"
    assert detail["active_sessions"] == []


async def test_enable_of_a_missing_user_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await users_service.enable_user(db, user_id=uuid.uuid4())


# ---------------------------------------------------------------------------
# routers/users.py -- the guard matrix
# ---------------------------------------------------------------------------


async def test_an_admin_cannot_disable_their_own_account(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Self-lockout protection. Nobody is left able to re-enable them."""
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(
        f"/api/v1/admin/users/{seeded_users.admin_id}/disable", headers=_auth(token)
    )
    assert resp.status_code == 403, resp.text
    assert "own account" in resp.json()["detail"]["message"]


async def test_a_manager_cannot_disable_a_peer(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Managers police students and teachers, not each other.

    Otherwise two managers in one organization can disable each other, and the
    tenant's account administration becomes a race.
    """
    token = await _bearer(engine, seeded_users.manager_id)
    resp = await client.post(
        f"/api/v1/admin/users/{seeded_users.hod_id}/disable", headers=_auth(token)
    )
    assert resp.status_code == 403, resp.text
    assert "peer" in resp.json()["detail"]["message"]


async def test_a_manager_cannot_reach_a_user_in_another_organization(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    """Not-found, not forbidden: whether that account exists is not their business."""
    token = await _bearer(engine, seeded_users.manager_id)
    resp = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/disable", headers=_auth(token)
    )
    assert resp.status_code == 404, resp.text


async def test_an_it_admin_is_not_org_restricted(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    """``system.administer`` is a global operator role, by design."""
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/disable", headers=_auth(token)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "inactive"


async def test_revoke_session_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    sid = await _seed_session(engine, cohort.user_ids[0])

    resp = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/sessions/{sid}/revoke",
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"session_id": str(sid), "revoked": True}

    again = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/sessions/{sid}/revoke",
        headers=_auth(token),
    )
    assert again.status_code == 404, again.text


async def test_user_detail_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.get(f"/api/v1/admin/users/{seeded_users.teacher_id}", headers=_auth(token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["id"] == str(seeded_users.teacher_id)
    assert "role_assignments" in body


async def test_user_detail_of_a_missing_user_is_404_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.get(f"/api/v1/admin/users/{uuid.uuid4()}", headers=_auth(token))
    assert resp.status_code == 404, resp.text


async def test_listing_pages_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    """The cursor must survive the round trip through a query string."""
    token = await _bearer(engine, seeded_users.admin_id)
    first = await client.get(
        "/api/v1/admin/users",
        params={"limit": 2, "q": f"cohort-{cohort.tag}"},
        headers=_auth(token),
    )
    assert first.status_code == 200, first.text
    assert len(first.json()["items"]) == 2
    cursor = first.json()["next_cursor"]
    assert cursor

    second = await client.get(
        "/api/v1/admin/users",
        params={"limit": 2, "q": f"cohort-{cohort.tag}", "cursor": cursor},
        headers=_auth(token),
    )
    assert second.status_code == 200, second.text
    first_ids = {r["user_id"] for r in first.json()["items"]}
    second_ids = {r["user_id"] for r in second.json()["items"]}
    assert not (first_ids & second_ids), "pages overlapped"


@pytest.mark.parametrize("limit", [0, 201])
async def test_listing_rejects_an_out_of_range_limit(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, limit: int
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.get("/api/v1/admin/users", params={"limit": limit}, headers=_auth(token))
    assert resp.status_code == 422, resp.text


async def test_users_endpoints_reject_a_student(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.student_id)
    resp = await client.get("/api/v1/admin/users", headers=_auth(token))
    assert resp.status_code == 403, resp.text


async def test_a_malformed_cursor_is_400_not_500_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """A hand-edited or truncated cursor is a client error, not a crash.

    Cursors travel in URLs and get copied, trimmed and re-pasted. Letting the
    decode failure escape as a 500 turns a mangled bookmark into an incident.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.get(
        "/api/v1/admin/users",
        params={"cursor": "!!!not-base64!!!"},
        headers=_auth(token),
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"] == "invalid_cursor"


async def test_the_listing_reports_role_codes(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Roles are resolved for the whole page in one lookup, not per row.

    The join is what the operator scans the list by; a row that comes back
    with an empty ``role_codes`` looks like an account with no access at all.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.get(
        "/api/v1/admin/users",
        params={"q": "test-teacher@abridgeai.local"},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    row = next(r for r in resp.json()["items"] if r["user_id"] == str(seeded_users.teacher_id))
    assert "teacher" in row["role_codes"]


async def test_disable_of_a_missing_user_is_404_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(f"/api/v1/admin/users/{uuid.uuid4()}/disable", headers=_auth(token))
    assert resp.status_code == 404, resp.text


async def test_enable_of_a_missing_user_is_404_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(f"/api/v1/admin/users/{uuid.uuid4()}/enable", headers=_auth(token))
    assert resp.status_code == 404, resp.text


async def test_revoke_on_a_missing_session_is_404_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    resp = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/sessions/{uuid.uuid4()}/revoke",
        headers=_auth(token),
    )
    assert resp.status_code == 404, resp.text


async def test_enable_over_http_restores_a_disabled_account(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)

    disabled = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/disable", headers=headers
    )
    assert disabled.status_code == 200, disabled.text

    enabled = await client.post(f"/api/v1/admin/users/{cohort.user_ids[0]}/enable", headers=headers)
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["status"] == "active"


async def test_a_manager_may_disable_a_non_peer_in_their_own_organization(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The permissive half of the guard matrix.

    Asserted alongside the refusals because a guard that rejects everything
    passes every negative test while making the feature useless -- a manager
    must still be able to disable a student in their own tenant.
    """
    token = await _bearer(engine, seeded_users.manager_id)
    headers = _auth(token)
    try:
        resp = await client.post(
            f"/api/v1/admin/users/{seeded_users.student_id}/disable", headers=headers
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "inactive"
    finally:
        # The student is shared seed data; leave it as it was found.
        admin_token = await _bearer(engine, seeded_users.admin_id)
        await client.post(
            f"/api/v1/admin/users/{seeded_users.student_id}/enable",
            headers=_auth(admin_token),
        )


async def test_a_student_cannot_read_another_users_detail(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.student_id)
    resp = await client.get(f"/api/v1/admin/users/{seeded_users.teacher_id}", headers=_auth(token))
    assert resp.status_code == 403, resp.text


async def test_a_student_cannot_disable_anyone(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    token = await _bearer(engine, seeded_users.student_id)
    resp = await client.post(
        f"/api/v1/admin/users/{cohort.user_ids[0]}/disable", headers=_auth(token)
    )
    assert resp.status_code == 403, resp.text


async def test_the_status_filter_survives_the_query_string(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, cohort: Cohort
) -> None:
    """``status`` is aliased from ``user_status`` on the handler.

    An alias that drifts silently stops filtering -- the page still renders,
    with every user on it regardless of which tab is selected.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)
    await client.post(f"/api/v1/admin/users/{cohort.user_ids[0]}/disable", headers=headers)

    resp = await client.get(
        "/api/v1/admin/users",
        params={"status": "inactive", "q": f"cohort-{cohort.tag}"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    returned = {r["user_id"] for r in resp.json()["items"]}
    assert returned == {str(cohort.user_ids[0])}
