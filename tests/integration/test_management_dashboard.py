"""Integration tests for the manager / faculty-dean dashboard (Tier 1).

``GET /api/v1/management/dashboard`` is a decision queue: courses that cannot be
published, programs needing attention, and the counts derived from both.

Two of these tests are the reason the endpoint exists at all rather than reusing
the teacher dashboard:

* :func:`test_manager_who_authors_nothing_still_sees_the_org` — the teacher
  dashboard resolves scope to courses the caller OWNS or is ASSIGNED to teach. A
  manager holds ``course.create`` so they PASS that permission gate and receive
  ``200 OK`` with an EMPTY course set. It fails OPEN: a page that renders
  perfectly and reports nothing. This test fails if the wrong resolver is ever
  wired back in.
* :func:`test_dashboard_excludes_other_organizations` — FR-2.6 / UR-MGR-04. Every
  section must be tenant-scoped server-side.

The rest pin the contracts the SPA relies on: parity with the publish gate (so
the queue and the 409 cannot disagree), server-side worst-first ordering, and the
deliberate ``None``-vs-``0`` distinction on the dean-only review count.
"""

from __future__ import annotations

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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import (
    CurrentUser,
    create_access_token,
    generate_token,
    hash_secret,
)
from abridgeai.features.courses.routers import management_dashboard_router
from abridgeai.features.courses.services.assignment import get_course_readiness

_URL = "/api/v1/management/dashboard"

#: Module positions are parked far out of the way: the seeded course is shared
#: session-wide and other test files hard-code position 1 on it, colliding on
#: ``modules_course_id_position_key``. See the note in tests/support/db_graph.py.
_MODULE_POSITION = 9300


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
        "script_location", str(Path(__file__).resolve().parents[2] / "migrations")
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
async def app(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(management_dashboard_router, prefix="/api/v1")
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
                "exp": datetime.now(tz=UTC) + timedelta(hours=1),
            },
        )
    return session_id


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> AsyncIterator[str]:
    sid = await _seed_session(engine, user_id)
    yield create_access_token(user_id=user_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def manager_bearer(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    async for token in _bearer(engine, seeded_users.manager_id):
        yield token


@pytest_asyncio.fixture
async def hod_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    async for token in _bearer(engine, seeded_users.hod_id):
        yield token


@pytest_asyncio.fixture
async def student_bearer(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    async for token in _bearer(engine, seeded_users.student_id):
        yield token


@pytest_asyncio.fixture
async def teacher_bearer(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    async for token in _bearer(engine, seeded_users.teacher_id):
        yield token


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """One blockable course in the caller's org, plus a whole foreign org.

    ``blocked_course`` is a DRAFT with no gradeable unit and no learning
    outcome, so it fails the publish gate on two counts and must appear in the
    queue. ``foreign_course`` belongs to a different organization and must never
    appear for a caller scoped to the seeded one.

    ``faculty_id`` on the blocked course is NOT optional: the seeded manager's
    role assignment is ``scope_kind='org_unit'`` on the seeded faculty, so the
    dashboard resolves faculty scope and lists courses by faculty. A course with
    a NULL faculty is legitimately out of scope and would silently never reach
    the queue, making this fixture test nothing.
    """
    suffix = uuid.uuid4().hex[:8]
    blocked_course = uuid.uuid4()
    foreign_org = uuid.uuid4()
    foreign_owner = uuid.uuid4()
    foreign_course = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, faculty_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :faculty, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": blocked_course,
                "org": seeded_users.organization_id,
                "faculty": seeded_users.org_unit_id,
                "owner": seeded_users.teacher_id,
                "slug": f"blocked-{suffix}",
                "title": f"ZZ Blocked Course {suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": foreign_org,
                "slug": f"foreign-org-{suffix}",
                "name": f"Foreign Org {suffix}",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": foreign_owner, "email": f"foreign-{suffix}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": foreign_course,
                "org": foreign_org,
                "owner": foreign_owner,
                "slug": f"foreign-course-{suffix}",
                "title": f"ZZ Foreign Course {suffix}",
            },
        )

    yield {
        "blocked_course": blocked_course,
        "foreign_org": foreign_org,
        "foreign_course": foreign_course,
        "foreign_owner": foreign_owner,
    }

    async with engine.begin() as conn:
        for course_id in (blocked_course, foreign_course):
            await conn.execute(
                text("DELETE FROM course_learning_outcomes WHERE course_id = :c"),
                {"c": course_id},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": foreign_owner})
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": foreign_org})


# ---------------------------------------------------------------------------
# The two tests this endpoint exists for
# ---------------------------------------------------------------------------


async def test_manager_who_authors_nothing_still_sees_the_org(
    client: httpx.AsyncClient,
    manager_bearer: str,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """A manager authoring zero courses must still see their organization's.

    This is the whole point of the feature. The teacher dashboard scopes to
    courses the caller owns or teaches; a manager holds ``course.create`` so
    they clear its permission gate and get ``200 OK`` with an EMPTY set — a
    dashboard that looks healthy and says nothing. Asserting the manager
    authors nothing AND still sees courses is what makes that regression
    impossible to reintroduce silently.
    """
    from abridgeai.features.courses.services.authoring import (
        list_authoring_courses_for_user,
    )

    async with session_factory() as session:
        authored = await list_authoring_courses_for_user(
            session,
            user=CurrentUser(user_id=seeded_users.manager_id, session_id=uuid.uuid4()),
        )

    resp = await client.get(_URL, headers=_auth(manager_bearer))
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["counts"]["courses_total"] > 0, (
        "manager sees an empty dashboard — the teacher scope resolver is wired in"
    )
    assert len(authored) < body["counts"]["courses_total"], (
        f"authored={len(authored)} is not narrower than in-scope="
        f"{body['counts']['courses_total']}; scope may be authored-courses"
    )


async def test_dashboard_excludes_other_organizations(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """No section may leak another tenant's rows (FR-2.6 / UR-MGR-04)."""
    resp = await client.get(_URL, headers=_auth(manager_bearer))
    assert resp.status_code == 200, resp.text
    body = resp.json()

    foreign = str(scenario["foreign_course"])
    blocked_ids = {row["course_id"] for row in body["blocked_courses"]}
    assert foreign not in blocked_ids

    foreign_org = str(scenario["foreign_org"])
    for row in body["blocked_courses"]:
        assert row["organization_id"] != foreign_org
    for row in body["programs_needing_attention"]:
        assert row["organization_id"] != foreign_org


# ---------------------------------------------------------------------------
# Contracts the SPA depends on
# ---------------------------------------------------------------------------


async def test_blocked_queue_agrees_with_the_publish_gate(
    client: httpx.AsyncClient,
    manager_bearer: str,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """Queue membership must equal ``get_course_readiness.can_publish`` inverted.

    The readiness checklist documents ``can_publish`` as the exact conjunction
    ``publish_course`` gates on. A queue that disagrees is worse than no queue:
    the manager trusts the green tick and blames the button. Rather than
    hard-coding an expectation, this asks the authoritative function about every
    course the dashboard returned plus the fixture's blocked one.
    """
    resp = await client.get(_URL, headers=_auth(manager_bearer))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    queued = {row["course_id"] for row in body["blocked_courses"]}

    blocked_course = scenario["blocked_course"]
    assert str(blocked_course) in queued, (
        "a draft with no gradeable unit and no outcome is missing from the queue"
    )

    async with session_factory() as session:
        for course_id in [blocked_course, *[uuid.UUID(c) for c in queued]]:
            readiness = await get_course_readiness(session, course_id)
            in_queue = str(course_id) in queued
            assert readiness["can_publish"] is not in_queue, (
                f"course {course_id} can_publish={readiness['can_publish']} "
                f"but in_queue={in_queue}"
            )


async def test_blocked_rows_carry_a_readable_reason(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Every row explains itself in words, not only in flags.

    A severity that exists only as a boolean or a colour does not tell a manager
    WHICH of four gates failed, and is unreadable to a screen reader.
    """
    resp = await client.get(_URL, headers=_auth(manager_bearer))
    assert resp.status_code == 200, resp.text
    rows = resp.json()["blocked_courses"]
    assert rows, "fixture guarantees at least one blocked course"

    target = next(
        row for row in rows if row["course_id"] == str(scenario["blocked_course"])
    )
    assert target["reason"].strip(), "reason must never be empty"
    assert target["reason_codes"], "machine-readable codes must accompany the sentence"
    # The fixture course is a draft with neither content nor outcomes.
    assert "no_gradeable_content" in target["reason_codes"]
    assert "no_learning_outcomes" in target["reason_codes"]
    assert target["gradeable_unit_count"] == 0
    assert target["learning_outcome_count"] == 0


async def test_server_sorts_stage_blockers_first(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Ordering is a server contract; the client must never re-rank.

    A required course with no gradeable unit locks its stage and every stage
    behind it for every student on that path, so it outranks a merely
    unpublishable course. Asserted as a partition rather than an exact
    permutation: any ``blocks_required_stage`` row must precede every row
    without it.
    """
    resp = await client.get(_URL, headers=_auth(manager_bearer))
    assert resp.status_code == 200, resp.text
    flags = [bool(row["blocks_required_stage"]) for row in resp.json()["blocked_courses"]]
    assert flags == sorted(flags, reverse=True), (
        f"stage blockers are not sorted first: {flags}"
    )


async def test_counts_agree_with_the_lists_beneath_them(
    client: httpx.AsyncClient, manager_bearer: str, scenario: dict[str, uuid.UUID]
) -> None:
    """A tile that disagrees with its own table is a trust bug.

    ``courses_blocked`` is defined as the length of the blocked queue, and the
    whole payload is one snapshot precisely so the two cannot drift.
    """
    resp = await client.get(_URL, headers=_auth(manager_bearer))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["counts"]["courses_blocked"] == len(body["blocked_courses"])
    assert (
        body["counts"]["courses_draft"] + body["counts"]["courses_published"]
        <= body["counts"]["courses_total"]
    )


# ---------------------------------------------------------------------------
# Dean vs manager, and authorization
# ---------------------------------------------------------------------------


async def test_review_count_is_none_for_a_manager_and_a_number_for_a_dean(
    client: httpx.AsyncClient, manager_bearer: str, hod_bearer: str
) -> None:
    """``None`` and ``0`` are different claims and must not be conflated.

    Only a Faculty Dean holds ``learning_program.switch.review``. For a manager
    the count is ``None`` — "not your queue". ``0`` would assert "no work
    waiting", which for someone who cannot see the queue is a statement the
    server has no business making. Asserted with ``is None`` rather than a
    falsy check, because ``assert not x`` passes for both.
    """
    manager = (await client.get(_URL, headers=_auth(manager_bearer))).json()
    assert manager["can_review_path_changes"] is False
    assert manager["counts"]["open_path_change_requests"] is None

    dean = (await client.get(_URL, headers=_auth(hod_bearer))).json()
    assert dean["can_review_path_changes"] is True
    assert isinstance(dean["counts"]["open_path_change_requests"], int)


async def test_scope_is_echoed_back(client: httpx.AsyncClient, manager_bearer: str) -> None:
    """The page must be able to name what it is showing.

    A dashboard whose heading claims a wider or narrower set than its body is
    the failure this feature exists to fix, so the resolved scope travels with
    the data instead of being inferred from the caller's role.
    """
    body = (await client.get(_URL, headers=_auth(manager_bearer))).json()
    assert body["scope_kind"] in {"course", "org_unit", "organization", "global"}


async def test_unprivileged_callers_are_refused(
    client: httpx.AsyncClient, student_bearer: str, teacher_bearer: str
) -> None:
    """A student and a plain teacher hold no staffing permission -> 403."""
    for token in (student_bearer, teacher_bearer):
        resp = await client.get(_URL, headers=_auth(token))
        assert resp.status_code == 403, resp.text


async def test_unauthenticated_is_refused(client: httpx.AsyncClient) -> None:
    resp = await client.get(_URL)
    assert resp.status_code in {401, 403}, resp.text
