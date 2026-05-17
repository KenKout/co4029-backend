"""Phase 3 characterization parity tests (T3.10).

Two execution modes:

* **Live legacy** -- when ``backend/`` is reachable at
  ``OLD_BACKEND_URL`` (default ``http://localhost:8000``) the tests
  capture snapshots from the legacy endpoints and replay them against
  the new ASGI app. Skipped automatically when legacy is unreachable
  (the standard local-dev case; legacy lives in a separate process).
* **Documented-divergence mode** -- always runs. The captured snapshot
  is synthesized in-memory to encode the legacy shape (audit columns
  absent, drafts leaked, auth permissive). The replay against the new
  app exercises the parity contracts at
  ``tests/parity/courses/`` and asserts that:

    1. ``GET /courses`` -- new response is a tolerated SUPERSET of the
       legacy shape (audit fields are NEW columns; ``allow_extra_fields``
       absorbs them so parity passes).
    2. ``GET /courses/{id}/content`` -- new response is a STRICT SUBSET
       of the legacy module list (DRAFT_VISIBILITY fix). The harness's
       ``behavior.row_count`` would otherwise FAIL parity; we assert
       the divergence directly via ``ReplayResult.divergences``.
    3. ``GET /teacher/courses/{id}`` -- new returns 403 where legacy
       returned 200 (FIX-SEC-1). Status mismatch is the divergence;
       the test asserts the direction (legacy permissive -> new strict).

The contracts at ``tests/parity/courses/*.yaml`` carry ``behavior.note``
fields documenting each accepted divergence so future readers see WHY
the contract tolerates the gap.
"""

from __future__ import annotations

import socket
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
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import (
    administration_router,
    assignment_router,
    authoring_router,
    learner_router,
    me_courses_router,
)
from characterization.contract import load_contract
from characterization.harness import Snapshot
from characterization.replay import compare

PARITY_DIR = Path(__file__).resolve().parent.parent / "parity" / "courses"


import abridgeai.features.interviews.models  # noqa: E402, F401  -- T6.1 registers interview_* tables

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


def _legacy_backend_running(host: str = "localhost", port: int = 8000) -> bool:
    """Probe the legacy backend with a 1s TCP connect; return False if unreachable."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


legacy_skip = pytest.mark.skipif(
    not _legacy_backend_running(),
    reason="legacy backend/ not running on localhost:8000 -- skip live parity capture",
)


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
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.include_router(me_courses_router, prefix="/api/v1")
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.include_router(assignment_router, prefix="/api/v1")
    fastapi_app.include_router(administration_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": sid,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return sid


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def published_course(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID | str]]:
    """One published course + one published module + one DRAFT module under it.

    The published module is the parity baseline. The draft module is the
    DRAFT_VISIBILITY witness: legacy backend exposed it on the content
    endpoint; the new endpoint excludes it.
    """
    suffix = uuid.uuid4().hex[:8]
    course_id = uuid.uuid4()
    pub_module_id = uuid.uuid4()
    draft_module_id = uuid.uuid4()
    slug = f"parity-course-{suffix}"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Parity Course', 'published')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.teacher_id,
                "slug": slug,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) VALUES "
                "(:p, :c, 'Pub', 1, 'published'), "
                "(:d, :c, 'Draft', 2, 'draft')"
            ),
            {"p": pub_module_id, "d": draft_module_id, "c": course_id},
        )
    data: dict[str, uuid.UUID | str] = {
        "course_id": course_id,
        "pub_module_id": pub_module_id,
        "draft_module_id": draft_module_id,
        "slug": slug,
    }
    yield data
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM modules WHERE course_id = :id"),
            {"id": course_id},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})


async def test_published_courses_parity_with_audit_extras(
    client: httpx.AsyncClient,
    student_bearer: str,
    published_course: dict[str, uuid.UUID | str],
) -> None:
    """``GET /courses`` -- new response carries audit columns the legacy lacked.

    Reconciliation §A13 -- ``created_by`` / ``updated_by`` / ``organization_id``
    / ``deleted_by`` / ``instructor`` / ``outcomes`` / ``tags`` are NEW on the
    public payload. The parity contract for ``api_v1_courses`` lists them
    under ``allow_extra_fields`` so the new response stays parity-OK.
    """
    response = await client.get(
        "/api/v1/courses?limit=100",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    new_body = response.json()

    course_id = str(published_course["course_id"])
    legacy_items = [
        {
            "id": item["id"],
            "slug": item["slug"],
            "title": item["title"],
            "status": item["status"],
            "description": item.get("description"),
        }
        for item in new_body["items"]
        if item["id"] == course_id
    ]
    assert legacy_items, "seeded course must surface in the listing"
    legacy_body = {"items": legacy_items}

    captured = Snapshot(
        backend="old",
        endpoint="/api/v1/courses",
        method="GET",
        status_code=200,
        headers={},
        body_json=legacy_body,
    )
    new_filtered = {
        "items": [item for item in new_body["items"] if item["id"] == course_id],
    }
    replayed = Snapshot(
        backend="new",
        endpoint="/api/v1/courses",
        method="GET",
        status_code=200,
        headers={},
        body_json=new_filtered,
    )
    contract = load_contract("/api/v1/courses", parity_dir=PARITY_DIR)
    result = compare(captured, replayed, contract)
    assert result.passed, (
        "audit-field extras must be tolerated by allow_extra_fields; "
        f"divergences={result.divergences}"
    )


async def test_content_tree_parity_documents_visibility_fix(
    client: httpx.AsyncClient,
    student_bearer: str,
    published_course: dict[str, uuid.UUID | str],
) -> None:
    """DRAFT_VISIBILITY: parity expectedly DIVERGES -- new excludes drafts.

    Pre-T3.7 the legacy backend leaked draft modules and module_items
    pointing to draft lessons through this endpoint. The new behaviour
    excludes them, so a row-count parity check FAILS by design. The
    test asserts:

    * The legacy-shaped captured snapshot has more modules than the
      replayed one (the divergence direction is correct).
    * Every module surfaced in the new response is also in the legacy
      response (new is a STRICT SUBSET, never a superset).

    See ``tests/parity/courses/api_v1_courses_{course_id}_content.yaml``
    behavior.note for the documented divergence.
    """
    course_id = published_course["course_id"]
    response = await client.get(
        f"/api/v1/courses/{course_id}/content",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    new_body = response.json()
    new_module_ids = {m["id"] for m in new_body.get("modules", [])}
    pub_module_id = str(published_course["pub_module_id"])
    draft_module_id = str(published_course["draft_module_id"])
    assert pub_module_id in new_module_ids
    assert draft_module_id not in new_module_ids, (
        "DRAFT_VISIBILITY fix: new content tree must exclude draft modules"
    )

    legacy_modules = [
        *new_body.get("modules", []),
        {"id": draft_module_id, "course_id": str(course_id), "title": "Draft", "position": 2},
    ]
    legacy_body = {**new_body, "modules": legacy_modules}

    captured = Snapshot(
        backend="old",
        endpoint=f"/api/v1/courses/{course_id}/content",
        method="GET",
        status_code=200,
        headers={},
        body_json=legacy_body,
    )
    replayed = Snapshot(
        backend="new",
        endpoint=f"/api/v1/courses/{course_id}/content",
        method="GET",
        status_code=200,
        headers={},
        body_json=new_body,
    )
    contract = load_contract(f"/api/v1/courses/{course_id}/content", parity_dir=PARITY_DIR)
    result = compare(captured, replayed, contract)
    assert not result.passed, (
        "row_count divergence (drafts pruned) must be flagged; if this assertion "
        "passes the contract is too permissive"
    )
    assert any("row_count" in d or "body diverges" in d for d in result.divergences), (
        f"expected row_count divergence, got {result.divergences}"
    )

    legacy_module_ids = {m["id"] for m in legacy_body["modules"]}
    assert new_module_ids.issubset(legacy_module_ids), (
        "new content tree must be a SUBSET of the legacy tree (never a superset)"
    )
    assert len(new_module_ids) < len(legacy_module_ids), (
        "draft pruning must shrink the module list, not extend it"
    )


async def test_teacher_courses_parity_documents_auth_tightening(
    client: httpx.AsyncClient,
    student_bearer: str,
    published_course: dict[str, uuid.UUID | str],
) -> None:
    """FIX-SEC-1: parity expectedly DIVERGES on status code -- new returns 403.

    Pre-T3.7 the legacy ``backend/app/routes/teacher/courses_router.py``
    used bare ``Depends(get_current_user)`` on this route, so any
    authenticated user (including a student) could read another
    teacher's authoring detail. The new authoring router enforces
    course-scoped ``course.update`` and rejects with 403.

    The test asserts the divergence direction (legacy 200 -> new 403)
    is the documented one and that the contract surface flags it.
    """
    course_id = published_course["course_id"]
    response = await client.get(
        f"/api/v1/teacher/courses/{course_id}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code in (403, 404, 405), (
        "student lacking course.update must NOT receive 200 from authoring router; "
        f"got {response.status_code}"
    )
    new_status = response.status_code
    legacy_status = 200

    captured = Snapshot(
        backend="old",
        endpoint=f"/api/v1/teacher/courses/{course_id}",
        method="GET",
        status_code=legacy_status,
        headers={},
        body_json={"id": str(course_id), "title": "Parity Course"},
    )
    replayed = Snapshot(
        backend="new",
        endpoint=f"/api/v1/teacher/courses/{course_id}",
        method="GET",
        status_code=new_status,
        headers={},
        body_json=response.json()
        if response.headers.get("content-type", "").startswith("application/json")
        else {"detail": "permission_denied"},
    )
    contract = load_contract(f"/api/v1/teacher/courses/{course_id}", parity_dir=PARITY_DIR)
    result = compare(captured, replayed, contract)
    assert not result.passed, (
        "auth-tightening must be flagged as a divergence; if this passes the "
        "contract is too permissive"
    )
    assert any("status_code" in d.lower() or "status" in d.lower() for d in result.divergences), (
        f"expected status divergence, got {result.divergences}"
    )

    assert new_status >= 400 > legacy_status, (
        "auth-tightening direction MUST be permissive -> strict; "
        f"legacy={legacy_status} new={new_status}"
    )


@legacy_skip
async def test_live_legacy_courses_parity_capture(
    client: httpx.AsyncClient,
    student_bearer: str,
) -> None:
    """Live capture from ``backend/`` -- only runs when legacy is reachable.

    Captures ``GET /api/v1/courses`` from the legacy backend, replays
    against the new ASGI app under the parity contract, and asserts
    parity OK. When legacy isn't running this test is skipped (the
    standard local-dev case).
    """
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5.0) as legacy:
        legacy_response = await legacy.get(
            "/api/v1/courses",
            headers={"Authorization": f"Bearer {student_bearer}"},
        )
    if legacy_response.status_code in (401, 403):
        pytest.skip(
            "legacy backend is running but rejected our JWT "
            "(JWT_SECRET_KEY mismatch between backends -- expected in dev)"
        )
    captured = Snapshot(
        backend="old",
        endpoint="/api/v1/courses",
        method="GET",
        status_code=legacy_response.status_code,
        headers={},
        body_json=legacy_response.json()
        if legacy_response.headers.get("content-type", "").startswith("application/json")
        else {},
    )

    new_response = await client.get(
        "/api/v1/courses",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    replayed = Snapshot(
        backend="new",
        endpoint="/api/v1/courses",
        method="GET",
        status_code=new_response.status_code,
        headers={},
        body_json=new_response.json(),
    )
    contract = load_contract("/api/v1/courses", parity_dir=PARITY_DIR)
    result = compare(captured, replayed, contract)
    assert result.passed, f"live legacy parity divergences: {result.divergences}"
