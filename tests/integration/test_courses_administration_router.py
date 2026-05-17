"""Integration tests for ``features.courses.routers.administration`` (T3.9).

Covers the IT-Admin operational surface at ``/api/v1/admin``. Acceptance
criteria from plan §4541-4545 + QA §4547-4564 map to:

* ``test_router_metadata`` — 5 endpoints registered under ``/admin``,
  no DELETE on ``/admin/courses/{id}`` (no-hard-delete invariant).
* ``test_unauthenticated_returns_401`` — bearer-less requests rejected.
* ``test_admin_token_lists_soft_deleted_courses`` — soft-delete a
  course, admin GET ``/admin/courses`` returns it.
* ``test_manager_token_403_on_admin`` — manager hits 403.
* ``test_student_token_403_on_admin`` — student hits 403.
* ``test_restore_clears_deleted_at_and_deleted_by`` — soft-delete then
  POST restore; assert ``deleted_at IS NULL``, ``deleted_by IS NULL``,
  ``updated_by = admin.user_id``.
* ``test_restore_404_on_unknown_course`` — POST restore for a random
  UUID returns 404.
* ``test_restore_404_on_active_course`` — POST restore on an already
  active course returns 404 (T3.5 service raises ``NotFoundError``).
* ``test_audit_endpoint_aggregates_ai_model_calls`` — seed a generation
  run + ai_model_calls rows, assert aggregation includes them.
* ``test_processing_endpoint_returns_recent_jobs`` — seed processing
  jobs joined via generation_runs, assert returned shape.
* ``test_stats_returns_counts_by_status`` — seed courses across
  statuses, assert stats counts match.
* ``test_no_hard_delete_endpoint_exists`` — ``router.routes`` has no
  DELETE on ``/admin/courses/{course_id}`` (plan §4526).
* ``test_restore_does_not_cascade_to_children`` — soft-delete a course
  with module children, restore the course, children remain
  soft-deleted (T0.15 cascade left them tombstoned).
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
from abridgeai.features.courses.routers import administration_router

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
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def manager_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.manager_id)
    yield create_access_token(user_id=seeded_users.manager_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Three scenario courses + AI-pipeline trail.

    * ``soft_deleted_course`` — ``deleted_at`` set; admin sees it.
    * ``active_course`` — published, no AI work.
    * ``audited_course`` — has a generation_run + processing_job + 2
      ai_model_calls (one row per call, totals deterministic).
    * ``module_a`` / ``lesson_a`` — children of ``soft_deleted_course``
      pre-tombstoned to assert restore is non-cascading.
    """
    suffix = uuid.uuid4().hex[:8]
    soft_deleted_course = uuid.uuid4()
    active_course = uuid.uuid4()
    audited_course = uuid.uuid4()
    module_a = uuid.uuid4()
    lesson_a = uuid.uuid4()
    generation_run = uuid.uuid4()
    processing_job = uuid.uuid4()
    call_one = uuid.uuid4()
    call_two = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status, "
                "deleted_at, deleted_by) VALUES "
                "(:sd, :org, :owner, :slug_sd, 'Soft-Deleted Course', 'draft', NOW(), :owner), "
                "(:ac, :org, :owner, :slug_ac, 'Active Course', 'published', NULL, NULL), "
                "(:au, :org, :owner, :slug_au, 'Audited Course', 'draft', NULL, NULL)"
            ),
            {
                "sd": soft_deleted_course,
                "ac": active_course,
                "au": audited_course,
                "org": seeded_users.organization_id,
                "owner": seeded_users.admin_id,
                "slug_sd": f"sd-course-{suffix}",
                "slug_ac": f"ac-course-{suffix}",
                "slug_au": f"au-course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, position, title, status, deleted_at) "
                "VALUES (:id, :cid, 1, :title, 'draft', NOW())"
            ),
            {"id": module_a, "cid": soft_deleted_course, "title": f"Mod-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, lesson_type, "
                "deleted_at) VALUES (:id, :mid, :slug, :title, 'video', NOW())"
            ),
            {
                "id": lesson_a,
                "mid": module_a,
                "slug": f"less-{suffix}",
                "title": f"Less-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO generation_runs (id, generation_type, source_scope_kind, "
                "course_id, status) VALUES (:id, 'quiz', 'course', :cid, 'completed')"
            ),
            {"id": generation_run, "cid": audited_course},
        )
        await conn.execute(
            text(
                "INSERT INTO processing_jobs (id, entity_type, entity_id, job_type, "
                "status, progress_percent) "
                "VALUES (:id, 'generation_run', :gid, 'generate_quiz', 'completed', 100)"
            ),
            {"id": processing_job, "gid": generation_run},
        )
        await conn.execute(
            text(
                "INSERT INTO ai_model_calls (id, generation_run_id, processing_job_id, "
                "operation, model_name, input_tokens, output_tokens, "
                "estimated_cost_usd, status) VALUES "
                "(:c1, :gid, :pid, 'chat_completion', 'gpt-4o', 100, 50, 0.0125, 'success'), "
                "(:c2, :gid, :pid, 'chat_completion', 'gpt-4o', 200, 75, 0.0250, 'success')"
            ),
            {"c1": call_one, "c2": call_two, "gid": generation_run, "pid": processing_job},
        )

    data: dict[str, uuid.UUID] = {
        "soft_deleted_course": soft_deleted_course,
        "active_course": active_course,
        "audited_course": audited_course,
        "module_a": module_a,
        "lesson_a": lesson_a,
        "generation_run": generation_run,
        "processing_job": processing_job,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM ai_model_calls WHERE id = ANY(:ids)"),
            {"ids": [call_one, call_two]},
        )
        await conn.execute(
            text("DELETE FROM processing_jobs WHERE id = :id"), {"id": processing_job}
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE id = :id"), {"id": generation_run}
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_a})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_a})
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [soft_deleted_course, active_course, audited_course]},
        )


def test_router_metadata() -> None:
    paths = {(r.path, tuple(sorted(r.methods))) for r in administration_router.routes}  # type: ignore[attr-defined]
    expected = {
        ("/admin/courses", ("GET",)),
        ("/admin/courses/{course_id}/restore", ("POST",)),
        ("/admin/courses/{course_id}/audit", ("GET",)),
        ("/admin/courses/{course_id}/processing", ("GET",)),
        ("/admin/courses/_stats", ("GET",)),
    }
    assert expected.issubset(paths)
    assert administration_router.prefix == "/admin"


def test_no_hard_delete_endpoint_exists() -> None:
    """Plan §4526: hard-delete is forbidden; restore is the recovery path."""
    for route in administration_router.routes:
        methods = getattr(route, "methods", set())
        path = getattr(route, "path", "")
        assert not ("DELETE" in methods and path.startswith("/admin/courses/{")), (
            f"Hard-delete endpoint registered: {methods} {path}"
        )


async def test_unauthenticated_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/admin/courses")
    assert response.status_code == 401


async def test_admin_token_lists_soft_deleted_courses(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        "/api/v1/admin/courses",
        params={"limit": 100, "include_deleted": "true"},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    ids = {item["id"] for item in body["items"]}
    assert str(scenario["soft_deleted_course"]) in ids
    assert str(scenario["active_course"]) in ids


async def test_manager_token_403_on_admin(client: httpx.AsyncClient, manager_bearer: str) -> None:
    response = await client.get(
        "/api/v1/admin/courses", headers={"Authorization": f"Bearer {manager_bearer}"}
    )
    assert response.status_code == 403


async def test_student_token_403_on_admin(client: httpx.AsyncClient, student_bearer: str) -> None:
    response = await client.get(
        "/api/v1/admin/courses", headers={"Authorization": f"Bearer {student_bearer}"}
    )
    assert response.status_code == 403


async def test_restore_clears_deleted_at_and_deleted_by(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
) -> None:
    response = await client.post(
        f"/api/v1/admin/courses/{scenario['soft_deleted_course']}/restore",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT deleted_at, deleted_by, updated_by FROM courses WHERE id = :id"),
                {"id": scenario["soft_deleted_course"]},
            )
        ).one()
    assert row.deleted_at is None
    assert row.deleted_by is None
    assert row.updated_by == seeded_users.admin_id


async def test_restore_404_on_unknown_course(client: httpx.AsyncClient, admin_bearer: str) -> None:
    response = await client.post(
        f"/api/v1/admin/courses/{uuid.uuid4()}/restore",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 404


async def test_restore_404_on_active_course(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    """T3.5 service raises NotFoundError when no soft-deleted row matches."""
    response = await client.post(
        f"/api/v1/admin/courses/{scenario['active_course']}/restore",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 404


async def test_audit_endpoint_aggregates_ai_model_calls(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        f"/api/v1/admin/courses/{scenario['audited_course']}/audit",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["course_id"] == str(scenario["audited_course"])
    assert body["total_calls"] == 2
    assert body["total_input_tokens"] == 300
    assert body["total_output_tokens"] == 125
    assert body["generation_runs"] == 1
    assert body["processing_jobs"] == 1
    assert abs(float(body["total_cost_usd"]) - 0.0375) < 1e-6


async def test_processing_endpoint_returns_recent_jobs(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        f"/api/v1/admin/courses/{scenario['audited_course']}/processing",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    job_ids = {row["id"] for row in body}
    assert str(scenario["processing_job"]) in job_ids
    matched = next(row for row in body if row["id"] == str(scenario["processing_job"]))
    assert matched["status"] == "completed"
    assert matched["job_type"] == "generate_quiz"
    assert matched["progress_percent"] == 100


async def test_stats_returns_counts_by_status(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    response = await client.get(
        "/api/v1/admin/courses/_stats",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_courses"] >= 2
    assert body["soft_deleted_courses"] >= 1
    statuses = {row["status"]: row["count"] for row in body["by_status"]}
    assert statuses.get("draft", 0) >= 1
    assert statuses.get("published", 0) >= 1
    assert isinstance(body["top_draft_owners"], list)


async def test_restore_does_not_cascade_to_children(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Plan / T3.5 docstring: restore is leaf-only.

    Children stay tombstoned so admins must explicitly restore each
    subtree level (matches services.administration documented behaviour).
    """
    response = await client.post(
        f"/api/v1/admin/courses/{scenario['soft_deleted_course']}/restore",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text

    async with engine.begin() as conn:
        module_deleted_at = (
            await conn.execute(
                text("SELECT deleted_at FROM modules WHERE id = :id"),
                {"id": scenario["module_a"]},
            )
        ).scalar_one()
        lesson_deleted_at = (
            await conn.execute(
                text("SELECT deleted_at FROM lessons WHERE id = :id"),
                {"id": scenario["lesson_a"]},
            )
        ).scalar_one()
    assert module_deleted_at is not None
    assert lesson_deleted_at is not None


def test_no_bare_get_current_user() -> None:
    """FIX-CRIT-4: every admin route must use a ``require_*`` factory.

    Source-level guard mirroring T1.10 / T1.13. Bare
    ``Depends(get_current_user)`` is the legacy bug shape.
    """
    src = (
        Path(__file__).resolve().parents[2]
        / "abridgeai"
        / "features"
        / "courses"
        / "routers"
        / "administration.py"
    ).read_text()
    for line in src.splitlines():
        if "Depends(get_current_user)" in line and "require_" not in line:
            raise AssertionError(f"Bare get_current_user in admin router: {line}")
