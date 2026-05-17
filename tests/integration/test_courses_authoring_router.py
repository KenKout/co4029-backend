"""Integration tests for ``features.courses.routers.authoring`` (T3.7).

Covers the FIX-SEC-1 invariant from Reconciliation §A9 + §E4: every
write endpoint either enforces a global permission (``course.create``)
or walks UP from the path-param sub-resource to the owning course and
runs the standard course-scoped check. The legacy bug -- bare
``Depends(get_current_user)`` on lesson / module / resource endpoints --
is verified absent both behaviourally (sibling-resource PATCH → 403) and
at the source level (grep for ``Depends(get_current_user)`` in the
authoring router source).

Test inventory (per plan §4418-4444):

* ``test_router_metadata`` -- 12 endpoints registered under ``/teacher``.
* ``test_unauthenticated_returns_401`` -- every write endpoint rejects
  no-bearer requests.
* ``test_student_403_on_authoring`` -- the seeded ``student_token`` lacks
  ``course.create`` / ``course.update``; every endpoint returns 403.
* ``test_owner_can_update_own_course`` -- ownership short-circuit fires.
* ``test_teacher_with_course_scope_can_update_assigned_course`` -- the
  seeded teacher (scope=course on test_course) PATCHes test_course → 200.
* ``test_teacher_403_on_sibling_course`` -- same teacher PATCHing a
  sibling-org course (no scope) returns 403 (course-level boundary).
* ``test_manager_org_propagation`` -- manager (scope=organization)
  PATCHes any course in the org → 200.
* ``test_publish_course_widens_status`` -- POST /publish flips status.
* ``test_archive_course`` -- POST /archive flips status.
* ``test_post_lesson_auto_creates_module_item`` -- Reconciliation §A5.
* ``test_reorder_module_items_no_unique_violation`` -- Reconciliation §A6
  two-phase swap survives a permutation.
* ``test_delete_lesson_resource_soft_deletes`` -- DELETE sets
  deleted_at + deleted_by.
* ``test_module_authoring_access_gap_fix`` -- **FIX-SEC-1**: PATCH a
  module belonging to a course the principal lacks scope on → 403.
* ``test_lesson_authoring_access_gap_fix`` -- **FIX-SEC-1** lesson form.
* ``test_resource_authoring_access_gap_fix`` -- **FIX-SEC-1** resource
  form (delete a sibling-course's resource → 403).
* ``test_no_bare_get_current_user_on_write_endpoints`` -- source-level
  grep guard ensuring no future regression reintroduces the legacy bug.
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
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import authoring_router

for _stub_name in ("learning_materials", "quizzes", "interview_configs"):
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
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
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
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
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
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID | str]]:
    """Two courses, two modules, two lessons, two resources -- A vs B.

    * Course-A is the seeded ``test_course`` (owner=admin_id; teacher has
      scope=course assignment on it).
    * Course-B is a sibling course in the same org, owned by manager_id;
      teacher has NO scope here -- this is the FIX-SEC-1 perimeter target.
    """
    course_b = uuid.uuid4()
    module_a = uuid.uuid4()
    module_b = uuid.uuid4()
    lesson_a = uuid.uuid4()
    lesson_a2 = uuid.uuid4()
    lesson_a3 = uuid.uuid4()
    lesson_b = uuid.uuid4()
    resource_a = uuid.uuid4()
    resource_b = uuid.uuid4()
    item_a1 = uuid.uuid4()
    item_a2 = uuid.uuid4()
    item_a3 = uuid.uuid4()
    storage_obj = uuid.uuid4()
    suffix = course_b.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Sibling Course', 'draft')"
            ),
            {
                "id": course_b,
                "org": seeded_users.organization_id,
                "owner": seeded_users.manager_id,
                "slug": f"sibling-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) VALUES "
                "(:ma, :ca, 'Module A', 1, 'draft'), "
                "(:mb, :cb, 'Module B', 1, 'draft')"
            ),
            {"ma": module_a, "mb": module_b, "ca": seeded_users.course_id, "cb": course_b},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) VALUES "
                "(:la, :ma, 'lesson-a', 'Lesson A', 'draft'), "
                "(:la2, :ma, 'lesson-a2', 'Lesson A2', 'draft'), "
                "(:la3, :ma, 'lesson-a3', 'Lesson A3', 'draft'), "
                "(:lb, :mb, 'lesson-b', 'Lesson B', 'draft')"
            ),
            {
                "la": lesson_a,
                "la2": lesson_a2,
                "la3": lesson_a3,
                "lb": lesson_b,
                "ma": module_a,
                "mb": module_b,
            },
        )
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, :b, :k)"),
            {"id": storage_obj, "b": "test-bucket", "k": f"materials/{suffix}.pdf"},
        )
        await conn.execute(
            text(
                "INSERT INTO lesson_resources "
                "(id, lesson_id, title, resource_type, position, "
                "visible_to_students, storage_object_id) VALUES "
                "(:ra, :la, 'Resource A', 'pdf', 1, TRUE, :so), "
                "(:rb, :lb, 'Resource B', 'pdf', 1, TRUE, :so)"
            ),
            {
                "ra": resource_a,
                "rb": resource_b,
                "la": lesson_a,
                "lb": lesson_b,
                "so": storage_obj,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO module_items (id, module_id, item_type, lesson_id, position) VALUES "
                "(:i1, :m, 'lesson', :l1, 1), "
                "(:i2, :m, 'lesson', :l2, 2), "
                "(:i3, :m, 'lesson', :l3, 3)"
            ),
            {
                "i1": item_a1,
                "i2": item_a2,
                "i3": item_a3,
                "m": module_a,
                "l1": lesson_a,
                "l2": lesson_a2,
                "l3": lesson_a3,
            },
        )

    data: dict[str, uuid.UUID | str] = {
        "course_a": seeded_users.course_id,
        "course_b": course_b,
        "module_a": module_a,
        "module_b": module_b,
        "lesson_a": lesson_a,
        "lesson_b": lesson_b,
        "resource_a": resource_a,
        "resource_b": resource_b,
        "item_a1": item_a1,
        "item_a2": item_a2,
        "item_a3": item_a3,
        "storage_object_id": storage_obj,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM lesson_resources WHERE lesson_id IN (SELECT id FROM lessons WHERE module_id = ANY(:ids))"
            ),
            {"ids": [module_a, module_b]},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = ANY(:ids)"),
            {"ids": [module_a, module_b]},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE module_id = ANY(:ids)"),
            {"ids": [module_a, module_b]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [module_a, module_b]},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_b})
        await conn.execute(text("DELETE FROM storage_objects WHERE id = :id"), {"id": storage_obj})


def test_router_metadata() -> None:
    paths = {(r.path, tuple(sorted(r.methods))) for r in authoring_router.routes}  # type: ignore[attr-defined]
    expected = {
        ("/teacher/courses", ("POST",)),
        ("/teacher/courses/{course_id}", ("PATCH",)),
        ("/teacher/courses/{course_id}/publish", ("POST",)),
        ("/teacher/courses/{course_id}/archive", ("POST",)),
        ("/teacher/courses/{course_id}/modules", ("POST",)),
        ("/teacher/modules/{module_id}", ("PATCH",)),
        ("/teacher/modules/{module_id}/prerequisites", ("PUT",)),
        ("/teacher/modules/{module_id}/items/reorder", ("PUT",)),
        ("/teacher/modules/{module_id}/lessons", ("POST",)),
        ("/teacher/lessons/{lesson_id}", ("PATCH",)),
        ("/teacher/lessons/{lesson_id}/resources", ("POST",)),
        ("/teacher/lesson-resources/{resource_id}", ("DELETE",)),
    }
    assert expected.issubset(paths)
    assert authoring_router.prefix == "/teacher"


async def test_unauthenticated_returns_401(
    client: httpx.AsyncClient, scenario: dict[str, uuid.UUID | str]
) -> None:
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "X"},
    )
    assert response.status_code == 401
    response = await client.delete(f"/api/v1/teacher/lesson-resources/{scenario['resource_a']}")
    assert response.status_code == 401


async def test_student_403_on_authoring(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "X"},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 403
    response = await client.patch(
        f"/api/v1/teacher/modules/{scenario['module_a']}",
        json={"title": "X"},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 403
    response = await client.patch(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}",
        json={"title": "X"},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 403


async def test_owner_can_update_own_course(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Admin owns the seeded test_course; ownership short-circuits the perm check."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "Owner Patched"},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["title"] == "Owner Patched"


async def test_teacher_with_course_scope_can_update_assigned_course(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Seeded teacher has scope=course on Course-A → course.update passes."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "Teacher Patched A"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text


async def test_teacher_403_on_sibling_course(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Teacher has no scope on Course-B → course-level boundary returns 403."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_b']}",
        json={"title": "Should Fail"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["error"] == "permission_denied"
    assert body["detail"]["scope"] == "course"


async def test_manager_org_propagation(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Manager has scope=organization with course.update → can patch any course in org."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "Manager Patched A"},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_b']}",
        json={"title": "Manager Patched B"},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text


async def test_publish_course_widens_status(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    response = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_b']}/publish",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "published"
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT status FROM courses WHERE id = :id"),
                {"id": scenario["course_b"]},
            )
        ).one()
        assert row.status == "published"


async def test_archive_course(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    response = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_b']}/archive",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "archived"


async def test_post_lesson_auto_creates_module_item(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """Reconciliation §A5 -- POST /modules/{id}/lessons creates lesson + ModuleItem."""
    response = await client.post(
        f"/api/v1/teacher/modules/{scenario['module_a']}/lessons",
        json={
            "module_id": str(scenario["module_a"]),
            "slug": f"new-{uuid.uuid4().hex[:6]}",
            "title": "Auto-Item Lesson",
            "lesson_type": "video",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 201, response.text
    new_lesson_id = response.json()["id"]
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT COUNT(*) AS n FROM module_items WHERE module_id = :m AND lesson_id = :l"),
            {"m": scenario["module_a"], "l": new_lesson_id},
        )
        assert result.one().n == 1


async def test_reorder_module_items_no_unique_violation(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """Reconciliation §A6 -- _OFFSET=100_000 two-phase swap survives a permutation."""
    new_order = [scenario["item_a3"], scenario["item_a1"], scenario["item_a2"]]
    response = await client.put(
        f"/api/v1/teacher/modules/{scenario['module_a']}/items/reorder",
        json={
            "module_id": str(scenario["module_a"]),
            "new_order": [str(i) for i in new_order],
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, position FROM module_items WHERE module_id = :m ORDER BY position"
                ),
                {"m": scenario["module_a"]},
            )
        ).all()
    by_id = {r.id: r.position for r in rows}
    assert by_id[scenario["item_a3"]] == 1
    assert by_id[scenario["item_a1"]] == 2
    assert by_id[scenario["item_a2"]] == 3


async def test_delete_lesson_resource_soft_deletes(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    response = await client.delete(
        f"/api/v1/teacher/lesson-resources/{scenario['resource_a']}",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 204, response.text
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT deleted_at, deleted_by FROM lesson_resources WHERE id = :id"),
                {"id": scenario["resource_a"]},
            )
        ).one()
    assert row.deleted_at is not None
    assert row.deleted_by == seeded_users.manager_id


async def test_module_authoring_access_gap_fix(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """**FIX-SEC-1** -- pre-T3.7 the legacy router used Depends(get_current_user)
    on /teacher/modules/{id}; teacher (with course.update on Course-A) could PATCH
    Module-B (which belongs to Course-B). T3.7 walks module → course before
    checking perms, so this now returns 403.
    """
    response = await client.patch(
        f"/api/v1/teacher/modules/{scenario['module_b']}",
        json={"title": "Should Fail"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["error"] == "permission_denied"


async def test_lesson_authoring_access_gap_fix(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """**FIX-SEC-1** -- lesson-form sibling-resource regression guard."""
    response = await client.patch(
        f"/api/v1/teacher/lessons/{scenario['lesson_b']}",
        json={"title": "Should Fail"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403


async def test_resource_authoring_access_gap_fix(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """**FIX-SEC-1** -- DELETE on a sibling-course's resource → 403."""
    response = await client.delete(
        f"/api/v1/teacher/lesson-resources/{scenario['resource_b']}",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403


def test_no_bare_get_current_user_on_write_endpoints() -> None:
    """Source-level guard: every endpoint route must depend on a require_*
    factory, never on bare get_current_user. Companion to the FIX-SEC-1
    perimeter tests; ensures no future regression reintroduces the legacy
    bug from ``backend/app/routes/teacher/courses_router.py``.
    """
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "courses"
        / "routers"
        / "authoring.py"
    ).read_text(encoding="utf-8")
    bare = re.findall(r"Depends\(get_current_user\)", src)
    assert bare == [], f"authoring.py uses bare Depends(get_current_user): {bare}"
