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
import abridgeai.features.materials.models  # noqa: F401  -- learning_materials FK target for lessons.primary_material_id
import abridgeai.features.quizzes.models  # noqa: F401  -- quizzes FK target for module_items.quiz_id
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import authoring_router
from abridgeai.features.courses.routers.learner import router as courses_learner_router
from abridgeai.features.materials.routers.authoring import (
    router as materials_authoring_router,
)

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
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.include_router(materials_authoring_router, prefix="/api/v1")
    fastapi_app.include_router(courses_learner_router, prefix="/api/v1")
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


async def test_create_course_resolves_org_from_token(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """``POST /teacher/courses`` derives ``organization_id`` and
    ``owner_user_id`` from the bearer token, NOT the payload.

    Regression for the FK violation triggered by the SPA hardcoding a
    placeholder UUID. The endpoint must:
      1. accept payloads that omit org/owner entirely;
      2. reject payloads that include them (strict-extras);
      3. write the manager's primary org and the manager as owner.
    """
    suffix = uuid.uuid4().hex[:8]
    response = await client.post(
        "/api/v1/teacher/courses",
        json={
            "slug": f"smoke-{suffix}",
            "title": f"Smoke Course {suffix}",
            "description": "regression for forged organization_id",
            "level": "beginner",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["organization_id"] == str(seeded_users.organization_id)
    assert body["owner_user_id"] == str(seeded_users.manager_id)
    assert body["status"] == "draft"
    try:
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT organization_id, owner_user_id, status FROM courses WHERE id = :id"
                    ),
                    {"id": body["id"]},
                )
            ).one()
        assert row.organization_id == seeded_users.organization_id
        assert row.owner_user_id == seeded_users.manager_id
        assert row.status == "draft"
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": body["id"]})


async def test_create_course_rejects_forged_organization_id(
    client: httpx.AsyncClient,
    manager_bearer: str,
) -> None:
    """A forged ``organization_id`` in the payload must be rejected at the
    schema layer (extra='forbid'), not silently honoured.

    This is the exact wire shape the SPA was sending before the fix
    (placeholder UUID + redundant owner_user_id); without strict-extras the
    backend would have happily passed it through to Postgres.
    """
    forged_org = "00000000-0000-0000-0000-000000000001"
    response = await client.post(
        "/api/v1/teacher/courses",
        json={
            "organization_id": forged_org,
            "owner_user_id": forged_org,
            "slug": f"forge-{uuid.uuid4().hex[:8]}",
            "title": "Forged",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 422, response.text


async def test_create_course_duplicate_slug_returns_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
) -> None:
    """Re-submitting the same slug must surface as 409, not the legacy 500.

    Regression for the ``UniqueViolation`` on ``uq_courses_org_slug`` —
    duplicate (organization_id, slug) used to bubble out as a 500
    IntegrityError; the service now maps it to ``ConflictError`` and the
    router renders 409 with the stable error code ``conflict``.
    """
    suffix = uuid.uuid4().hex[:8]
    body = {
        "slug": f"dup-{suffix}",
        "title": f"Dup Slug Course {suffix}",
    }
    first = await client.post(
        "/api/v1/teacher/courses",
        json=body,
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert first.status_code == 201, first.text
    created_id = first.json()["id"]
    try:
        second = await client.post(
            "/api/v1/teacher/courses",
            json=body,
            headers={"Authorization": f"Bearer {manager_bearer}"},
        )
        assert second.status_code == 409, second.text
        detail = second.json()["detail"]
        assert detail["error"] == "conflict"
        assert "course_slug_taken" in detail["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": created_id})


async def test_check_course_slug_reports_availability(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
) -> None:
    """``GET /teacher/courses/check-slug`` returns ``available=true`` for a
    free slug and ``available=false`` once a course occupies that slug in
    the caller's primary organization.

    Frontend uses this for inline validation on the new-course form so
    teachers see a duplicate before submission instead of after a 409.
    """
    suffix = uuid.uuid4().hex[:8]
    free_slug = f"check-{suffix}"
    auth = {"Authorization": f"Bearer {manager_bearer}"}

    free_resp = await client.get(
        f"/api/v1/teacher/courses/check-slug?slug={free_slug}",
        headers=auth,
    )
    assert free_resp.status_code == 200, free_resp.text
    assert free_resp.json() == {"available": True}

    create_resp = await client.post(
        "/api/v1/teacher/courses",
        json={"slug": free_slug, "title": "Slug Check Course"},
        headers=auth,
    )
    assert create_resp.status_code == 201, create_resp.text
    created_id = create_resp.json()["id"]
    try:
        taken_resp = await client.get(
            f"/api/v1/teacher/courses/check-slug?slug={free_slug}",
            headers=auth,
        )
        assert taken_resp.status_code == 200, taken_resp.text
        assert taken_resp.json() == {"available": False}
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": created_id})


async def test_check_course_slug_requires_create_permission(
    client: httpx.AsyncClient,
    student_bearer: str,
) -> None:
    """A student token (no ``course.create``) must be denied. Same auth
    contract as ``POST /teacher/courses`` so the SPA cannot probe for
    existing slugs from an unauthorized session.
    """
    response = await client.get(
        "/api/v1/teacher/courses/check-slug?slug=anything",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 403


async def test_create_module_duplicate_position_returns_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``uq_modules_course_position`` collisions surface as 409.

    The seeded course already has a Module at position 1; a second
    request targeting the same position must be rejected without a
    500 IntegrityError.
    """
    response = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_a']}/modules",
        json={
            "course_id": str(scenario["course_a"]),
            "title": "Dup-Position Module",
            "position": 1,
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "conflict"
    assert "module_position_taken" in detail["message"]


async def test_create_lesson_duplicate_slug_returns_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``uq_lessons_module_slug`` collisions surface as 409.

    The seeded module already has ``lesson-a``; a second lesson with the
    same slug under the same module must return 409.
    """
    response = await client.post(
        f"/api/v1/teacher/modules/{scenario['module_a']}/lessons",
        json={
            "module_id": str(scenario["module_a"]),
            "slug": "lesson-a",
            "title": "Dup-Slug Lesson",
            "lesson_type": "video",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "conflict"
    assert "lesson_slug_taken" in detail["message"]


async def test_create_lesson_resource_duplicate_position_returns_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``uq_lesson_resources_position`` collisions surface as 409."""
    response = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}/resources",
        json={
            "lesson_id": str(scenario["lesson_a"]),
            "title": "Dup-Position Resource",
            "resource_type": "pdf",
            "storage_object_id": str(scenario["storage_object_id"]),
            "position": 1,
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "conflict"
    assert "lesson_resource_position_taken" in detail["message"]


async def test_get_authoring_courses_lists_owned_courses(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /teacher/courses`` returns courses owned by the bearer.

    Replaces the legacy 405 the SPA was hitting (only POST was registered
    on this path). The seeded admin owns ``test_course`` so the response
    must contain it.
    """
    response = await client.get(
        "/api/v1/teacher/courses",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    course_ids = {c["id"] for c in body}
    assert str(scenario["course_a"]) in course_ids


async def test_get_authoring_courses_includes_assigned_courses(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """The teacher seed has ``role=teacher, scope=course`` on ``course_a``
    (Course-A) but does NOT own it. They must still see it in their
    authoring list — that's the "courses I co-author" path.
    """
    response = await client.get(
        "/api/v1/teacher/courses",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    course_ids = {c["id"] for c in body}
    assert str(scenario["course_a"]) in course_ids


async def test_get_authoring_courses_excludes_unrelated_courses(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Teacher has no scope on Course-B; the list must NOT leak it."""
    response = await client.get(
        "/api/v1/teacher/courses",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    course_ids = {c["id"] for c in body}
    assert str(scenario["course_b"]) not in course_ids


async def test_get_authoring_courses_unauthenticated_returns_401(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/api/v1/teacher/courses")
    assert response.status_code == 401


async def test_get_authoring_course_detail(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    response = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(scenario["course_a"])
    assert body["organization_id"]


async def test_get_authoring_course_detail_unknown_returns_404(
    client: httpx.AsyncClient,
    admin_bearer: str,
) -> None:
    """Unknown course ids must surface as 404, not 500 or empty 200."""
    response = await client.get(
        f"/api/v1/teacher/courses/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 404


async def test_get_authoring_course_detail_no_scope_returns_403(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Teacher with no scope on Course-B gets 403, not the row."""
    response = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_b']}",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403


async def test_get_authoring_course_content(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /teacher/courses/{id}/content`` returns the authoring tree
    (drafts included, owners only)."""
    response = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_a']}/content",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "course" in body
    assert "modules" in body
    assert body["course"]["id"] == str(scenario["course_a"])


async def test_get_authoring_course_roster(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /teacher/courses/{id}/roster`` returns the same shape as
    ``/dept/courses/{id}/roster`` so the SPA's existing
    useTeacherCourseRoster hook works without code changes.
    """
    response = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_a']}/roster",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    if body:
        sample = body[0]
        for key in (
            "enrollment_id",
            "student_id",
            "primary_email",
            "status",
            "enrolled_at",
        ):
            assert key in sample


async def test_get_authoring_lesson(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /teacher/lessons/{id}`` returns the authoring projection."""
    response = await client.get(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(scenario["lesson_a"])


async def test_get_authoring_lesson_not_found_returns_404(
    client: httpx.AsyncClient,
    admin_bearer: str,
) -> None:
    response = await client.get(
        f"/api/v1/teacher/lessons/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 404


async def test_list_authoring_lesson_resources(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /teacher/lessons/{id}/resources`` returns ALL resources, including
    hidden / draft (no ``visible_to_students=TRUE`` filter).

    The seeded resource_a has ``visible_to_students=TRUE`` so the count
    matches the learner-side endpoint here, but the contract diverges
    once a teacher hides a resource.
    """
    response = await client.get(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}/resources",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    resource_ids = {r["id"] for r in body}
    assert str(scenario["resource_a"]) in resource_ids


async def test_get_authoring_resource_download_url(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /teacher/lesson-resources/{id}/download-url`` returns a
    presigned URL with the SPA's expected ``stream_url`` field name.

    S3 is monkey-patched — same pattern the learner-side router tests
    use — so the test is hermetic.
    """
    from datetime import UTC, timedelta
    from datetime import datetime as _datetime

    from abridgeai.features.courses.services import authoring as authoring_service

    expires_at = _datetime.now(tz=UTC) + timedelta(minutes=5)

    async def _fake_create_stream_url(_target, *, response_headers=None, settings=None):  # noqa: ANN001
        del response_headers, settings
        return ("https://stub.local/signed", expires_at)

    monkeypatch.setattr(authoring_service, "create_stream_url", _fake_create_stream_url)

    response = await client.get(
        f"/api/v1/teacher/lesson-resources/{scenario['resource_a']}/download-url",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stream_url"] == "https://stub.local/signed"
    assert "expires_at" in body


async def test_get_authoring_resource_download_url_not_found(
    client: httpx.AsyncClient,
    admin_bearer: str,
) -> None:
    """Unknown resource ids return 404 from the auth dep, not 500."""
    response = await client.get(
        f"/api/v1/teacher/lesson-resources/{uuid.uuid4()}/download-url",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 404


async def test_update_module_item_unlock_rule(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``PATCH /teacher/module-items/{id}`` updates ``unlock_rule_json``.

    The schema explicitly allowlists this single field — position +
    polymorphic identity are immutable here (reorder + create / delete
    are the only ways to change them).
    """
    response = await client.patch(
        f"/api/v1/teacher/module-items/{scenario['item_a1']}",
        json={"unlock_rule_json": {"requires_quiz_pass": True}},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(scenario["item_a1"])


async def test_delete_module_item_soft_deletes(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """``DELETE /teacher/module-items/{id}`` soft-deletes the row.

    Sibling positions stay intact; the lesson/quiz target survives
    because module_items is just an ordering link.
    """
    response = await client.delete(
        f"/api/v1/teacher/module-items/{scenario['item_a3']}",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 204, response.text
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT deleted_at FROM module_items WHERE id = :id"),
                {"id": scenario["item_a3"]},
            )
        ).one()
    assert row.deleted_at is not None


async def test_lesson_processing_summary_zero_when_no_materials(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /teacher/lessons/{id}/processing-summary`` returns zeroes for an
    empty lesson — the SPA should never have to null-check the response.
    """
    response = await client.get(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}/processing-summary",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lesson_id"] == str(scenario["lesson_a"])
    for key in (
        "materials_total",
        "versions_total",
        "pending_versions",
        "processing_versions",
        "completed_versions",
        "failed_versions",
    ):
        assert body[key] == 0


async def test_get_lesson_outline_returns_synthetic_section(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /lessons/{id}/outline`` returns the SPA-expected shape.

    Until ``build_lesson_outline`` is ported the response carries one
    synthetic ``body`` section so the SPA renders the empty-state
    correctly. Contract: same field set as the eventual semantic-section
    response, so the frontend doesn't need a feature flag.
    """
    response = await client.get(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}/outline",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lesson_id"] == str(scenario["lesson_a"])
    assert isinstance(body["sections"], list)
    assert len(body["sections"]) >= 1
    sample = body["sections"][0]
    for key in (
        "id",
        "title",
        "depth",
        "chunk_count",
        "char_count",
        "page_range",
        "content_role",
        "preview",
    ):
        assert key in sample
