"""End-to-end Phase 3 integration suite (T3.10).

Proves the 4-router + DRAFT_VISIBILITY composition closes Phase 3:

* Mount all 5 routers (``learner_router``, ``me_courses_router``,
  ``authoring_router``, ``assignment_router``, ``administration_router``)
  on a single FastAPI app under ``/api/v1`` -- the same shape T3.10
  documents in ``routers/__init__.py``.
* Exercise the real wired stack -- nothing is mocked. ``get_db`` is
  overridden with the test session factory (T1.13 pattern); JWTs are
  minted against real ``auth_sessions`` rows so ``get_current_user``
  authenticates without bypass.

The plan body (§4570-4623) lists three e2e scenarios:

1. **Full lifecycle** (``test_full_course_lifecycle_manager_teacher_student``)
   -- Manager creates course -> HOD assigns Teacher -> Teacher adds
   Module + Lesson + LessonResource (auto ModuleItem per §A5) -> Manager
   publishes -> Student reads -> Student is rejected from authoring.
2. **HOD oversight** (``test_hod_oversight_dept_courses_view``) -- HOD
   GETs the dept courses list (sees drafts) and the authoring tree.
3. **Admin restore** (``test_admin_soft_delete_then_restore``) -- soft-
   delete a course (raw SQL; admin router intentionally exposes no
   ``DELETE`` per plan §4526), admin sees it via ``GET /admin/courses``,
   ``POST /admin/courses/{id}/restore`` clears the tombstone.
4. **Draft module hidden** (``test_draft_module_excluded_from_student_content_tree``)
   -- DRAFT_VISIBILITY: published course + draft module -> student's
   content tree returns the course but the module list excludes drafts.
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
import abridgeai.features.materials.models  # noqa: F401  -- learning_materials FK target for lessons.primary_material_id
import abridgeai.features.quizzes.models  # noqa: F401  -- quizzes FK target for module_items.quiz_id
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
    """Mount all 5 Phase 3 routers under ``/api/v1`` -- the production shape."""

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
async def hod_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.hod_id)
    yield create_access_token(user_id=seeded_users.hod_id, session_id=sid)
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
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def storage_object(engine: AsyncEngine) -> AsyncIterator[uuid.UUID]:
    """Seed a single ``storage_objects`` row backing lesson-resource creates."""
    obj_id = uuid.uuid4()
    suffix = obj_id.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, :b, :k)"),
            {"id": obj_id, "b": "test-bucket", "k": f"materials/e2e-{suffix}.pdf"},
        )
    try:
        yield obj_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM storage_objects WHERE id = :id"), {"id": obj_id})


@pytest_asyncio.fixture
async def cleanup_courses(engine: AsyncEngine) -> AsyncIterator[set[uuid.UUID]]:
    """Test-scoped registry of created course IDs -- teardown removes them."""
    ids: set[uuid.UUID] = set()
    yield ids
    if not ids:
        return
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM lesson_resources WHERE lesson_id IN ("
                "SELECT l.id FROM lessons l JOIN modules m ON m.id = l.module_id "
                "WHERE m.course_id = ANY(:ids))"
            ),
            {"ids": list(ids)},
        )
        await conn.execute(
            text(
                "DELETE FROM module_items WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = ANY(:ids))"
            ),
            {"ids": list(ids)},
        )
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = ANY(:ids))"
            ),
            {"ids": list(ids)},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE course_id = ANY(:ids)"),
            {"ids": list(ids)},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE course_id = ANY(:ids)"),
            {"ids": list(ids)},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": list(ids)},
        )


async def test_full_course_lifecycle_manager_teacher_student(
    client: httpx.AsyncClient,
    manager_bearer: str,
    hod_bearer: str,
    teacher_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    storage_object: uuid.UUID,
    cleanup_courses: set[uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """The full Phase 3 happy path running end-to-end across all 5 routers.

    Walks through every router in production order:

    1. Manager (``course.create``) creates a course in the HOD's
       org_unit so the HOD's scope-bound staffing perm applies.
    2. HOD (``course.assign_teacher`` at ``scope=org_unit``) assigns the
       seeded teacher to the new course via ``/dept`` -- writes a fresh
       ``user_role_assignments`` row with ``scope_kind='course'``.
    3. Teacher (now ``scope=course`` on the new course via the assignment
       above) authors a Module + Lesson + LessonResource and publishes
       the module + lesson. ``add_lesson`` auto-creates the linking
       ``ModuleItem`` per Reconciliation §A5.
    4. Manager publishes the course (``course.publish``).
    5. Student (``course.read``) reads the course by slug, by id, and
       fetches the published content tree.
    6. Student is rejected by ``POST /teacher/courses`` (403 -- lacks
       ``course.create``).
    """
    suffix = uuid.uuid4().hex[:8]
    new_slug = f"e2e-course-{suffix}"

    # 1. Manager creates the course in HOD's org_unit so the HOD can staff it.
    create_payload = {
        "org_unit_id": str(seeded_users.org_unit_id),
        "slug": new_slug,
        "title": "E2E Lifecycle Course",
        "description": "Phase 3 e2e proof",
    }
    response = await client.post(
        "/api/v1/teacher/courses",
        json=create_payload,
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 201, response.text
    course_id = uuid.UUID(response.json()["id"])
    cleanup_courses.add(course_id)
    assert response.json()["status"] == "draft"

    # 2. HOD assigns the seeded teacher to the new course.
    assign_response = await client.post(
        f"/api/v1/dept/courses/{course_id}/teachers",
        json={"user_id": str(seeded_users.teacher_id)},
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert assign_response.status_code == 201, assign_response.text
    body = assign_response.json()
    assert body["course_id"] == str(course_id)
    assert body["user_id"] == str(seeded_users.teacher_id)
    assert body["role_code"] == "teacher"
    assert body["scope_kind"] == "course"
    assert body["granted_by"] == str(seeded_users.hod_id)

    # 3a. Teacher (now scope=course) creates a module on the course.
    module_response = await client.post(
        f"/api/v1/teacher/courses/{course_id}/modules",
        json={
            "course_id": str(course_id),
            "title": "Module 1",
            "position": 1,
        },
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert module_response.status_code == 201, module_response.text
    module_id = uuid.UUID(module_response.json()["id"])

    # 3b. Teacher creates a lesson -- service auto-creates the ModuleItem (§A5).
    lesson_response = await client.post(
        f"/api/v1/teacher/modules/{module_id}/lessons",
        json={
            "module_id": str(module_id),
            "slug": f"e2e-lesson-{suffix}",
            "title": "Lesson 1",
            "lesson_type": "video",
        },
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert lesson_response.status_code == 201, lesson_response.text
    lesson_id = uuid.UUID(lesson_response.json()["id"])

    async with engine.begin() as conn:
        item_count = (
            (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) AS n FROM module_items "
                        "WHERE module_id = :m AND lesson_id = :l"
                    ),
                    {"m": module_id, "l": lesson_id},
                )
            )
            .one()
            .n
        )
    assert item_count == 1, "lesson POST should auto-create exactly one ModuleItem (§A5)"

    # 3c. Teacher attaches a lesson resource.
    resource_response = await client.post(
        f"/api/v1/teacher/lessons/{lesson_id}/resources",
        json={
            "lesson_id": str(lesson_id),
            "title": "Slides PDF",
            "resource_type": "pdf",
            "storage_object_id": str(storage_object),
            "position": 1,
            "visible_to_students": True,
        },
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert resource_response.status_code == 201, resource_response.text

    # 3d. Teacher promotes module + lesson to published so they appear in the learner tree.
    publish_module = await client.patch(
        f"/api/v1/teacher/modules/{module_id}",
        json={"status": "published"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert publish_module.status_code == 200, publish_module.text
    assert publish_module.json()["status"] == "published"

    publish_lesson = await client.patch(
        f"/api/v1/teacher/lessons/{lesson_id}",
        json={"status": "published"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert publish_lesson.status_code == 200, publish_lesson.text
    assert publish_lesson.json()["status"] == "published"

    # 4. Manager publishes the course.
    course_publish = await client.post(
        f"/api/v1/teacher/courses/{course_id}/publish",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert course_publish.status_code == 200, course_publish.text
    assert course_publish.json()["status"] == "published"

    # 5a. Student reads the course by slug.
    by_slug = await client.get(
        f"/api/v1/courses/by-slug/{new_slug}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert by_slug.status_code == 200, by_slug.text
    assert by_slug.json()["id"] == str(course_id)
    assert by_slug.json()["status"] == "published"

    # 5b. Student fetches the content tree -- the published module is present.
    content = await client.get(
        f"/api/v1/courses/{course_id}/content",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert content.status_code == 200, content.text
    tree = content.json()
    assert tree["course"]["id"] == str(course_id)
    module_ids = {m["id"] for m in tree.get("modules", [])}
    assert str(module_id) in module_ids

    # 6. Student is denied authoring.
    denied = await client.post(
        "/api/v1/teacher/courses",
        json={**create_payload, "slug": f"{new_slug}-denied"},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert denied.status_code == 403


async def test_hod_oversight_dept_courses_view(
    client: httpx.AsyncClient,
    manager_bearer: str,
    hod_bearer: str,
    seeded_users: SeededUsers,
    cleanup_courses: set[uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """HOD can list dept courses and reach drafts in their org_unit.

    Plan §4575 -- "HOD oversees: gets dept course list, sees draft work".
    Manager creates a draft course in the HOD's org_unit; the HOD's
    scope-derived ``GET /dept/courses`` returns the draft.
    """
    suffix = uuid.uuid4().hex[:8]

    create_response = await client.post(
        "/api/v1/teacher/courses",
        json={
            "org_unit_id": str(seeded_users.org_unit_id),
            "slug": f"hod-dept-{suffix}",
            "title": "HOD Oversight Course",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert create_response.status_code == 201, create_response.text
    course_id = uuid.UUID(create_response.json()["id"])
    cleanup_courses.add(course_id)

    async with engine.begin() as conn:
        status = (
            await conn.execute(
                text("SELECT status FROM courses WHERE id = :id"),
                {"id": course_id},
            )
        ).scalar_one()
    assert status == "draft", "course must be in DRAFT for the oversight assertion"

    listing = await client.get(
        "/api/v1/dept/courses",
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert listing.status_code == 200, listing.text
    ids = {row["id"] for row in listing.json()}
    assert str(course_id) in ids, "HOD scope=org_unit must include drafts in their unit"

    # HOD walks the org_unit-scoped browse path too (separate endpoint, separate
    # require_org_unit_permission factory).
    by_unit = await client.get(
        f"/api/v1/dept/org-units/{seeded_users.org_unit_id}/courses",
        headers={"Authorization": f"Bearer {hod_bearer}"},
    )
    assert by_unit.status_code == 200, by_unit.text
    unit_ids = {row["id"] for row in by_unit.json()}
    assert str(course_id) in unit_ids


async def test_admin_soft_delete_then_restore(
    client: httpx.AsyncClient,
    admin_bearer: str,
    manager_bearer: str,
    seeded_users: SeededUsers,
    cleanup_courses: set[uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """Admin sees soft-deleted courses and can restore them.

    The admin router intentionally exposes no ``DELETE`` endpoint per
    plan §4526 (no-hard-delete invariant). The legacy soft-delete path
    is the canonical recovery model: tombstone via service / DB op,
    then ``POST /admin/courses/{id}/restore`` lifts the tombstone.

    The test soft-deletes via raw SQL (closest analogue to a service
    write) so we can keep the e2e self-contained without depending on a
    yet-unbuilt Phase 7 admin DELETE surface.
    """
    suffix = uuid.uuid4().hex[:8]

    create_response = await client.post(
        "/api/v1/teacher/courses",
        json={
            "org_unit_id": str(seeded_users.org_unit_id),
            "slug": f"admin-restore-{suffix}",
            "title": "Admin Restore Course",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert create_response.status_code == 201, create_response.text
    course_id = uuid.UUID(create_response.json()["id"])
    cleanup_courses.add(course_id)

    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET deleted_at = NOW(), deleted_by = :admin WHERE id = :id"),
            {"admin": seeded_users.admin_id, "id": course_id},
        )

    listing = await client.get(
        "/api/v1/admin/courses",
        params={"limit": 100, "include_deleted": "true"},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert listing.status_code == 200, listing.text
    ids = {item["id"] for item in listing.json()["items"]}
    assert str(course_id) in ids, "soft-deleted course must surface in admin list"

    restore = await client.post(
        f"/api/v1/admin/courses/{course_id}/restore",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert restore.status_code == 200, restore.text

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT deleted_at, deleted_by, updated_by FROM courses WHERE id = :id"),
                {"id": course_id},
            )
        ).one()
    assert row.deleted_at is None
    assert row.deleted_by is None
    assert row.updated_by == seeded_users.admin_id

    listing_after = await client.get(
        "/api/v1/admin/courses",
        params={"limit": 100, "include_deleted": "false"},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert listing_after.status_code == 200, listing_after.text
    after_ids = {item["id"] for item in listing_after.json()["items"]}
    assert str(course_id) in after_ids, "restored course must reappear in active listing"


async def test_draft_module_excluded_from_student_content_tree(
    client: httpx.AsyncClient,
    manager_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    cleanup_courses: set[uuid.UUID],
    engine: AsyncEngine,
) -> None:
    """DRAFT_VISIBILITY: draft modules are filtered from learner content tree.

    Plan §4153 / Reconciliation §A9: the legacy backend leaked draft
    children to the learner endpoints. The fixed behaviour excludes
    them entirely. Manager creates a published course with one
    published module and one draft module under it; the learner content
    tree must surface only the published module.
    """
    suffix = uuid.uuid4().hex[:8]

    create_response = await client.post(
        "/api/v1/teacher/courses",
        json={
            "org_unit_id": str(seeded_users.org_unit_id),
            "slug": f"draft-module-{suffix}",
            "title": "Draft Visibility Course",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert create_response.status_code == 201, create_response.text
    course_id = uuid.UUID(create_response.json()["id"])
    cleanup_courses.add(course_id)

    pub_module_id = uuid.uuid4()
    draft_module_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) VALUES "
                "(:p, :c, 'Pub', 1, 'published'), "
                "(:d, :c, 'Draft', 2, 'draft')"
            ),
            {"p": pub_module_id, "d": draft_module_id, "c": course_id},
        )
        await conn.execute(
            text("UPDATE courses SET status = 'published' WHERE id = :id"),
            {"id": course_id},
        )

    content = await client.get(
        f"/api/v1/courses/{course_id}/content",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert content.status_code == 200, content.text
    module_ids = {m["id"] for m in content.json().get("modules", [])}
    assert str(pub_module_id) in module_ids
    assert str(draft_module_id) not in module_ids, (
        "DRAFT_VISIBILITY: learner content tree must exclude draft modules"
    )
