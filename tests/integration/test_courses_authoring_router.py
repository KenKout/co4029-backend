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
        # Outcomes reference courses with ondelete=NO ACTION (they are
        # soft-deleted in the service layer, never cascaded), so they must go
        # before the course row or the teardown trips the FK and poisons every
        # later test in the session.
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = ANY(:ids)"),
            {"ids": [seeded_users.course_id, course_b]},
        )
        # Teacher assignments + the org's min-teachers override are added by
        # tests/_publish_ready; remove ONLY those (the owner staffed as CI),
        # never the seeded teacher assignment on course_a that the rest of the
        # suite depends on.
        await conn.execute(
            text(
                "DELETE FROM user_role_assignments "
                "WHERE scope_kind = 'course' AND course_role = 'course_instructor' "
                "AND course_id = ANY(:ids) AND user_id IN ("
                "  SELECT owner_user_id FROM courses WHERE id = ANY(:ids))"
            ),
            {"ids": [seeded_users.course_id, course_b]},
        )
        await conn.execute(
            text("DELETE FROM system_settings WHERE organization_id = :o"),
            {"o": seeded_users.organization_id},
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
        ("/teacher/modules/{module_id}/lessons", ("GET",)),
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
    """Seeded teacher has scope=course on Course-A → course.update passes.

    ``course.update`` is the CONTENT permission: description, the study-time
    estimate and the teacher's own contact details. Course identity (title,
    slug), lifecycle (status) and delivery policy (level, caps, thumbnail,
    org_unit) are manager-owned and 403 even with valid course scope.
    """
    ok = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"description": "Teacher-authored description"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["description"] == "Teacher-authored description"

    blocked = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "Teacher Patched A"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert blocked.status_code == 403, blocked.text
    assert blocked.json()["detail"]["error"] == "permission_denied"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("description", "teacher may write this"),
        ("estimated_minutes", 42),
        ("contact_email", "teach@example.test"),
        ("contact_phone", "+84900000000"),
        ("contact_website_url", "https://example.test/course"),
        ("contact_social_url", "https://example.test/social"),
    ],
)
async def test_teacher_may_patch_content_fields(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    field: str,
    value: object,
) -> None:
    """The six fields a teacher owns (user decision 2026-08-06)."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={field: value},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()[field] == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "Renamed by teacher"),
        ("slug", "teacher-renamed-slug"),
        # The hole this closes: `status` rode in on course.update, so a teacher
        # could publish their own course and skip the manager gate entirely.
        ("status", "published"),
        ("thumbnail_object_id", "00000000-0000-0000-0000-000000000001"),
    ],
)
async def test_teacher_cannot_patch_manager_owned_fields(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    field: str,
    value: object,
) -> None:
    """Identity, lifecycle and delivery policy are manager-owned."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={field: value},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403, f"{field} must be manager-only: {response.text}"
    assert response.json()["detail"]["error"] == "permission_denied"


async def test_teacher_cannot_publish_via_mixed_patch(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """A payload mixing an allowed field with `status` must reject WHOLLY.

    Otherwise the description lands and the publish is silently dropped (or
    worse, applied) — the check runs before the patch so it is all-or-nothing.
    """
    before = await _course_status(engine, scenario["course_a"])
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"description": "smuggled", "status": "published"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403, response.text
    assert await _course_status(engine, scenario["course_a"]) == before


async def test_manager_may_patch_status(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """The other side of the boundary: a manager still owns lifecycle."""
    response = await client.patch(
        f"/api/v1/teacher/courses/{scenario['course_a']}",
        json={"title": "Manager-renamed"},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["title"] == "Manager-renamed"


async def _course_status(engine: AsyncEngine, course_id: object) -> str:
    async with engine.begin() as conn:
        return (
            await conn.execute(
                text("SELECT status FROM courses WHERE id = :id"), {"id": course_id}
            )
        ).scalar_one()


async def test_teacher_cannot_upload_a_thumbnail(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """The thumbnail's side door.

    `thumbnail_object_id` is manager-only in the PATCH allow-list, but the
    upload endpoint pointed the course at a new image itself — gated on
    course.update it would have let a teacher change the artwork anyway.
    """
    response = await client.put(
        f"/api/v1/teacher/courses/{scenario['course_a']}/thumbnail",
        content=b"\x89PNG\r\n\x1a\n" + b"0" * 64,
        headers={
            "Authorization": f"Bearer {teacher_bearer}",
            "Content-Type": "image/png",
        },
    )
    assert response.status_code == 403, response.text


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


async def _publish_ready(engine: AsyncEngine, course_id: object) -> None:
    """Satisfy BOTH publish gates so a course can go live.

    Publishing is gated on (a) at least one gradeable unit — published lesson,
    quiz or interview, because a course with nothing to grade can never be
    completed by a student, and (b) at least one learning outcome, because a
    course that never says what it teaches should not be offered.

    The scenario fixture seeds only DRAFT lessons and no outcomes, so any test
    that publishes has to supply both. That mirrors the real flow: content is
    authored and outcomes are stated before the course goes live.
    """
    async with engine.begin() as conn:
        org_id, owner_id = (
            await conn.execute(
                text(
                    "SELECT organization_id, owner_user_id FROM courses WHERE id = :cid"
                ),
                {"cid": course_id},
            )
        ).one()
        # Idempotent: several tests call this helper against the same course.
        # Pin the org's teacher-minimum to 1 and staff the course owner as
        # Course Instructor so these publish tests stay focused on the
        # content/outcome gates (the staffing gate lives in the assignment
        # router tests).
        await conn.execute(
            text(
                "DELETE FROM system_settings WHERE organization_id = :o "
                "AND setting_key = 'courses.min_teachers_per_course'"
            ),
            {"o": org_id},
        )
        await conn.execute(
            text(
                "INSERT INTO system_settings (organization_id, setting_key, "
                "setting_value_json) VALUES (:o, 'courses.min_teachers_per_course', '1')"
            ),
            {"o": org_id},
        )
        # The SPA/admin path invalidates the resolver cache on write; the raw
        # SQL insert here bypasses it, and a prior test may have cached this
        # org's settings (TTL 30s), so clear it so publish resolves min=1.
        from abridgeai.core.runtime_settings import invalidate_settings_cache  # noqa: PLC0415

        invalidate_settings_cache()
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, course_id, "
                "granted_by, course_role) "
                "SELECT gen_random_uuid(), :owner, r.id, 'course', :org, :cid, :owner, "
                "'course_instructor' FROM roles r WHERE r.code = 'teacher' "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM user_role_assignments WHERE course_id = :cid "
                "  AND scope_kind = 'course' AND deleted_at IS NULL "
                "  AND active_until IS NULL)"
            ),
            {"owner": owner_id, "org": org_id, "cid": course_id},
        )
        await conn.execute(
            text(
                "UPDATE lessons SET status = 'published' WHERE id = ("
                "  SELECT l.id FROM lessons l JOIN modules m ON m.id = l.module_id"
                "  WHERE m.course_id = :cid LIMIT 1)"
            ),
            {"cid": course_id},
        )
        # Idempotent: several tests call this helper against the same course.
        await conn.execute(
            text(
                "INSERT INTO course_learning_outcomes "
                "(id, course_id, position, outcome_text) "
                "SELECT gen_random_uuid(), :cid, 1, 'Publish-gate outcome' "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM course_learning_outcomes "
                "  WHERE course_id = :cid AND deleted_at IS NULL)"
            ),
            {"cid": course_id},
        )


async def test_publish_course_widens_status(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    await _publish_ready(engine, scenario["course_b"])
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


async def test_list_module_lessons_for_authoring_includes_drafts(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """Authoring GET returns drafts (sibling of learner endpoint that publishes-only filters).

    FR-5 quiz panel needs the full list when teacher is building a quiz on
    a yet-unpublished module — see ``courses.routers.authoring``.
    """
    response = await client.get(
        f"/api/v1/teacher/modules/{scenario['module_a']}/lessons",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    # The scenario fixture provisions draft lessons; learner endpoint would
    # filter them out, authoring must surface them.
    assert len(body) >= 1
    for lesson in body:
        assert lesson["module_id"] == str(scenario["module_a"])


async def test_list_module_lessons_for_authoring_unknown_module_returns_404(
    client: httpx.AsyncClient,
    manager_bearer: str,
) -> None:
    response = await client.get(
        f"/api/v1/teacher/modules/{uuid.uuid4()}/lessons",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 404


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
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE course_id = :id"),
                {"id": body["id"]},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": body["id"]})


async def test_manager_creating_a_course_is_not_auto_assigned_as_teacher(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
) -> None:
    """A manager creating a course on a teacher's behalf must NOT become a
    co-teacher on it.

    Create used to auto-assign the creator unconditionally. Since assignment is
    purely additive (``assign_teacher_to_course`` returns early when a record
    exists and never removes anyone), every course a manager ever created kept
    them in its teacher list permanently — cluttering their authoring list and
    the dept teachers tab. The manager keeps ownership and
    ``course.delete``/``course.publish``; neither needs a teacher row.
    """
    suffix = uuid.uuid4().hex[:8]
    response = await client.post(
        "/api/v1/teacher/courses",
        json={"slug": f"mgr-noassign-{suffix}", "title": f"Mgr {suffix}"},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 201, response.text
    course_id = response.json()["id"]
    try:
        async with engine.begin() as conn:
            count = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM user_role_assignments "
                        "WHERE course_id = :id AND deleted_at IS NULL"
                    ),
                    {"id": course_id},
                )
            ).scalar_one()
        assert count == 0, "manager must not be auto-assigned as a teacher"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE course_id = :id"),
                {"id": course_id},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})


async def test_teacher_cannot_create_a_course_at_all(
    client: httpx.AsyncClient,
    teacher_bearer: str,
) -> None:
    """Recorded because it is why the auto-assign was always wrong.

    ``course.create`` is held by admin and manager only, so a teacher can never
    reach create. The "reasonable for a teacher self-creating" justification for
    auto-assigning the creator therefore described a case that cannot happen:
    in practice the creator was always a manager or admin.
    """
    response = await client.post(
        "/api/v1/teacher/courses",
        json={"slug": f"tea-denied-{uuid.uuid4().hex[:8]}", "title": "Denied"},
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["required"] == ["course.create"]


async def test_creator_holding_the_teacher_role_is_auto_assigned(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """The surviving half of the rule: a creator who genuinely IS a teacher
    still gets the row, so a dual-role user's own course stays in their
    authoring list. Guards against "just delete the auto-assign" regressing
    the case the feature was built for.
    """
    role_assignment_id = uuid.uuid4()
    async with engine.begin() as conn:
        teacher_role_id = (
            await conn.execute(text("SELECT id FROM roles WHERE code = 'teacher'"))
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, granted_by) "
                "VALUES (:id, :uid, :rid, 'organization', :org, :uid)"
            ),
            {
                "id": role_assignment_id,
                "uid": seeded_users.manager_id,
                "rid": teacher_role_id,
                "org": seeded_users.organization_id,
            },
        )
    course_id: str | None = None
    try:
        response = await client.post(
            "/api/v1/teacher/courses",
            json={"slug": f"dual-{uuid.uuid4().hex[:8]}", "title": "Dual role"},
            headers={"Authorization": f"Bearer {manager_bearer}"},
        )
        assert response.status_code == 201, response.text
        course_id = response.json()["id"]
        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT user_id FROM user_role_assignments "
                        "WHERE course_id = :id AND deleted_at IS NULL"
                    ),
                    {"id": course_id},
                )
            ).all()
        assert [r.user_id for r in rows] == [seeded_users.manager_id]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE id = :id"),
                {"id": role_assignment_id},
            )
            if course_id is not None:
                await conn.execute(
                    text("DELETE FROM user_role_assignments WHERE course_id = :id"),
                    {"id": course_id},
                )
                await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})


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
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE course_id = :id"),
                {"id": created_id},
            )
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
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE course_id = :id"),
                {"id": created_id},
            )
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
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID | str],
    seeded_users: SeededUsers,
) -> None:
    """``GET /teacher/courses/{id}/roster`` returns the ``{course_id,
    students: [...]}`` envelope the SPA's ``useTeacherCourseRoster`` hook
    actually expects (progress/risk fields included), not the flat
    ``RosterEntry`` list the ``/dept`` HOD-scope roster uses.

    Regression: the endpoint used to return ``list[RosterEntry]`` (no
    envelope, no progress/risk) while the frontend read
    ``roster?.students``, which is ``undefined`` on a bare array — the
    Students page silently rendered "no students enrolled" for every
    course that actually had students.
    """
    enrollment_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status, source) "
                "VALUES (:id, :course, :student, 'active', 'manual')"
            ),
            {
                "id": enrollment_id,
                "course": scenario["course_a"],
                "student": seeded_users.student_id,
            },
        )
    try:
        response = await client.get(
            f"/api/v1/teacher/courses/{scenario['course_a']}/roster",
            headers={"Authorization": f"Bearer {admin_bearer}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["course_id"] == str(scenario["course_a"])
        assert isinstance(body["students"], list)
        assert len(body["students"]) == 1
        row = body["students"][0]
        assert row["enrollment_id"] == str(enrollment_id)
        assert row["student_id"] == str(seeded_users.student_id)
        assert row["enrollment_status"] == "active"
        assert "primary_email" in row
        assert "progress_percent" in row
        assert row["at_risk_level"] in ("none", "low", "medium", "high")
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM course_enrollments WHERE id = :id"),
                {"id": enrollment_id},
            )


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


async def test_get_lesson_outline_accepts_grouping_query_params(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """``GET /lessons/{id}/outline`` honors ``slides_per_section`` and
    ``section_grouping`` query params.

    The synthetic-section fallback (no chunks ingested for ``lesson_a``)
    doesn't exercise the grouping logic itself, but the endpoint must
    accept the params without 422 — that's what the SPA dropdown wires
    through. End-to-end grouping behaviour is covered by
    ``test_quiz_outline.py``.
    """
    for grouping in ("auto", "fixed"):
        for size in (1, 4, 16):
            response = await client.get(
                f"/api/v1/teacher/lessons/{scenario['lesson_a']}/outline",
                params={
                    "slides_per_section": size,
                    "section_grouping": grouping,
                },
                headers={"Authorization": f"Bearer {admin_bearer}"},
            )
            assert response.status_code == 200, (
                f"failed for grouping={grouping} size={size}: {response.text}"
            )

    # Out-of-range slides_per_section is rejected with 422 (Field ge=1, le=20).
    response = await client.get(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}/outline",
        params={"slides_per_section": 0},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 422, response.text

    # Unknown grouping value rejected with 422 (Literal validation).
    response = await client.get(
        f"/api/v1/teacher/lessons/{scenario['lesson_a']}/outline",
        params={"section_grouping": "invalid"},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_course_outcome_crud_and_reindex_on_delete(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """LO CRUD (§LO-1/2): create appends, delete renumbers to a 1..N chain.

    admin owns course_a, so it holds ``course.update`` there. Creates four
    outcomes (positions 1..4), deletes the 2nd, and asserts the survivors
    compact to contiguous positions 1,2,3 in their original relative order —
    the strict-chain requirement for the ``L.O.x`` display codes.
    """
    course_a = scenario["course_a"]
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    base = f"/api/v1/teacher/courses/{course_a}/outcomes"

    created: list[dict[str, object]] = []
    for text_val in ("Understand X", "Apply Y", "Analyze Z", "Evaluate W"):
        resp = await client.post(base, json={"outcome_text": text_val}, headers=auth)
        assert resp.status_code == 201, resp.text
        created.append(resp.json())

    # Positions are server-assigned 1..4 in creation order.
    assert [o["position"] for o in created] == [1, 2, 3, 4]

    # List returns them ordered by position.
    list_resp = await client.get(base, headers=auth)
    assert list_resp.status_code == 200, list_resp.text
    listed = list_resp.json()
    assert [o["outcome_text"] for o in listed] == [
        "Understand X",
        "Apply Y",
        "Analyze Z",
        "Evaluate W",
    ]

    # Edit the third outcome's text.
    third_id = created[2]["id"]
    patch_resp = await client.patch(
        f"{base}/{third_id}",
        json={"outcome_text": "Analyze Z (revised)"},
        headers=auth,
    )
    assert patch_resp.status_code == 200, patch_resp.text
    assert patch_resp.json()["outcome_text"] == "Analyze Z (revised)"

    # Delete the SECOND outcome ("Apply Y").
    second_id = created[1]["id"]
    del_resp = await client.delete(f"{base}/{second_id}", headers=auth)
    assert del_resp.status_code == 204, del_resp.text

    # Survivors renumber to a contiguous 1..N chain, order preserved.
    after = (await client.get(base, headers=auth)).json()
    assert [(o["position"], o["outcome_text"]) for o in after] == [
        (1, "Understand X"),
        (2, "Analyze Z (revised)"),
        (3, "Evaluate W"),
    ]

    # Cleanup: HARD-delete the rows (the DELETE endpoint only soft-deletes,
    # which would leave course_learning_outcomes rows referencing the shared
    # seeded course and trip the conftest course teardown's FK on hard-DELETE).
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
            {"cid": course_a},
        )


@pytest.mark.asyncio
async def test_course_outcome_student_403(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict[str, uuid.UUID | str],
) -> None:
    """A student lacks ``course.update`` — outcome writes return 403."""
    course_a = scenario["course_a"]
    auth = {"Authorization": f"Bearer {student_bearer}"}
    resp = await client.post(
        f"/api/v1/teacher/courses/{course_a}/outcomes",
        json={"outcome_text": "Sneaky"},
        headers=auth,
    )
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_course_outcome_teacher_403_lo_manage_split(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """Teacher (course.update, NOT learning_outcome.manage) cannot author LOs.

    The §LO split: learning outcomes are manager-owned. The teacher holds a
    course-scoped ``teacher`` role on course_a (so ``course.update`` for
    content), but LO create/edit/delete now require ``learning_outcome.manage``
    — which the teacher lacks — so all three writes return 403. This is the
    regression guard for the split.
    """
    course_a = scenario["course_a"]
    auth = {"Authorization": f"Bearer {teacher_bearer}"}
    base = f"/api/v1/teacher/courses/{course_a}/outcomes"

    # Seed one LO directly so PATCH/DELETE have a target (bypass the API since
    # the whole point is that the teacher can't create one).
    outcome_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO course_learning_outcomes "
                "(id, course_id, outcome_text, position, created_at, updated_at) "
                "VALUES (:id, :cid, 'Seeded LO', 1, NOW(), NOW())"
            ),
            {"id": outcome_id, "cid": course_a},
        )

    try:
        create = await client.post(base, json={"outcome_text": "Teacher LO"}, headers=auth)
        assert create.status_code == 403, create.text

        patch = await client.patch(
            f"{base}/{outcome_id}", json={"outcome_text": "edited"}, headers=auth
        )
        assert patch.status_code == 403, patch.text

        delete = await client.delete(f"{base}/{outcome_id}", headers=auth)
        assert delete.status_code == 403, delete.text

        # The teacher CAN still read the LOs (content alignment needs it).
        listing = await client.get(base, headers=auth)
        assert listing.status_code == 200, listing.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
                {"cid": course_a},
            )


@pytest.mark.asyncio
async def test_course_outcome_manager_can_author(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """Manager (org-scoped learning_outcome.manage) can fully author LOs.

    Exercised on course_a, which the manager does NOT own (admin owns it) — so
    this proves authorisation flows through the ``learning_outcome.manage``
    grant, not ownership.
    """
    course_a = scenario["course_a"]
    auth = {"Authorization": f"Bearer {manager_bearer}"}
    base = f"/api/v1/teacher/courses/{course_a}/outcomes"

    try:
        create = await client.post(base, json={"outcome_text": "Manager LO"}, headers=auth)
        assert create.status_code == 201, create.text
        outcome_id = create.json()["id"]

        patch = await client.patch(
            f"{base}/{outcome_id}", json={"outcome_text": "Manager LO v2"}, headers=auth
        )
        assert patch.status_code == 200, patch.text
        assert patch.json()["outcome_text"] == "Manager LO v2"

        delete = await client.delete(f"{base}/{outcome_id}", headers=auth)
        assert delete.status_code == 204, delete.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
                {"cid": course_a},
            )


@pytest.mark.asyncio
async def test_course_outcome_owner_teacher_cannot_bypass(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """A teacher who OWNS a course still cannot author its LOs.

    The owner short-circuit is disabled for LO endpoints (allow_owner=False), so
    even course ownership does not confer LO authoring — only the manager's
    ``learning_outcome.manage`` grant does. This is the interaction that a naive
    'just regate the endpoint' fix would miss.
    """
    owned_course = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Teacher Owned', 'draft')"
            ),
            {
                "id": owned_course,
                "org": seeded_users.organization_id,
                "owner": seeded_users.teacher_id,
                "slug": f"teacher-owned-{owned_course.hex[:8]}",
            },
        )

    auth = {"Authorization": f"Bearer {teacher_bearer}"}
    base = f"/api/v1/teacher/courses/{owned_course}/outcomes"
    try:
        # Sanity: the teacher CAN edit the course itself (course.update via
        # ownership — content fields like description), proving the 403 below
        # is specifically the LO gate. (Title/slug are manager-owned identity,
        # so this sanity check uses a content field.)
        meta = await client.patch(
            f"/api/v1/teacher/courses/{owned_course}",
            json={"description": "Owner-authored description"},
            headers=auth,
        )
        assert meta.status_code == 200, meta.text

        create = await client.post(base, json={"outcome_text": "Owner LO"}, headers=auth)
        assert create.status_code == 403, create.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
                {"cid": owned_course},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": owned_course})


async def test_course_outcome_hierarchy(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """Hierarchy (arbitrary depth): nest, dotted codes, re-parent, cycle guard,
    and subtree delete.

    Builds L.O.1 with children 1.1 and 1.2, and a grandchild 1.1.1; verifies
    the derived dotted ``code`` + ``depth``; re-parents; rejects a cycle; and
    confirms deleting a parent removes its whole subtree and compacts siblings.
    """
    course_a = scenario["course_a"]
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    base = f"/api/v1/teacher/courses/{course_a}/outcomes"

    # Two top-level outcomes.
    lo1 = (await client.post(base, json={"outcome_text": "Root one"}, headers=auth)).json()
    lo2 = (await client.post(base, json={"outcome_text": "Root two"}, headers=auth)).json()
    assert lo1["code"] == "1" and lo1["depth"] == 0
    assert lo2["code"] == "2" and lo2["depth"] == 0

    # Two children under L.O.1, then a grandchild under the first child.
    c1 = (
        await client.post(
            base, json={"outcome_text": "Child A", "parent_id": lo1["id"]}, headers=auth
        )
    ).json()
    c2 = (
        await client.post(
            base, json={"outcome_text": "Child B", "parent_id": lo1["id"]}, headers=auth
        )
    ).json()
    g1 = (
        await client.post(
            base, json={"outcome_text": "Grandchild", "parent_id": c1["id"]}, headers=auth
        )
    ).json()
    assert c1["code"] == "1.1" and c1["depth"] == 1
    assert c2["code"] == "1.2" and c2["depth"] == 1
    assert g1["code"] == "1.1.1" and g1["depth"] == 2

    # List is in tree order with codes.
    listed = (await client.get(base, headers=auth)).json()
    assert [(o["code"], o["outcome_text"]) for o in listed] == [
        ("1", "Root one"),
        ("1.1", "Child A"),
        ("1.1.1", "Grandchild"),
        ("1.2", "Child B"),
        ("2", "Root two"),
    ]

    # Cycle guard: moving L.O.1 under its own grandchild must 4xx.
    cycle = await client.patch(f"{base}/{lo1['id']}", json={"parent_id": g1["id"]}, headers=auth)
    assert cycle.status_code == 400, cycle.text

    # Re-parent Child B (1.2) to be a child of Root two → becomes 2.1.
    moved = await client.patch(f"{base}/{c2['id']}", json={"parent_id": lo2["id"]}, headers=auth)
    assert moved.status_code == 200, moved.text
    after_move = {o["id"]: o for o in (await client.get(base, headers=auth)).json()}
    assert after_move[c2["id"]]["code"] == "2.1"

    # Delete L.O.1 → its subtree (Child A + Grandchild) goes too; Root two
    # (and its new child) survive, and Root two compacts to code "1".
    del_resp = await client.delete(f"{base}/{lo1['id']}", headers=auth)
    assert del_resp.status_code == 204, del_resp.text
    remaining = (await client.get(base, headers=auth)).json()
    texts = {o["outcome_text"] for o in remaining}
    assert texts == {"Root two", "Child B"}
    by_text = {o["outcome_text"]: o for o in remaining}
    assert by_text["Root two"]["code"] == "1"
    assert by_text["Child B"]["code"] == "1.1"

    # Cleanup (hard-delete; endpoint only soft-deletes).
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
            {"cid": course_a},
        )


@pytest.mark.asyncio
async def test_course_outcomes_frozen_once_published(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """LOs are editable only while the course is a draft.

    Outcomes double as the graded assessment scale, so once a course is
    published they are frozen: create/update/delete each return 409. And a
    published course can never be reverted to draft (also 409) — publishing is
    a one-way door. Edits are allowed again... never, by design.
    """
    course_a = scenario["course_a"]
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    base = f"/api/v1/teacher/courses/{course_a}/outcomes"

    # While DRAFT: create one outcome (allowed) so we have an id to edit/delete.
    created = await client.post(base, json={"outcome_text": "Understand X"}, headers=auth)
    assert created.status_code == 201, created.text
    outcome_id = created.json()["id"]

    # Publish the course (draft -> published). Needs a gradeable unit first:
    # publishing is gated on one, and the fixture's lessons are drafts.
    await _publish_ready(engine, course_a)
    pub = await client.post(f"/api/v1/teacher/courses/{course_a}/publish", headers=auth)
    assert pub.status_code == 200, pub.text
    assert pub.json()["status"] == "published"

    # Now every LO write is frozen with 409.
    add_after = await client.post(base, json={"outcome_text": "Apply Y"}, headers=auth)
    assert add_after.status_code == 409, add_after.text

    edit_after = await client.patch(
        f"{base}/{outcome_id}", json={"outcome_text": "revised"}, headers=auth
    )
    assert edit_after.status_code == 409, edit_after.text

    del_after = await client.delete(f"{base}/{outcome_id}", headers=auth)
    assert del_after.status_code == 409, del_after.text

    # The single outcome is untouched (still one, original text).
    listed = (await client.get(base, headers=auth)).json()
    assert [o["outcome_text"] for o in listed] == ["Understand X"]

    # A published course can never be reverted to draft.
    revert = await client.patch(
        f"/api/v1/teacher/courses/{course_a}", json={"status": "draft"}, headers=auth
    )
    assert revert.status_code == 409, revert.text
    still = (await client.get(f"/api/v1/teacher/courses/{course_a}", headers=auth)).json()
    assert still["status"] == "published"

    # Cleanup (hard-delete; endpoint only soft-deletes).
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
            {"cid": course_a},
        )


async def test_teacher_with_course_scope_cannot_delete_course(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """Course deletion is manager-owned: a teacher holding ``course.update``
    scope on the course (not the owner) gets 403 even though they can author
    its content (FIX: ``allow_owner=False`` on ``course.delete``)."""
    auth = {"Authorization": f"Bearer {teacher_bearer}"}
    resp = await client.delete(f"/api/v1/teacher/courses/{scenario['course_a']}", headers=auth)
    assert resp.status_code == 403, resp.text

    # The course is still there (soft-delete never ran).
    detail = await client.get(f"/api/v1/teacher/courses/{scenario['course_a']}", headers=auth)
    assert detail.status_code == 200, detail.text
    assert detail.json()["id"] == str(scenario["course_a"])


async def test_manager_can_delete_course(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, uuid.UUID | str],
    engine: AsyncEngine,
) -> None:
    """A manager holding ``course.delete`` at org scope CAN soft-delete the
    course via the teacher route (same permission gate, no owner needed)."""
    auth = {"Authorization": f"Bearer {manager_bearer}"}
    resp = await client.delete(f"/api/v1/teacher/courses/{scenario['course_a']}", headers=auth)
    assert resp.status_code == 204, resp.text

    # Soft-delete tombstone applied.
    async with engine.begin() as conn:
        deleted_at = (
            await conn.execute(
                text("SELECT deleted_at FROM courses WHERE id = :cid"),
                {"cid": scenario["course_a"]},
            )
        ).scalar_one()
    assert deleted_at is not None

    # Restore for the rest of the suite (test isolation).
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET deleted_at = NULL, deleted_by = NULL WHERE id = :cid"),
            {"cid": scenario["course_a"]},
        )


async def test_creator_holding_the_teacher_role_gets_a_notification(
    client: httpx.AsyncClient,
    admin_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Creating a course auto-assigns a teacher-creator — and must notify them.

    The regression: only the explicit ``POST /dept/courses/{id}/teachers``
    route notified. ``create_course`` writes an identical teacher assignment
    when the creator holds the teacher role, and said nothing — so whether a
    teacher heard about a course they now own depended on which code path
    happened to create the row.

    Driven with the ADMIN token on purpose. ``course.create`` is not a teacher
    permission, so a pure teacher cannot reach this path at all; the real
    scenario is an account holding BOTH roles (admin/manager to create,
    teacher to be auto-assigned), which is exactly how it was reported.
    """
    slug = f"notify-create-{uuid.uuid4().hex[:8]}"
    # Give the admin the teacher role for this test — that dual-role account is
    # the only way the auto-assign branch is reachable. The org membership goes
    # with it: `create_course` resolves the owner's primary organization from
    # the token, and the seeded admin has none.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(user_id, organization_id, status) VALUES (:uid, :org, 'active') "
                "ON CONFLICT DO NOTHING"
            ),
            {"uid": seeded_users.admin_id, "org": seeded_users.organization_id},
        )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id, active_from, granted_by) "
                "SELECT :uid, r.id, 'organization', :org, now(), :uid "
                "FROM roles r WHERE r.code = 'teacher'"
            ),
            {"uid": seeded_users.admin_id, "org": seeded_users.organization_id},
        )

    response = await client.post(
        "/api/v1/teacher/courses",
        json={"title": "Notified On Create", "slug": slug},
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert response.status_code == 201, response.text
    course_id = uuid.UUID(response.json()["id"])

    try:
        async with engine.begin() as conn:
            # The assignment exists...
            assigned = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM user_role_assignments ura "
                        "JOIN roles r ON r.id = ura.role_id "
                        "WHERE ura.course_id = :cid AND ura.user_id = :uid "
                        "AND r.code = 'teacher' AND ura.deleted_at IS NULL"
                    ),
                    {"cid": course_id, "uid": seeded_users.admin_id},
                )
            ).scalar_one()
            assert assigned == 1, "creator holding the teacher role is auto-assigned"

            # ...and so does the notification telling them about it.
            rows = (
                await conn.execute(
                    text(
                        "SELECT category, title FROM notifications "
                        "WHERE user_id = :uid AND entity_id = :cid"
                    ),
                    {"uid": seeded_users.admin_id, "cid": course_id},
                )
            ).all()
        assert len(rows) == 1, f"expected exactly one notification, got {rows}"
        assert "Notified On Create" in rows[0][1]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM notifications WHERE entity_id = :cid"), {"cid": course_id}
            )
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE course_id = :cid"),
                {"cid": course_id},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :cid"), {"cid": course_id})
            # Drop the teacher grant added for this test so the shared admin
            # fixture is left exactly as found.
            await conn.execute(
                text(
                    "DELETE FROM user_role_assignments ura USING roles r "
                    "WHERE ura.role_id = r.id AND r.code = 'teacher' "
                    "AND ura.user_id = :uid AND ura.scope_kind = 'organization'"
                ),
                {"uid": seeded_users.admin_id},
            )
            await conn.execute(
                text(
                    "DELETE FROM organization_memberships "
                    "WHERE user_id = :uid AND organization_id = :org"
                ),
                {"uid": seeded_users.admin_id, "org": seeded_users.organization_id},
            )


async def test_manager_creating_a_course_is_not_auto_assigned_or_notified(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """A manager is not a teacher, so there is no assignment and nothing to say.

    Pins the other half: the notification follows the ASSIGNMENT, not the act
    of creating. A manager creating on someone's behalf must not accumulate as
    a co-teacher, and must not be told they were handed teaching work.
    """
    slug = f"notify-mgr-{uuid.uuid4().hex[:8]}"
    response = await client.post(
        "/api/v1/teacher/courses",
        json={"title": "Manager Created", "slug": slug},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 201, response.text
    course_id = uuid.UUID(response.json()["id"])

    try:
        async with engine.begin() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM notifications WHERE entity_id = :cid"),
                    {"cid": course_id},
                )
            ).scalar_one()
        assert count == 0, "manager creating a course is not being assigned to teach it"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE course_id = :cid"),
                {"cid": course_id},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :cid"), {"cid": course_id})


async def _fresh_draft_course(
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> tuple[str, str]:
    """Create a throwaway draft course + its outcomes base URL.

    Inserted via SQL, not the API: the seeded admin fixture has no org
    membership, so ``POST /teacher/courses`` 400s for it. The shared seeded
    course_a also gets published (one-way door) by
    ``test_course_outcomes_frozen_once_published``, so LO mutation tests must
    not depend on it staying a draft — they create their own course instead
    and delete it in ``finally``.
    """
    course_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'LO Outliner Test', 'draft')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.admin_id,
                "slug": f"lo-out-{course_id.hex[:8]}",
            },
        )
    return str(course_id), f"/api/v1/teacher/courses/{course_id}/outcomes"


async def _delete_course_hard(engine: AsyncEngine, course_id: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :cid"), {"cid": course_id})


async def test_course_outcome_reorder_within_siblings(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """PATCH position moves an outcome to a 1-based slot among its siblings.

    The outliner's "drop between rows 3 and 4" maps to position=4. Codes are
    derived from positions at read time, so the display code follows the move
    while the UUID identity stays put.
    """
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    course_id, base = await _fresh_draft_course(engine, seeded_users)
    try:
        created: list[dict[str, object]] = []
        for text_val in ("One", "Two", "Three"):
            resp = await client.post(base, json={"outcome_text": text_val}, headers=auth)
            assert resp.status_code == 201, resp.text
            created.append(resp.json())

        # Move "Three" (position 3) to slot 1.
        three_id = created[2]["id"]
        moved = await client.patch(
            f"{base}/{three_id}", json={"position": 1}, headers=auth
        )
        assert moved.status_code == 200, moved.text
        assert moved.json()["position"] == 1
        assert moved.json()["code"] == "1"

        listed = (await client.get(base, headers=auth)).json()
        assert [o["outcome_text"] for o in listed] == ["Three", "One", "Two"]
        assert [o["position"] for o in listed] == [1, 2, 3]

        # Reparent with an explicit slot: nest "One" under "Three" at slot 1.
        one_id = created[0]["id"]
        nested = await client.patch(
            f"{base}/{one_id}",
            json={"parent_id": three_id, "position": 2},
            headers=auth,
        )
        assert nested.status_code == 200, nested.text
        assert nested.json()["parent_id"] == str(three_id)
        assert nested.json()["code"] == "1.1"
        assert nested.json()["depth"] == 1

        listed = (await client.get(base, headers=auth)).json()
        assert [(o["outcome_text"], o["code"]) for o in listed] == [
            ("Three", "1"),
            ("One", "1.1"),
            ("Two", "2"),
        ]
    finally:
        await _delete_course_hard(engine, course_id)


async def test_course_outcome_delete_promotes_children(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """promote_children=true keeps the kids, re-parenting them to the parent's
    level, instead of cascading the delete down the subtree."""
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    course_id, base = await _fresh_draft_course(engine, seeded_users)
    try:
        root = (await client.post(base, json={"outcome_text": "Root"}, headers=auth)).json()
        child_a = (
            await client.post(
                base, json={"outcome_text": "Child A", "parent_id": root["id"]}, headers=auth
            )
        ).json()
        await client.post(
            base, json={"outcome_text": "Child B", "parent_id": root["id"]}, headers=auth
        )
        await client.post(
            base,
            json={"outcome_text": "Grandchild", "parent_id": child_a["id"]},
            headers=auth,
        )
        await client.post(base, json={"outcome_text": "Sibling"}, headers=auth)

        del_resp = await client.delete(
            f"{base}/{root['id']}?promote_children=true", headers=auth
        )
        assert del_resp.status_code == 204, del_resp.text

        listed = (await client.get(base, headers=auth)).json()
        assert [(o["outcome_text"], o["code"], o["depth"]) for o in listed] == [
            ("Child A", "1", 0),
            ("Grandchild", "1.1", 1),
            ("Child B", "2", 0),
            ("Sibling", "3", 0),
        ]
    finally:
        await _delete_course_hard(engine, course_id)


async def test_course_outcome_delete_cascades_by_default(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Without promote_children the whole subtree goes (legacy behaviour)."""
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    course_id, base = await _fresh_draft_course(engine, seeded_users)
    try:
        root = (await client.post(base, json={"outcome_text": "Root"}, headers=auth)).json()
        await client.post(
            base, json={"outcome_text": "Kid", "parent_id": root["id"]}, headers=auth
        )

        del_resp = await client.delete(f"{base}/{root['id']}", headers=auth)
        assert del_resp.status_code == 204, del_resp.text
        listed = (await client.get(base, headers=auth)).json()
        assert listed == []
    finally:
        await _delete_course_hard(engine, course_id)


async def test_course_outcome_duplicate_deep_copies_subtree(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Duplicate copies the subtree with fresh ids, inserted after the original."""
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    course_id, base = await _fresh_draft_course(engine, seeded_users)
    try:
        root = (await client.post(base, json={"outcome_text": "Root"}, headers=auth)).json()
        child = (
            await client.post(
                base, json={"outcome_text": "Kid", "parent_id": root["id"]}, headers=auth
            )
        ).json()

        dup = await client.post(f"{base}/{root['id']}/duplicate", headers=auth)
        assert dup.status_code == 201, dup.text
        assert dup.json()["outcome_text"] == "Root"
        assert dup.json()["id"] != root["id"]

        listed = (await client.get(base, headers=auth)).json()
        assert [(o["outcome_text"], o["code"]) for o in listed] == [
            ("Root", "1"),
            ("Kid", "1.1"),
            ("Root", "2"),
            ("Kid", "2.1"),
        ]
        assert listed[2]["id"] == dup.json()["id"]
        assert listed[3]["id"] != child["id"]
        assert listed[3]["parent_id"] == listed[2]["id"]
    finally:
        await _delete_course_hard(engine, course_id)


async def test_course_outcome_question_count_surfaces_mappings(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """question_count tells the delete dialog exactly which outcomes have
    quiz questions mapped, so the confirmation can say "will lose mapping"."""
    auth = {"Authorization": f"Bearer {admin_bearer}"}
    course_id, base = await _fresh_draft_course(engine, seeded_users)
    quiz_id, module_id = uuid.uuid4(), uuid.uuid4()
    try:
        out = (await client.post(base, json={"outcome_text": "Mapped"}, headers=auth)).json()
        await client.post(base, json={"outcome_text": "Plain"}, headers=auth)

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO modules (id, course_id, title, position, status) "
                    "VALUES (:id, :cid, 'M', 1, 'published')"
                ),
                {"id": module_id, "cid": course_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                    "VALUES (:id, :cid, :mid, 'Q', 'published')"
                ),
                {"id": quiz_id, "cid": course_id, "mid": module_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, learning_outcome_id, "
                    "review_status) "
                    "VALUES (:id, :qid, 1, 'multiple_choice', 'P?', :loid, 'approved')"
                ),
                {"id": uuid.uuid4(), "qid": quiz_id, "loid": out["id"]},
            )

        listed = (await client.get(base, headers=auth)).json()
        by_text = {o["outcome_text"]: o["question_count"] for o in listed}
        assert by_text["Mapped"] == 1
        assert by_text["Plain"] == 0
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM quiz_questions WHERE quiz_id = :qid"),
                {"qid": quiz_id},
            )
            await conn.execute(text("DELETE FROM quizzes WHERE id = :id"), {"id": quiz_id})
            await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await _delete_course_hard(engine, course_id)
