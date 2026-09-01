"""Phase 7 cross-feature integration suite (T7.6).

Closes Phase 7 by exercising the full ``create_app()`` factory across
three cross-feature scenarios. Mocks LLMs and ARQ at the boundaries so
the suite stays deterministic and offline.

Scenarios:

1. ``test_enroll_progress_notify_lifecycle`` -- Manager bulk-enrolls a
   student; the test additionally dispatches a course-announcement
   notification via the locked T7.4 cross-feature surface
   (``services.dispatch.send_notification``) and asserts the row landed
   in the student's inbox. Then the student records a material
   engagement and the lesson-progress aggregator advances the
   completion percent past zero.
2. ``test_career_path_lifecycle`` -- Manager creates 2 courses, builds a
   career path, adds both courses, publishes, enrolls the student. The
   student lists their career enrollments and reads back path progress.
3. ``test_admin_org_scoped_stats`` -- Manager (org-scoped via
   ``system.stats.read``) sees only their org's overview counts; IT
   Admin (``system.administer``) sees the global counts that include a
   second seeded organisation.
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

import abridgeai.ai.models  # noqa: F401  -- register processing_jobs / generation_runs / ai_model_calls
import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.career_paths.models  # noqa: F401  -- register CareerPathCourse
import abridgeai.features.courses.models  # noqa: F401  -- register course catalog
import abridgeai.features.enrollments.models  # noqa: F401  -- register enrollments tables
import abridgeai.features.identity.models  # noqa: F401  -- register users
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* (FK targets)
import abridgeai.features.notifications.models  # noqa: F401  -- register notifications
import abridgeai.features.progress.models  # noqa: F401  -- register lesson_progress
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz tables
from abridgeai.api import create_app
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import (
    create_access_token,
    generate_token,
    hash_secret,
)
from abridgeai.features.admin.routers.processing import (
    get_arq_pool as get_admin_arq_pool,
)
from abridgeai.features.interviews.routers.authoring import (
    get_arq_pool as get_interview_authoring_arq_pool,
)
from abridgeai.features.interviews.routers.learner import (
    get_arq_pool as get_interview_learner_arq_pool,
)
from abridgeai.features.materials.routers.authoring import (
    get_arq_pool as get_materials_arq_pool,
)
from abridgeai.features.notifications.services import dispatch as notify_dispatch
from abridgeai.features.quizzes.routers.authoring import (
    get_arq_pool as get_quiz_arq_pool,
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
    """Mount the full ``create_app()`` and stub all ARQ pool deps."""

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _none_pool() -> object | None:
        return None

    fastapi_app = create_app()
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    for dep in (
        get_admin_arq_pool,
        get_materials_arq_pool,
        get_quiz_arq_pool,
        get_interview_authoring_arq_pool,
        get_interview_learner_arq_pool,
    ):
        fastapi_app.dependency_overrides[dep] = _none_pool
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(eng: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with eng.begin() as conn:
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
async def manager_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.manager_id)
    yield create_access_token(user_id=seeded_users.manager_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def manager_org_membership(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[None]:
    """Add an ``organization_memberships`` row so admin scope resolves for Manager.

    ``resolve_admin_scope`` (T7.5) prefers the ``organization_memberships``
    table; the seeded fixture only writes ``user_role_assignments``. Inserting
    here keeps the setup local to this scenario.
    """
    async with engine.begin() as conn:
        inserted = await conn.execute(
            text(
                "INSERT INTO organization_memberships (user_id, organization_id, status) "
                "VALUES (:u, :o, 'active') ON CONFLICT DO NOTHING "
                "RETURNING id"
            ),
            {"u": seeded_users.manager_id, "o": seeded_users.organization_id},
        )
        # Did WE create it? conftest seeds this membership session-wide, so
        # the INSERT is normally a no-op — and the teardown below used to
        # delete unconditionally, destroying session state every later test
        # depends on. Only remove the row if this fixture actually made one.
        created = inserted.first() is not None
    yield
    if created:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM organization_memberships "
                    "WHERE user_id = :u AND organization_id = :o"
                ),
                {"u": seeded_users.manager_id, "o": seeded_users.organization_id},
            )


def test_create_app_mounts_phase_7_routes() -> None:
    """``create_app()`` exposes routes from every Phase 7 feature."""
    app_ = create_app()
    paths = {getattr(r, "path", "") for r in app_.routes}
    expected = [
        "/api/v1/me/enrollments",
        "/api/v1/management/courses/{course_id}/enrollments/bulk",
        "/api/v1/me/progress/material-engagement",
        "/api/v1/career-paths",
        "/api/v1/me/career-enrollments",
        "/api/v1/management/career-paths",
        "/api/v1/me/notifications",
        "/api/v1/admin/stats/overview",
        "/api/v1/admin/audit/role-changes",
        "/api/v1/admin/processing/jobs",
        "/api/v1/admin/users",
    ]
    missing = [p for p in expected if p not in paths]
    assert not missing, f"Missing Phase 7 paths: {missing}"


async def test_enroll_progress_notify_lifecycle(
    client: httpx.AsyncClient,
    manager_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
) -> None:
    """Cross-feature: bulk-enroll → notify → record engagement → progress aggregates."""
    suffix = uuid.uuid4().hex[:8]
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_id = uuid.uuid4()
    material_id = uuid.uuid4()
    version_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, faculty_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :ou, :owner, :slug, :title, 'published')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "ou": seeded_users.org_unit_id,
                "owner": seeded_users.admin_id,
                "slug": f"phase7-course-{suffix}",
                "title": "Phase 7 Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Mod 1', 1, 'published')"
            ),
            {"m": module_id, "c": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, "
                "lesson_type, estimated_minutes) "
                "VALUES (:l, :m, :slug, 'Lesson 1', 'published', 'video', 10)"
            ),
            {"l": lesson_id, "m": module_id, "slug": f"phase7-lesson-{suffix}"},
        )
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, :b, :k)"),
            {"id": storage_id, "b": "test-bucket", "k": f"phase7-{suffix}.pdf"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :lid, 'Phase 7 Material', 'pdf')"
            ),
            {"id": material_id, "lid": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, is_current, processing_status) "
                "VALUES (:vid, :mid, :sid, 1, TRUE, 'ready')"
            ),
            {"vid": version_id, "mid": material_id, "sid": storage_id},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = :vid WHERE id = :id"),
            {"vid": version_id, "id": material_id},
        )

    bulk_response = await client.post(
        f"/api/v1/management/courses/{course_id}/enrollments/bulk",
        json={"user_ids": [str(seeded_users.student_id)]},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert bulk_response.status_code == 200, bulk_response.text
    body = bulk_response.json()
    assert str(seeded_users.student_id) in body["enrolled"]
    assert body["failures"] == []

    async with session_factory() as session:
        await notify_dispatch.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="course_announcement",
            title="You have been enrolled",
            body=f"Manager added you to course {course_id}",
            entity_type="course",
            entity_id=course_id,
            arq_pool=None,
        )
        await session.commit()

    list_resp = await client.get(
        "/api/v1/me/notifications?limit=50",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert list_resp.status_code == 200, list_resp.text
    inbox = list_resp.json()
    matched = [n for n in inbox if n.get("entity_id") == str(course_id)]
    assert matched, f"course_announcement notification not found in inbox: {inbox}"
    assert matched[0]["category"] == "course_announcement"

    me_enroll_resp = await client.get(
        "/api/v1/me/enrollments",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert me_enroll_resp.status_code == 200, me_enroll_resp.text
    enrolled_course_ids = {row["course_id"] for row in me_enroll_resp.json()}
    assert str(course_id) in enrolled_course_ids

    started = datetime.now(tz=UTC) - timedelta(minutes=5)
    ended = started + timedelta(minutes=4)
    engagement_resp = await client.post(
        "/api/v1/me/progress/material-engagement",
        json={
            "material_version_id": str(version_id),
            "engagement_seconds": 240,
            "scroll_position_percent": "75",
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
        },
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert engagement_resp.status_code == 201, engagement_resp.text

    progress_resp = await client.get(
        f"/api/v1/me/progress/lessons/{lesson_id}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert progress_resp.status_code == 200, progress_resp.text
    progress_body = progress_resp.json()
    assert float(progress_body["completion_percent"]) > 0
    assert progress_body["status"] in {"in_progress", "completed"}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_progress WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(
            text("DELETE FROM material_engagement WHERE material_version_id = :v"),
            {"v": version_id},
        )
        await conn.execute(
            text("DELETE FROM notifications WHERE entity_id = :c"),
            {"c": course_id},
        )
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"),
            {"c": course_id},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = NULL WHERE id = :id"),
            {"id": material_id},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :v"),
            {"v": version_id},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :m"),
            {"m": material_id},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :s"),
            {"s": storage_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})


async def test_career_path_lifecycle(
    client: httpx.AsyncClient,
    manager_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """Manager creates path + adds courses + enrolls student; student reads progress."""
    suffix = uuid.uuid4().hex[:8]
    course_a = uuid.uuid4()
    course_b = uuid.uuid4()

    async with engine.begin() as conn:
        for cid, slug, title in (
            (course_a, f"path-course-a-{suffix}", "Path Course A"),
            (course_b, f"path-course-b-{suffix}", "Path Course B"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO courses (id, organization_id, owner_user_id, "
                    "slug, title, status) "
                    "VALUES (:id, :org, :owner, :slug, :title, 'published')"
                ),
                {
                    "id": cid,
                    "org": seeded_users.organization_id,
                    "owner": seeded_users.admin_id,
                    "slug": slug,
                    "title": title,
                },
            )
            # The publish gate requires at least one gradeable unit per staged
            # course (published lesson/quiz/interview behind a live
            # module_items row) — bare courses can never be published.
            module_id = uuid.uuid4()
            lesson_id = uuid.uuid4()
            await conn.execute(
                text(
                    "INSERT INTO modules (id, course_id, title, position, status) "
                    "VALUES (:id, :cid, 'M', 1, 'published')"
                ),
                {"id": module_id, "cid": cid},
            )
            await conn.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                    "VALUES (:id, :mid, :slug, 'L', 'published', 'video')"
                ),
                {"id": lesson_id, "mid": module_id, "slug": f"{slug}-l1"},
            )
            await conn.execute(
                text(
                    "INSERT INTO module_items (id, module_id, item_type, lesson_id, position) "
                    "VALUES (gen_random_uuid(), :mid, 'lesson', :lid, 1)"
                ),
                {"mid": module_id, "lid": lesson_id},
            )

    create_resp = await client.post(
        "/api/v1/management/career-paths",
        json={
            "slug": f"phase7-path-{suffix}",
            "name": "Phase 7 Path",
            "description": "Cross-feature scenario path",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert create_resp.status_code == 201, create_resp.text
    path_id = uuid.UUID(create_resp.json()["id"])

    # Stage authoring precedes course authoring: CareerPathCourseAdd requires
    # a stage_id, so create the first stage up front.
    stage_resp = await client.post(
        f"/api/v1/management/career-paths/{path_id}/stages",
        json={"title": "Phase 7 Stage", "unlock_policy": "always"},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert stage_resp.status_code == 201, stage_resp.text
    stage_id = uuid.UUID(stage_resp.json()["id"])

    for course_id, position in ((course_a, 1), (course_b, 2)):
        add_resp = await client.post(
            f"/api/v1/management/career-paths/{path_id}/courses",
            json={
                "course_id": str(course_id),
                "stage_id": str(stage_id),
                "position": position,
                "is_required": True,
            },
            headers={"Authorization": f"Bearer {manager_bearer}"},
        )
        assert add_resp.status_code == 201, add_resp.text

    publish_resp = await client.post(
        f"/api/v1/management/career-paths/{path_id}/publish",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert publish_resp.status_code == 200, publish_resp.text
    assert publish_resp.json()["status"] == "published"

    # Direct career-path enrollment is DISABLED (the endpoint 409s with
    # `direct_career_path_enrollment_disabled`): a student reaches a path only
    # by being enrolled in a Learning Program that pins it, then selecting it.
    # So the scenario now walks the supported route — program -> publish ->
    # enroll -> select-path — which is also what produces the
    # student_career_enrollments projection the learner endpoints below read.
    faculty_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', 'Phase 7 Faculty', :code)"
            ),
            {"id": faculty_id, "org": seeded_users.organization_id, "code": f"P7-{suffix[:6]}"},
        )
        # decide/approve is dean-only, but creating and enrolling needs the
        # manager to hold manager|hod at the program's org OR faculty. The
        # seeded manager is org_UNIT-scoped (roles.yaml pins them to the
        # conftest faculty), not organization-scoped as this once assumed, so
        # they need an explicit grant on THIS faculty — plus the matching
        # active user_faculty_assignments row the org_unit branch requires.
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, org_unit_id, granted_by) "
                "SELECT gen_random_uuid(), :mgr, id, 'org_unit', :org, :faculty, :mgr "
                "FROM roles WHERE code = 'manager' AND deleted_at IS NULL"
            ),
            {
                "mgr": seeded_users.manager_id,
                "org": seeded_users.organization_id,
                "faculty": faculty_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_faculty_assignments "
                "(id, user_id, organization_id, faculty_id, status) "
                "VALUES (gen_random_uuid(), :mgr, :org, :faculty, 'active')"
            ),
            {
                "mgr": seeded_users.manager_id,
                "org": seeded_users.organization_id,
                "faculty": faculty_id,
            },
        )

    program_resp = await client.post(
        "/api/v1/management/learning-programs",
        json={
            "faculty_id": str(faculty_id),
            "slug": f"phase7-program-{suffix}",
            "name": "Phase 7 Program",
            "career_path_ids": [str(path_id)],
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert program_resp.status_code == 201, program_resp.text
    program_id = uuid.UUID(program_resp.json()["id"])

    program_publish = await client.post(
        f"/api/v1/management/learning-programs/{program_id}/publish",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert program_publish.status_code == 200, program_publish.text

    enroll_resp = await client.post(
        f"/api/v1/management/learning-programs/{program_id}/students",
        json={"student_ids": [str(seeded_users.student_id)]},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert enroll_resp.status_code == 201, enroll_resp.text
    enrollment_id = enroll_resp.json()[0]["id"]

    # The student picks the path. This is what writes the career-enrollment
    # projection (career_paths.api.public::ensure_program_path_access) that
    # /me/career-enrollments authorizes through.
    select_resp = await client.post(
        f"/api/v1/me/learning-program-enrollments/{enrollment_id}/select-path",
        json={"career_path_id": str(path_id)},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert select_resp.status_code == 200, select_resp.text

    direct_enroll = await client.post(
        f"/api/v1/management/career-paths/{path_id}/students",
        json={"student_id": str(seeded_users.student_id)},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert direct_enroll.status_code == 409, direct_enroll.text
    assert (
        "direct_career_path_enrollment_disabled"
        in direct_enroll.json()["detail"]["message"]
    )

    me_resp = await client.get(
        "/api/v1/me/career-enrollments",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert me_resp.status_code == 200, me_resp.text
    enrollments = me_resp.json()
    matched = [e for e in enrollments if e["career_path_id"] == str(path_id)]
    assert matched, f"student should see their career enrollment: {enrollments}"

    progress_resp = await client.get(
        f"/api/v1/me/career-enrollments/{path_id}/progress",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert progress_resp.status_code == 200, progress_resp.text
    progress_body = progress_resp.json()
    assert progress_body["course_count"] == 2
    assert progress_body["completed_courses"] == 0
    assert progress_body["overall_percent"] == 0

    async with engine.begin() as conn:
        # Program rows first: program_path_attempts / program_enrollments FK
        # the career path and its version (NO ACTION), and the entitlement
        # rows written by select-path FK the courses.
        await conn.execute(
            text(
                "DELETE FROM course_enrollment_entitlements WHERE source_id IN "
                "(SELECT a.id FROM program_path_attempts a JOIN program_enrollments e "
                " ON e.id = a.program_enrollment_id WHERE e.learning_program_id = :prog)"
            ),
            {"prog": program_id},
        )
        await conn.execute(
            text(
                "DELETE FROM program_path_attempts WHERE program_enrollment_id IN "
                "(SELECT id FROM program_enrollments WHERE learning_program_id = :prog)"
            ),
            {"prog": program_id},
        )
        await conn.execute(
            text("DELETE FROM program_enrollments WHERE learning_program_id = :prog"),
            {"prog": program_id},
        )
        await conn.execute(
            text(
                "DELETE FROM learning_program_version_paths WHERE program_version_id IN "
                "(SELECT id FROM learning_program_versions WHERE learning_program_id = :prog)"
            ),
            {"prog": program_id},
        )
        await conn.execute(
            text("DELETE FROM learning_program_versions WHERE learning_program_id = :prog"),
            {"prog": program_id},
        )
        await conn.execute(
            text("DELETE FROM learning_programs WHERE id = :prog"), {"prog": program_id}
        )
        await conn.execute(
            text(
                "DELETE FROM student_stage_progress WHERE enrollment_id IN "
                "(SELECT id FROM student_career_enrollments WHERE career_path_id = :p)"
            ),
            {"p": path_id},
        )
        await conn.execute(
            text("DELETE FROM student_career_enrollments WHERE career_path_id = :p"),
            {"p": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM career_course_items WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :p)"
            ),
            {"p": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM career_path_stages WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :p)"
            ),
            {"p": path_id},
        )
        await conn.execute(
            text("DELETE FROM career_paths WHERE id = :p"),
            {"p": path_id},
        )
        # The manager grants added above FK the faculty, so they go first.
        await conn.execute(
            text("DELETE FROM user_faculty_assignments WHERE faculty_id = :f"),
            {"f": faculty_id},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE org_unit_id = :f"),
            {"f": faculty_id},
        )
        await conn.execute(text("DELETE FROM org_units WHERE id = :f"), {"f": faculty_id})
        # Career enrollment auto-enrolls the student into member courses;
        # those rows FK the courses (NO ACTION) and must go first.
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = ANY(:ids)"),
            {"ids": [course_a, course_b]},
        )
        # Fixture gradeable units (module + published lesson + module_items).
        await conn.execute(
            text(
                "DELETE FROM module_items WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id = ANY(:ids))"
            ),
            {"ids": [course_a, course_b]},
        )
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id = ANY(:ids))"
            ),
            {"ids": [course_a, course_b]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE course_id = ANY(:ids)"),
            {"ids": [course_a, course_b]},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [course_a, course_b]},
        )


async def test_admin_org_scoped_stats(
    client: httpx.AsyncClient,
    manager_bearer: str,
    admin_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
    manager_org_membership: None,
) -> None:
    """Manager sees only their org's overview; IT Admin sees global counts."""
    del manager_org_membership

    suffix = uuid.uuid4().hex[:8]
    other_org = uuid.uuid4()
    other_user = uuid.uuid4()
    other_course = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Other Org', 'active')"
            ),
            {"id": other_org, "slug": f"other-{suffix}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": other_user, "email": f"other-stats-{suffix}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, "
                "slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Other Course', 'published')"
            ),
            {
                "id": other_course,
                "org": other_org,
                "u": other_user,
                "slug": f"other-stats-course-{suffix}",
            },
        )

    manager_resp = await client.get(
        "/api/v1/admin/stats/overview",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert manager_resp.status_code == 200, manager_resp.text
    manager_overview = manager_resp.json()

    admin_resp = await client.get(
        "/api/v1/admin/stats/overview",
        headers={"Authorization": f"Bearer {admin_bearer}"},
    )
    assert admin_resp.status_code == 200, admin_resp.text
    admin_overview = admin_resp.json()

    assert admin_overview["total_courses"] >= manager_overview["total_courses"] + 1, (
        f"global count should include the second-org course; got {admin_overview} vs {manager_overview}"
    )

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM courses WHERE id = :c"),
            {"c": other_course},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = :u"),
            {"u": other_user},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :o"),
            {"o": other_org},
        )
