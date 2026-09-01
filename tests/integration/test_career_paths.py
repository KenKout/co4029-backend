"""Integration tests for ``features.career_paths`` (T7.3)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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

import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.learning_programs.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.career_paths.routers import (
    authoring_management_router,
    authoring_teacher_router,
    career_paths_learner_router,
    me_career_enrollments_router,
)
# The supported student-to-path route goes through a Learning Program
# (direct career-path enrollment is disabled), so this mini-app also mounts
# the program routers for the enroll/progress tests.
from abridgeai.features.learning_programs.routers import (
    learner_router as learning_programs_learner_router,
)
from abridgeai.features.learning_programs.routers import (
    management_router as learning_programs_management_router,
)
from tests.support.db_graph import hard_delete_graph


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
    fastapi_app.include_router(career_paths_learner_router, prefix="/api/v1")
    fastapi_app.include_router(me_career_enrollments_router, prefix="/api/v1")
    fastapi_app.include_router(authoring_management_router, prefix="/api/v1")
    fastapi_app.include_router(authoring_teacher_router, prefix="/api/v1")
    fastapi_app.include_router(learning_programs_learner_router, prefix="/api/v1")
    fastapi_app.include_router(learning_programs_management_router, prefix="/api/v1")
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


async def _enroll_student_via_program(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    *,
    manager_bearer: str,
    student_bearer: str,
    path_id: uuid.UUID,
    student_id: uuid.UUID,
    suffix: str,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Route a student onto a career path through its Learning Program.

    Direct career-path enrollment is disabled (user decision: students reach
    a path only via a Learning Program), so this is the supported route —
    the same walk ``test_career_path_lifecycle`` uses: create a faculty unit,
    create + publish the program pinned to the path, enroll the student,
    then have the student select the path (which writes the
    ``student_career_enrollments`` projection via
    ``career_paths.api.public.ensure_program_path_access``).

    Returns ``(faculty_org_unit_id, program_id)`` so teardown can remove
    the rows this created.
    """
    faculty_id = uuid.uuid4()
    org_id = await org_id_of(path_id, engine)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', 'CP Enroll Faculty', :code)"
            ),
            {"id": faculty_id, "org": org_id, "code": f"cp-fac-{suffix}"},
        )
        # 0094 scoped program-authoring to the program's faculty: an operator
        # must hold manager/hod AT that faculty (with the matching affiliation
        # row), not merely at some org-unit. The seeded manager is scoped to
        # the seed faculty only, so grant him a faculty-scoped manager role +
        # affiliation for THIS throwaway faculty to operate here.
        manager_id = (
            await conn.execute(
                text("SELECT id FROM users WHERE primary_email = 'test-manager@abridgeai.local'")
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, org_unit_id) "
                "SELECT gen_random_uuid(), :uid, r.id, 'org_unit', :org, :fid "
                "FROM roles r WHERE r.code = 'manager'"
            ),
            {"uid": manager_id, "org": org_id, "fid": faculty_id},
        )
        await conn.execute(
            text(
                "INSERT INTO user_faculty_assignments "
                "(id, user_id, organization_id, faculty_id, status) "
                "VALUES (gen_random_uuid(), :uid, :org, :fid, 'active')"
            ),
            {"uid": manager_id, "org": org_id, "fid": faculty_id},
        )
        # The scenario keeps its path VERSION draft (several suites mutate it
        # directly), but Learning Program creation requires a PUBLISHED path
        # version. This test's own path instance is fully built, so promoting
        # its version is safe and only affects this fixture's copy.
        await conn.execute(
            text(
                "UPDATE career_path_versions SET status = 'published' "
                "WHERE career_path_id = :pid AND status != 'published'"
            ),
            {"pid": path_id},
        )

    program_resp = await client.post(
        "/api/v1/management/learning-programs",
        json={
            "faculty_id": str(faculty_id),
            "slug": f"cp-lp-{suffix}",
            "name": f"CP LP {suffix}",
            "career_path_ids": [str(path_id)],
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert program_resp.status_code == 201, program_resp.text
    program_id = uuid.UUID(program_resp.json()["id"])

    publish = await client.post(
        f"/api/v1/management/learning-programs/{program_id}/publish",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert publish.status_code == 200, publish.text

    enroll = await client.post(
        f"/api/v1/management/learning-programs/{program_id}/students",
        json={"student_ids": [str(student_id)]},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert enroll.status_code == 201, enroll.text
    enrollment_id = enroll.json()[0]["id"]

    select = await client.post(
        f"/api/v1/me/learning-program-enrollments/{enrollment_id}/select-path",
        json={"career_path_id": str(path_id)},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert select.status_code == 200, select.text
    return faculty_id, program_id


async def org_id_of(path_id: uuid.UUID, engine: AsyncEngine) -> uuid.UUID:
    """Organization owning ``path_id`` (the helper's sneaky dependency)."""

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT organization_id FROM career_paths WHERE id = :pid"),
                {"pid": path_id},
            )
        ).scalar_one()
    return row


async def _insert_course_with_lesson(
    engine: AsyncEngine,
    *,
    organization_id: uuid.UUID,
    owner_id: uuid.UUID,
    slug: str,
    title: str,
    status: str,
) -> tuple[uuid.UUID, uuid.UUID]:
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, :status)"
            ),
            {
                "id": course_id,
                "org": organization_id,
                "owner": owner_id,
                "slug": slug,
                "title": title,
                "status": status,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, :title, 1, :status)"
            ),
            {
                "id": module_id,
                "cid": course_id,
                "title": f"{title} module",
                "status": "published" if status == "published" else "draft",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:id, :mid, :slug, :title, :status, 'video')"
            ),
            {
                "id": lesson_id,
                "mid": module_id,
                "slug": f"{slug}-lesson-1",
                "title": f"{title} lesson 1",
                "status": "published" if status == "published" else "draft",
            },
        )
    return course_id, lesson_id


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, object]]:
    suffix = uuid.uuid4().hex[:8]
    path_slug = f"path-{suffix}"
    path_id = uuid.uuid4()
    stage_id = uuid.uuid4()

    pub_a_id, pub_a_lesson = await _insert_course_with_lesson(
        engine,
        organization_id=seeded_users.organization_id,
        owner_id=seeded_users.admin_id,
        slug=f"cp-pub-a-{suffix}",
        title="Pub A",
        status="published",
    )
    pub_b_id, pub_b_lesson = await _insert_course_with_lesson(
        engine,
        organization_id=seeded_users.organization_id,
        owner_id=seeded_users.admin_id,
        slug=f"cp-pub-b-{suffix}",
        title="Pub B",
        status="published",
    )
    draft_id, _draft_lesson = await _insert_course_with_lesson(
        engine,
        organization_id=seeded_users.organization_id,
        owner_id=seeded_users.admin_id,
        slug=f"cp-drf-{suffix}",
        title="Draft C",
        status="draft",
    )

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, :name, 'published')"
            ),
            {
                "id": path_id,
                "org": seeded_users.organization_id,
                "slug": path_slug,
                "name": f"Career path {suffix}",
            },
        )
        # Migration 0070: every course item belongs to a stage. One 'always'
        # stage reproduces the pre-stage flat-list behaviour these tests assert.
        # Gap 3 (0074): stages/items hang off a published v1.
        # NOTE: kept DRAFT — several suites mutate this path directly, and
        # published versions are frozen (the pinned promise). Path status is
        # 'published' so the path-level guards still apply.
        version_id = (
            await conn.execute(
                text(
                    "INSERT INTO career_path_versions "
                    "(id, career_path_id, version_no, status) "
                    "VALUES (gen_random_uuid(), :pid, 1, 'draft') "
                    "RETURNING id"
                ),
                {"pid": path_id},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, version_id, position, unlock_policy, enforcement) "
                "VALUES (:sid, :vid, 1, 'always', 'advisory')"
            ),
            {"sid": stage_id, "vid": version_id},
        )
        for position, course_id in enumerate([pub_a_id, pub_b_id, draft_id], start=1):
            await conn.execute(
                text(
                    "INSERT INTO career_course_items "
                    "(version_id, course_id, stage_id, position, is_required) "
                    "VALUES (:vid, :cid, :sid, :pos, TRUE)"
                ),
                {"vid": version_id, "cid": course_id, "sid": stage_id, "pos": position},
            )

    yield {
        "path_id": path_id,
        "path_slug": path_slug,
        "stage_id": stage_id,
        "pub_a_id": pub_a_id,
        "pub_b_id": pub_b_id,
        "draft_id": draft_id,
        "pub_a_lesson": pub_a_lesson,
        "pub_b_lesson": pub_b_lesson,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_progress WHERE lesson_id = ANY(:ids)"),
            {"ids": [pub_a_lesson, pub_b_lesson]},
        )
        await conn.execute(
            text(
                "DELETE FROM student_stage_progress WHERE enrollment_id IN "
                "(SELECT id FROM student_career_enrollments WHERE career_path_id = :pid)"
            ),
            {"pid": path_id},
        )
        await conn.execute(
            text("DELETE FROM student_career_enrollments WHERE career_path_id = :pid"),
            {"pid": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM course_enrollment_entitlements WHERE source_id IN "
                "(SELECT a.id FROM program_path_attempts a JOIN program_enrollments e "
                " ON e.id = a.program_enrollment_id WHERE e.learning_program_id IN "
                " (SELECT id FROM learning_programs WHERE faculty_id IN "
                "  (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM program_path_attempts WHERE program_enrollment_id IN "
                "(SELECT id FROM program_enrollments WHERE learning_program_id IN "
                " (SELECT id FROM learning_programs WHERE faculty_id IN "
                "  (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM program_enrollments WHERE learning_program_id IN "
                "(SELECT id FROM learning_programs WHERE faculty_id IN "
                " (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%'))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM learning_program_version_paths WHERE program_version_id IN "
                "(SELECT id FROM learning_program_versions WHERE learning_program_id IN "
                " (SELECT id FROM learning_programs WHERE faculty_id IN "
                "  (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM learning_program_versions WHERE learning_program_id IN "
                "(SELECT id FROM learning_programs WHERE faculty_id IN "
                " (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%'))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM learning_programs WHERE faculty_id IN "
                "(SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')"
            )
        )
        # The enroll helper grants the seeded manager a faculty-scoped manager
        # role + affiliation for each throwaway faculty it creates.
        await conn.execute(
            text(
                "DELETE FROM user_role_assignments WHERE org_unit_id IN "
                "(SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM user_faculty_assignments WHERE faculty_id IN "
                "(SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')"
            )
        )
        await conn.execute(text("DELETE FROM org_units WHERE code LIKE 'cp-fac-%'"))
        await conn.execute(
            text(
                "DELETE FROM career_course_items WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
            {"pid": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM career_path_stages WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
            {"pid": path_id},
        )
        await conn.execute(
            text("DELETE FROM career_paths WHERE id = :pid"),
            {"pid": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN (SELECT id FROM modules WHERE course_id = ANY(:cids))"
            ),
            {"cids": [pub_a_id, pub_b_id, draft_id]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE course_id = ANY(:cids)"),
            {"cids": [pub_a_id, pub_b_id, draft_id]},
        )
        # Graph-driven: career auto-enroll adds course_enrollments rows that
        # block a bare course delete (NO ACTION FKs).
        await hard_delete_graph(conn, "courses", [str(c) for c in (pub_a_id, pub_b_id, draft_id)])


def test_no_self_enroll_route_exists() -> None:
    learner_paths = {
        (r.path, tuple(sorted(r.methods)))  # type: ignore[attr-defined]
        for r in me_career_enrollments_router.routes  # type: ignore[attr-defined]
    }
    forbidden = {
        ("/me/career-enrollments", ("POST",)),
        ("/me/career-enrollments/", ("POST",)),
        ("/me/career-enrollments/{career_path_id}", ("POST",)),
    }
    assert not (forbidden & learner_paths)


async def test_no_self_enroll(client: httpx.AsyncClient, student_bearer: str) -> None:
    response = await client.post(
        "/api/v1/me/career-enrollments",
        json={"career_path_id": str(uuid.uuid4())},
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code in (404, 405)


async def test_path_filters_draft_courses(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict[str, object],
) -> None:
    response = await client.get(
        f"/api/v1/career-paths/{scenario['path_slug']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    course_ids = [c["course_id"] for c in body["courses"]]
    assert str(scenario["pub_a_id"]) in course_ids
    assert str(scenario["pub_b_id"]) in course_ids
    assert str(scenario["draft_id"]) not in course_ids
    assert len(body["courses"]) == 2


async def test_path_impact_reports_active_students_per_stage(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Gap 3 §2.1: blast radius of editing a published path.

    Two active students — one on Stage 1 (no latches), one on Stage 2
    (Stage 1 latched) — plus a dropped and a completed enrollment that must
    NOT count.
    """
    suffix = uuid.uuid4().hex[:8]
    path_id, stage1, stage2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    course_a, course_b = uuid.uuid4(), uuid.uuid4()
    on_stage1, on_stage2, dropped, completed = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )

    async with engine.begin() as conn:
        for cid, slug, title in (
            (course_a, f"impact-a-{suffix}", "Impact A"),
            (course_b, f"impact-b-{suffix}", "Impact B"),
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
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, 'Impact Path', 'published')"
            ),
            {"id": path_id, "org": seeded_users.organization_id, "slug": f"impact-{suffix}"},
        )
        version_id = (
            await conn.execute(
                text(
                    "INSERT INTO career_path_versions "
                    "(id, career_path_id, version_no, status, published_at) "
                    "VALUES (gen_random_uuid(), :pid, 1, 'published', NOW()) "
                    "RETURNING id"
                ),
                {"pid": path_id},
            )
        ).scalar_one()
        for sid, pos in ((stage1, 1), (stage2, 2)):
            await conn.execute(
                text(
                    "INSERT INTO career_path_stages "
                    "(id, version_id, position, unlock_policy, enforcement) "
                    "VALUES (:sid, :vid, :pos, 'always', 'advisory')"
                ),
                {"sid": sid, "vid": version_id, "pos": pos},
            )
        for cid, sid, pos in ((course_a, stage1, 1), (course_b, stage2, 1)):
            await conn.execute(
                text(
                    "INSERT INTO career_course_items "
                    "(version_id, course_id, stage_id, position, is_required) "
                    "VALUES (:vid, :cid, :sid, :pos, TRUE)"
                ),
                {"vid": version_id, "cid": cid, "sid": sid, "pos": pos},
            )
        for uid, status in (
            (on_stage1, "active"),
            (on_stage2, "active"),
            (dropped, "dropped"),
            (completed, "completed"),
        ):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"impact-{uid.hex[:6]}@test.local"},
            )
            await conn.execute(
                text(
                    "INSERT INTO student_career_enrollments "
                    "(id, career_path_id, version_id, student_id, status) "
                    "VALUES (gen_random_uuid(), :pid, :vid, :sid, :status)"
                ),
                {"pid": path_id, "vid": version_id, "sid": uid, "status": status},
            )
        # on_stage2 latched Stage 1 → currently walking Stage 2.
        await conn.execute(
            text(
                "INSERT INTO student_stage_progress (enrollment_id, stage_id) "
                "SELECT e.id, :sid FROM student_career_enrollments e "
                "WHERE e.career_path_id = :pid AND e.student_id = :uid"
            ),
            {"sid": stage1, "pid": path_id, "uid": on_stage2},
        )

    response = await client.get(
        f"/api/v1/management/career-paths/{path_id}/impact",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active_enrollments"] == 2
    by_position = {s["position"]: s for s in body["stages"]}
    assert set(by_position) == {1, 2}
    # Stage 1: only the no-latch student is on it; the Stage-2 student has
    # latched it, so neither counts as still-to-do... except the on-stage-1
    # student still has it ahead.
    assert by_position[1]["students_in_stage"] == 1
    assert by_position[1]["students_not_completed"] == 1
    # Stage 2: the Stage-2 student is on it; both active students still have
    # it ahead.
    assert by_position[2]["students_in_stage"] == 1
    assert by_position[2]["students_not_completed"] == 2

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM student_stage_progress WHERE enrollment_id IN "
                "(SELECT id FROM student_career_enrollments WHERE career_path_id = :pid)"
            ),
            {"pid": path_id},
        )
        await conn.execute(
            text("DELETE FROM student_career_enrollments WHERE career_path_id = :pid"),
            {"pid": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM course_enrollment_entitlements WHERE source_id IN "
                "(SELECT a.id FROM program_path_attempts a JOIN program_enrollments e "
                " ON e.id = a.program_enrollment_id WHERE e.learning_program_id IN "
                " (SELECT id FROM learning_programs WHERE faculty_id IN "
                "  (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM program_path_attempts WHERE program_enrollment_id IN "
                "(SELECT id FROM program_enrollments WHERE learning_program_id IN "
                " (SELECT id FROM learning_programs WHERE faculty_id IN "
                "  (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM program_enrollments WHERE learning_program_id IN "
                "(SELECT id FROM learning_programs WHERE faculty_id IN "
                " (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%'))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM learning_program_version_paths WHERE program_version_id IN "
                "(SELECT id FROM learning_program_versions WHERE learning_program_id IN "
                " (SELECT id FROM learning_programs WHERE faculty_id IN "
                "  (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM learning_program_versions WHERE learning_program_id IN "
                "(SELECT id FROM learning_programs WHERE faculty_id IN "
                " (SELECT id FROM org_units WHERE code LIKE 'cp-fac-%'))"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM learning_programs WHERE faculty_id IN "
                "(SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')"
            )
        )
        # The enroll helper grants the seeded manager a faculty-scoped manager
        # role + affiliation for each throwaway faculty it creates.
        await conn.execute(
            text(
                "DELETE FROM user_role_assignments WHERE org_unit_id IN "
                "(SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM user_faculty_assignments WHERE faculty_id IN "
                "(SELECT id FROM org_units WHERE code LIKE 'cp-fac-%')"
            )
        )
        await conn.execute(text("DELETE FROM org_units WHERE code LIKE 'cp-fac-%'"))
        await conn.execute(
            text(
                "DELETE FROM career_course_items WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
            {"pid": path_id},
        )
        await conn.execute(
            text(
                "DELETE FROM career_path_stages WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
            {"pid": path_id},
        )
        await conn.execute(text("DELETE FROM career_paths WHERE id = :pid"), {"pid": path_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(on_stage1), str(on_stage2), str(dropped), str(completed)]},
        )
        await hard_delete_graph(
            conn, "courses", [str(course_a), str(course_b)]
        )


async def test_manager_enroll_student(
    client: httpx.AsyncClient,
    manager_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    """A student reaches a path THROUGH its Learning Program (direct
    enrollment is disabled), which writes the ``student_career_enrollments``
    projection the learner endpoints authorize through.
    """
    direct = await client.post(
        f"/api/v1/management/career-paths/{scenario['path_id']}/students",
        json={"student_id": str(seeded_users.student_id)},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert direct.status_code == 409, direct.text
    assert "direct_career_path_enrollment_disabled" in direct.json()["detail"]["message"]

    suffix = uuid.uuid4().hex[:8]
    await _enroll_student_via_program(
        client,
        engine,
        manager_bearer=manager_bearer,
        student_bearer=student_bearer,
        path_id=scenario["path_id"],  # type: ignore[arg-type]
        student_id=seeded_users.student_id,
        suffix=suffix,
    )

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT status FROM student_career_enrollments "
                    "WHERE career_path_id = :pid AND student_id = :sid"
                ),
                {"pid": scenario["path_id"], "sid": seeded_users.student_id},
            )
        ).one_or_none()
    assert row is not None
    assert row.status == "active"


async def test_manager_enroll_teacher_rejected(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
    scenario: dict[str, object],
) -> None:
    """Direct career-path enrollment is DISABLED at the endpoint level.

    The old student-only backstop (teacher target 409s "not a student") is
    unreachable: the endpoint short-circuits with
    ``direct_career_path_enrollment_disabled`` before any role check. A
    teacher is kept off a pathway by the same gate as everyone else, and in
    the supported flow the program enrollment step validates who may join.
    """
    response = await client.post(
        f"/api/v1/management/career-paths/{scenario['path_id']}/students",
        json={"student_id": str(seeded_users.teacher_id)},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 409, response.text
    assert "direct_career_path_enrollment_disabled" in response.json()["detail"]["message"]


async def test_manager_list_includes_stage_course_counts(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
) -> None:
    """The management list enriches every row with stage/course counts.

    The scenario path has exactly one stage and three attached courses —
    the numbers the management table now shows instead of N+1 detail calls.
    """
    response = await client.get(
        "/api/v1/management/career-paths",
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text
    rows = {p["id"]: p for p in response.json()}
    row = rows[str(scenario["path_id"])]
    assert row["stage_count"] == 1
    assert row["course_count"] == 3


async def test_progress_aggregate(
    client: httpx.AsyncClient,
    manager_bearer: str,
    student_bearer: str,
    seeded_users: SeededUsers,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    await _enroll_student_via_program(
        client,
        engine,
        manager_bearer=manager_bearer,
        student_bearer=student_bearer,
        path_id=scenario["path_id"],  # type: ignore[arg-type]
        student_id=seeded_users.student_id,
        suffix=suffix,
    )

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lesson_progress "
                "(id, user_id, lesson_id, status, completion_percent, total_time_seconds) "
                "VALUES (uuid_generate_v4(), :uid, :lid, 'completed', :pct, 600)"
            ),
            {
                "uid": seeded_users.student_id,
                "lid": scenario["pub_a_lesson"],
                "pct": Decimal("100"),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lesson_progress "
                "(id, user_id, lesson_id, status, completion_percent, total_time_seconds) "
                "VALUES (uuid_generate_v4(), :uid, :lid, 'in_progress', :pct, 200)"
            ),
            {
                "uid": seeded_users.student_id,
                "lid": scenario["pub_b_lesson"],
                "pct": Decimal("50"),
            },
        )
        # D2: `completed_courses` counts courses whose COURSE ENROLLMENT is
        # 'completed', not those at 100% lesson progress — the two are
        # deliberately different (a course with no enrollment row has no
        # status to be complete). Pattern B means path assignment no longer
        # creates these rows, so the test creates them explicitly: pub_a
        # finished, pub_b still in flight.
        for course_key, enrollment_status in (
            ("pub_a_id", "completed"),
            ("pub_b_id", "active"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO course_enrollments "
                    "(id, course_id, student_id, status, source) "
                    "VALUES (uuid_generate_v4(), :cid, :uid, :st, 'manager_bulk')"
                ),
                {
                    "cid": scenario[course_key],
                    "uid": seeded_users.student_id,
                    "st": enrollment_status,
                },
            )

    response = await client.get(
        f"/api/v1/me/career-enrollments/{scenario['path_id']}/progress",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["course_count"] == 2
    assert body["completed_courses"] == 1
    assert body["in_progress_courses"] == 1
    # 50%, not the old 75%. Completion is counted in whole UNITS now
    # (lesson/quiz/interview done or not done), so pub_b's half-watched lesson
    # contributes 0 rather than 50: one of two courses is finished. The old
    # fractional lesson average is what let a course carrying an unanswered
    # quiz read as complete, so the loss of partial credit is the point.
    assert body["overall_percent"] == 50.0


async def test_reorder_courses_in_path(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    new_order = [
        str(scenario["draft_id"]),
        str(scenario["pub_a_id"]),
        str(scenario["pub_b_id"]),
    ]
    response = await client.put(
        f"/api/v1/management/career-paths/{scenario['path_id']}/courses/reorder",
        json={"course_ids": new_order},
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 200, response.text

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT course_id, position FROM career_course_items "
                    "WHERE version_id IN (SELECT id FROM career_path_versions "
                    "WHERE career_path_id = :pid) ORDER BY position"
                ),
                {"pid": scenario["path_id"]},
            )
        ).all()
    positions = [(str(row.course_id), row.position) for row in rows]
    assert positions == [
        (str(scenario["draft_id"]), 1),
        (str(scenario["pub_a_id"]), 2),
        (str(scenario["pub_b_id"]), 3),
    ]


async def test_archive_path_idempotent(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
) -> None:
    headers = {"Authorization": f"Bearer {manager_bearer}"}
    first = await client.post(
        f"/api/v1/management/career-paths/{scenario['path_id']}/archive",
        headers=headers,
    )
    assert first.status_code == 200
    assert first.json()["status"] == "archived"

    second = await client.post(
        f"/api/v1/management/career-paths/{scenario['path_id']}/archive",
        headers=headers,
    )
    assert second.status_code == 409


async def test_teacher_only_authoring(
    client: httpx.AsyncClient,
    teacher_bearer: str,
    seeded_users: SeededUsers,
) -> None:
    response = await client.post(
        "/api/v1/management/career-paths",
        json={
            "slug": f"teacher-attempt-{uuid.uuid4().hex[:6]}",
            "name": "Teacher should not be allowed",
        },
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 403


async def test_create_publish_lifecycle(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    suffix = uuid.uuid4().hex[:6]
    create_resp = await client.post(
        "/api/v1/management/career-paths",
        json={
            "slug": f"lc-{suffix}",
            "name": f"Lifecycle {suffix}",
            "description": "lifecycle test",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert create_resp.status_code == 201, create_resp.text
    body = create_resp.json()
    path_id = body["id"]
    assert body["status"] == "draft"
    auth = {"Authorization": f"Bearer {manager_bearer}"}

    # Publish GATE (rev 3 two-class validation): a path with no stages is
    # merely *unfinished*, which is fine to save but not to publish.
    premature = await client.post(
        f"/api/v1/management/career-paths/{path_id}/publish", headers=auth
    )
    assert premature.status_code == 400, premature.text
    assert "path_has_no_stages" in premature.text

    # Add a stage — still empty, so the gate must STILL refuse...
    stage_resp = await client.post(
        f"/api/v1/management/career-paths/{path_id}/stages",
        json={"title": "Stage 1", "unlock_policy": "always"},
        headers=auth,
    )
    assert stage_resp.status_code == 201, stage_resp.text
    stage_id = stage_resp.json()["id"]

    empty_stage = await client.post(
        f"/api/v1/management/career-paths/{path_id}/publish", headers=auth
    )
    assert empty_stage.status_code == 400, empty_stage.text
    assert "stage_has_no_courses" in empty_stage.text

    # ...until the stage holds a published course.
    course_id, lesson_id = await _insert_course_with_lesson(
        engine,
        organization_id=seeded_users.organization_id,
        owner_id=seeded_users.admin_id,
        slug=f"lc-course-{suffix}",
        title="Lifecycle course",
        status="published",
    )
    add_resp = await client.post(
        f"/api/v1/management/career-paths/{path_id}/courses",
        json={"stage_id": stage_id, "course_id": str(course_id), "is_required": True},
        headers=auth,
    )
    assert add_resp.status_code == 201, add_resp.text

    publish_resp = await client.post(
        f"/api/v1/management/career-paths/{path_id}/publish", headers=auth
    )
    assert publish_resp.status_code == 200, publish_resp.text
    assert publish_resp.json()["status"] == "published"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM career_course_items WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :id)"
            ),
            {"id": path_id}
        )
        await conn.execute(
            text(
                "DELETE FROM career_path_stages WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :id)"
            ),
            {"id": path_id}
        )
        await conn.execute(text("DELETE FROM career_paths WHERE id = :id"), {"id": path_id})
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE course_id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})


async def test_create_career_path_resolves_org_from_token(
    client: httpx.AsyncClient,
    manager_bearer: str,
    seeded_users: SeededUsers,
    engine: AsyncEngine,
) -> None:
    """``POST /management/career-paths`` derives ``organization_id`` from the
    bearer token, NOT the payload.

    Mirrors the contract introduced for ``POST /teacher/courses`` so a
    manager in Org A cannot create a path in Org B by editing the
    request body.
    """
    suffix = uuid.uuid4().hex[:6]
    response = await client.post(
        "/api/v1/management/career-paths",
        json={
            "slug": f"derive-{suffix}",
            "name": f"Server-Derived {suffix}",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["organization_id"] == str(seeded_users.organization_id)
    try:
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT organization_id FROM career_paths WHERE id = :id"),
                    {"id": body["id"]},
                )
            ).one()
        assert row.organization_id == seeded_users.organization_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM career_paths WHERE id = :id"), {"id": body["id"]})


async def test_create_career_path_rejects_forged_organization_id(
    client: httpx.AsyncClient,
    manager_bearer: str,
) -> None:
    """A forged ``organization_id`` in the payload must be rejected at the
    schema layer (``extra='forbid'``).

    Same hostile wire shape that prompted the courses fix; without
    strict-extras the backend would have honoured the spoofed id.
    """
    forged_org = "00000000-0000-0000-0000-000000000001"
    response = await client.post(
        "/api/v1/management/career-paths",
        json={
            "organization_id": forged_org,
            "slug": f"forge-{uuid.uuid4().hex[:6]}",
            "name": "Forged",
        },
        headers={"Authorization": f"Bearer {manager_bearer}"},
    )
    assert response.status_code == 422, response.text


async def test_create_career_path_duplicate_slug_returns_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
) -> None:
    """``career_paths_organization_id_slug_key`` collisions surface as 409."""
    suffix = uuid.uuid4().hex[:6]
    body = {
        "slug": f"dup-{suffix}",
        "name": "Dup Path",
    }
    auth = {"Authorization": f"Bearer {manager_bearer}"}
    first = await client.post("/api/v1/management/career-paths", json=body, headers=auth)
    assert first.status_code == 201, first.text
    created_id = first.json()["id"]
    try:
        second = await client.post("/api/v1/management/career-paths", json=body, headers=auth)
        assert second.status_code == 409, second.text
        detail = second.json()["detail"]
        assert detail["error"] == "conflict"
        assert "career_path_slug_taken" in detail["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM career_paths WHERE id = :id"), {"id": created_id})


async def test_add_unpublished_course_to_published_path_is_rejected(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    """A draft course must not be attachable to a PUBLISHED path — the path is
    a live surface and an invisible item would never appear for students
    (guard on ``path.status == 'published'`` in add_course_to_path). Draft
    paths have no enrollees, so they may hold draft courses while the
    manager builds the skeleton; the publish gate re-checks every link.
    """
    suffix = uuid.uuid4().hex[:6]
    auth = {"Authorization": f"Bearer {manager_bearer}"}

    create_resp = await client.post(
        "/api/v1/management/career-paths",
        json={"slug": f"guard-{suffix}", "name": "Guard Path"},
        headers=auth,
    )
    assert create_resp.status_code == 201, create_resp.text
    fresh_path_id = create_resp.json()["id"]

    # Every course item now belongs to a stage, so the guard is exercised
    # through a real stage.
    stage_resp = await client.post(
        f"/api/v1/management/career-paths/{fresh_path_id}/stages",
        json={"title": "Stage 1", "unlock_policy": "always"},
        headers=auth,
    )
    assert stage_resp.status_code == 201, stage_resp.text
    fresh_stage_id = stage_resp.json()["id"]

    try:
        # The publish gate requires >= 1 course per stage, so seed the stage
        # with a published course first (legal on the draft path), then take
        # the path live.
        seed = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/courses",
            json={
                "stage_id": fresh_stage_id,
                "course_id": str(scenario["pub_a_id"]),
                "is_required": True,
            },
            headers=auth,
        )
        assert seed.status_code == 201, seed.text

        pub = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/publish",
            headers=auth,
        )
        assert pub.status_code == 200, pub.text

        # Gap 3 (D1b pinned): the published version is FROZEN — adding any
        # course now requires a fork first (409 version_published).
        reject = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/courses",
            json={
                "stage_id": fresh_stage_id,
                "course_id": str(scenario["draft_id"]),
                "is_required": True,
            },
            headers=auth,
        )
        assert reject.status_code == 409, reject.text
        detail = reject.json()["detail"]
        assert "version" in detail["message"]

        # Fork (copy-on-write) -> the draft course is still rejected on the
        # published path (path-level guard: only published courses attach),
        # while a published course is accepted on the draft version.
        fork_resp = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/versions",
            headers=auth,
        )
        assert fork_resp.status_code == 201, fork_resp.text

        # The fork cloned the stages — resolve the DRAFT version's stage id.
        draft_stages = await client.get(
            f"/api/v1/management/career-paths/{fresh_path_id}/stages",
            headers=auth,
        )
        assert draft_stages.status_code == 200, draft_stages.text
        draft_stage_id = draft_stages.json()[0]["id"]

        reject = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/courses",
            json={
                "stage_id": draft_stage_id,
                "course_id": str(scenario["draft_id"]),
                "is_required": True,
            },
            headers=auth,
        )
        assert reject.status_code == 409, reject.text
        detail = reject.json()["detail"]
        assert detail["error"] == "conflict"
        # Names the COURSE and its actual status — a bare uuid + "is not
        # published" read as if the PATH were unpublished.
        assert "course_not_published" in detail["message"]
        assert "draft course" in detail["message"]
        assert "Guard Path" in detail["message"]

        # Published course -> still accepted (on the draft version).
        ok = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/courses",
            json={
                "stage_id": draft_stage_id,
                "course_id": str(scenario["pub_b_id"]),
                "is_required": True,
            },
            headers=auth,
        )
        assert ok.status_code == 201, ok.text

        # And the draft was never attached — only the two published courses
        # (across both versions: v1 published + v2 draft clone).
        async with engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        text(
                            "SELECT course_id FROM career_course_items WHERE version_id IN "
                            "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
                        ),
                        {"pid": fresh_path_id},
                    )
                )
                .scalars()
                .all()
            )
        assert set(str(r) for r in rows) == {
            str(scenario["pub_a_id"]),
            str(scenario["pub_b_id"]),
        }
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                "DELETE FROM career_course_items WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
                {"pid": fresh_path_id},
            )
            await conn.execute(
                text(
                "DELETE FROM career_path_stages WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
                {"pid": fresh_path_id},
            )
            await conn.execute(
                text("DELETE FROM career_paths WHERE id = :id"), {"id": fresh_path_id}
            )


async def test_add_draft_course_to_draft_path_is_allowed(
    client: httpx.AsyncClient,
    manager_bearer: str,
    scenario: dict[str, object],
    engine: AsyncEngine,
) -> None:
    """A draft path may hold draft courses — no enrollees exist yet, so
    nothing can break; the manager builds the skeleton before the path goes
    live. The published-course requirement applies only to published paths.
    """
    suffix = uuid.uuid4().hex[:6]
    auth = {"Authorization": f"Bearer {manager_bearer}"}

    create_resp = await client.post(
        "/api/v1/management/career-paths",
        json={"slug": f"draft-{suffix}", "name": "Draft Skeleton Path"},
        headers=auth,
    )
    assert create_resp.status_code == 201, create_resp.text
    fresh_path_id = create_resp.json()["id"]

    stage_resp = await client.post(
        f"/api/v1/management/career-paths/{fresh_path_id}/stages",
        json={"title": "Stage 1", "unlock_policy": "always"},
        headers=auth,
    )
    assert stage_resp.status_code == 201, stage_resp.text
    fresh_stage_id = stage_resp.json()["id"]

    try:
        ok = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/courses",
            json={
                "stage_id": fresh_stage_id,
                "course_id": str(scenario["draft_id"]),
                "is_required": True,
            },
            headers=auth,
        )
        assert ok.status_code == 201, ok.text

        # Publishing the path now must fail: the stage holds a draft course
        # (the publish gate's completeness check, not the mutation guard).
        pub = await client.post(
            f"/api/v1/management/career-paths/{fresh_path_id}/publish",
            headers=auth,
        )
        assert pub.status_code == 400, pub.text
        body = pub.json()["detail"]
        assert "not a published course" in body["message"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                "DELETE FROM career_course_items WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
                {"pid": fresh_path_id},
            )
            await conn.execute(
                text(
                "DELETE FROM career_path_stages WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :pid)"
            ),
                {"pid": fresh_path_id},
            )
            await conn.execute(
                text("DELETE FROM career_paths WHERE id = :id"), {"id": fresh_path_id}
            )
