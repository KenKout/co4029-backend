from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.features.materials.queries import (
    get_latest_processing_job,
    get_latest_ready_version,
    get_material_for_authoring,
    get_material_with_versions,
    get_visible_material,
    list_all_materials,
    list_failed_jobs_recent,
    list_jobs_in_progress,
    list_visible_materials,
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
async def fixture_data(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner = uuid.uuid4()
    course_id = uuid.uuid4()
    other_course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    other_module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    other_lesson_id = uuid.uuid4()
    storage_id = uuid.uuid4()

    mat_visible_ready = uuid.uuid4()
    mat_invisible_ready = uuid.uuid4()
    mat_visible_processing = uuid.uuid4()
    mat_archived = uuid.uuid4()
    # A material whose newest version is unservable while an older one is
    # ready: the re-upload case. One still in the pipeline, one whose ingest
    # failed outright.
    mat_reupload_in_flight = uuid.uuid4()
    mat_reupload_failed = uuid.uuid4()

    ver_v1_ready = uuid.uuid4()
    ver_v2_ready_current = uuid.uuid4()
    ver_invisible_ready = uuid.uuid4()
    ver_processing = uuid.uuid4()
    ver_archived_cancelled = uuid.uuid4()
    ver_other_course = uuid.uuid4()
    ver_in_flight_old_ready = uuid.uuid4()
    ver_in_flight_new_pending = uuid.uuid4()
    ver_failed_old_ready = uuid.uuid4()
    ver_failed_new = uuid.uuid4()

    mat_other_course = uuid.uuid4()

    job_old = uuid.uuid4()
    job_mid = uuid.uuid4()
    job_new = uuid.uuid4()
    job_running = uuid.uuid4()
    job_succeeded = uuid.uuid4()
    job_failed_recent = uuid.uuid4()
    job_failed_old = uuid.uuid4()
    job_other_course_running = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Materials Org', 'active')"
            ),
            {"id": org_id, "slug": f"morg-{org_id.hex[:8]}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": owner, "email": f"owner-{owner.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:c1, :o, :u, :s1, 'Course One', 'published'), "
                "(:c2, :o, :u, :s2, 'Course Two', 'published')"
            ),
            {
                "c1": course_id,
                "c2": other_course_id,
                "o": org_id,
                "u": owner,
                "s1": f"course-1-{course_id.hex[:6]}",
                "s2": f"course-2-{other_course_id.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m1, :c1, 'Module One', 1, 'published'), "
                "(:m2, :c2, 'Module Two', 1, 'published')"
            ),
            {"m1": module_id, "m2": other_module_id, "c1": course_id, "c2": other_course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l1, :m1, 'lesson-1', 'Lesson One', 'published'), "
                "(:l2, :m2, 'lesson-2', 'Lesson Two', 'published')"
            ),
            {"l1": lesson_id, "l2": other_lesson_id, "m1": module_id, "m2": other_module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type, size_bytes) "
                "VALUES (:id, 'test-bucket', 'materials/test.pdf', 'application/pdf', 1024)"
            ),
            {"id": storage_id},
        )

        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type, "
                "ai_processing_enabled, visible_to_students) VALUES "
                "(:m1, :l, 'Visible Ready', 'pdf', TRUE, TRUE), "
                "(:m2, :l, 'Invisible Ready', 'pdf', TRUE, FALSE), "
                "(:m3, :l, 'Visible Processing', 'pdf', TRUE, TRUE), "
                "(:m4, :l, 'Archived', 'pdf', TRUE, TRUE), "
                "(:m5, :l2, 'Other Lesson Material', 'pdf', TRUE, TRUE), "
                "(:m6, :l, 'Re-upload In Flight', 'pdf', TRUE, TRUE), "
                "(:m7, :l, 'Re-upload Failed', 'pdf', TRUE, TRUE)"
            ),
            {
                "m1": mat_visible_ready,
                "m2": mat_invisible_ready,
                "m3": mat_visible_processing,
                "m4": mat_archived,
                "m5": mat_other_course,
                "m6": mat_reupload_in_flight,
                "m7": mat_reupload_failed,
                "l": lesson_id,
                "l2": other_lesson_id,
            },
        )

        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, is_current, processing_status) "
                "VALUES "
                "(:v1, :m1, :s, 1, FALSE, 'ready'), "
                "(:v2, :m1, :s, 2, TRUE, 'ready'), "
                "(:vinv, :m2, :s, 1, TRUE, 'ready'), "
                "(:vproc, :m3, :s, 1, TRUE, 'extracting'), "
                "(:vcan, :m4, :s, 1, TRUE, 'cancelled'), "
                "(:voc, :m5, :s, 1, TRUE, 'ready'), "
                "(:vifold, :m6, :s, 1, FALSE, 'ready'), "
                "(:vifnew, :m6, :s, 2, TRUE, 'pending'), "
                "(:vfold, :m7, :s, 1, FALSE, 'ready'), "
                "(:vfnew, :m7, :s, 2, TRUE, 'failed')"
            ),
            {
                "v1": ver_v1_ready,
                "v2": ver_v2_ready_current,
                "vinv": ver_invisible_ready,
                "vproc": ver_processing,
                "vcan": ver_archived_cancelled,
                "voc": ver_other_course,
                "vifold": ver_in_flight_old_ready,
                "vifnew": ver_in_flight_new_pending,
                "vfold": ver_failed_old_ready,
                "vfnew": ver_failed_new,
                "m1": mat_visible_ready,
                "m2": mat_invisible_ready,
                "m3": mat_visible_processing,
                "m4": mat_archived,
                "m5": mat_other_course,
                "m6": mat_reupload_in_flight,
                "m7": mat_reupload_failed,
                "s": storage_id,
            },
        )
        await conn.execute(
            text(
                "UPDATE learning_materials SET current_version_id = CASE id "
                "WHEN :m1 THEN :v2 WHEN :m2 THEN :vinv WHEN :m3 THEN :vproc "
                "WHEN :m4 THEN :vcan WHEN :m5 THEN :voc "
                "WHEN :m6 THEN :vifnew WHEN :m7 THEN :vfnew END "
                "WHERE id IN (:m1, :m2, :m3, :m4, :m5, :m6, :m7)"
            ),
            {
                "m1": mat_visible_ready,
                "m2": mat_invisible_ready,
                "m3": mat_visible_processing,
                "m4": mat_archived,
                "m5": mat_other_course,
                "m6": mat_reupload_in_flight,
                "m7": mat_reupload_failed,
                "v2": ver_v2_ready_current,
                "vinv": ver_invisible_ready,
                "vproc": ver_processing,
                "vcan": ver_archived_cancelled,
                "voc": ver_other_course,
                "vifnew": ver_in_flight_new_pending,
                "vfnew": ver_failed_new,
            },
        )

        now = datetime.now(tz=UTC)
        await conn.execute(
            text(
                "INSERT INTO processing_jobs (id, entity_type, entity_id, job_type, status, "
                "progress_percent, retry_count, created_at, updated_at) VALUES "
                "(:jold, 'material_version', :v1, 'full_pipeline', 'pending', 0, 0, :t_old, :t_old), "
                "(:jmid, 'material_version', :v1, 'full_pipeline', 'pending', 0, 0, :t_mid, :t_mid), "
                "(:jnew, 'material_version', :v1, 'full_pipeline', 'pending', 50, 0, :t_new, :t_new), "
                "(:jrun, 'material_version', :vproc, 'full_pipeline', 'running', 50, 0, :now, :now), "
                "(:jsucc, 'material_version', :v2, 'full_pipeline', 'completed', 100, 0, :now, :now), "
                "(:jfr, 'material_version', :vinv, 'full_pipeline', 'failed', 25, 1, :now, :now), "
                "(:jfo, 'material_version', :vinv, 'full_pipeline', 'failed', 25, 1, :t_old2, :t_old2), "
                "(:joc, 'material_version', :voc, 'full_pipeline', 'running', 10, 0, :now, :now)"
            ),
            {
                "jold": job_old,
                "jmid": job_mid,
                "jnew": job_new,
                "jrun": job_running,
                "jsucc": job_succeeded,
                "jfr": job_failed_recent,
                "jfo": job_failed_old,
                "joc": job_other_course_running,
                "v1": ver_v1_ready,
                "v2": ver_v2_ready_current,
                "vinv": ver_invisible_ready,
                "vproc": ver_processing,
                "voc": ver_other_course,
                "t_old": now - timedelta(hours=3),
                "t_mid": now - timedelta(hours=2),
                "t_new": now - timedelta(hours=1),
                "t_old2": now - timedelta(days=10),
                "now": now,
            },
        )

    data = {
        "org_id": org_id,
        "owner": owner,
        "course_id": course_id,
        "other_course_id": other_course_id,
        "module_id": module_id,
        "other_module_id": other_module_id,
        "lesson_id": lesson_id,
        "other_lesson_id": other_lesson_id,
        "storage_id": storage_id,
        "mat_visible_ready": mat_visible_ready,
        "mat_invisible_ready": mat_invisible_ready,
        "mat_visible_processing": mat_visible_processing,
        "mat_archived": mat_archived,
        "mat_other_course": mat_other_course,
        "mat_reupload_in_flight": mat_reupload_in_flight,
        "mat_reupload_failed": mat_reupload_failed,
        "ver_in_flight_old_ready": ver_in_flight_old_ready,
        "ver_in_flight_new_pending": ver_in_flight_new_pending,
        "ver_failed_old_ready": ver_failed_old_ready,
        "ver_failed_new": ver_failed_new,
        "ver_v1_ready": ver_v1_ready,
        "ver_v2_ready_current": ver_v2_ready_current,
        "ver_invisible_ready": ver_invisible_ready,
        "ver_processing": ver_processing,
        "ver_archived_cancelled": ver_archived_cancelled,
        "ver_other_course": ver_other_course,
        "job_old": job_old,
        "job_mid": job_mid,
        "job_new": job_new,
        "job_running": job_running,
        "job_succeeded": job_succeeded,
        "job_failed_recent": job_failed_recent,
        "job_failed_old": job_failed_old,
        "job_other_course_running": job_other_course_running,
        "now": now,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM processing_jobs WHERE id = ANY(:ids)"),
            {
                "ids": [
                    job_old,
                    job_mid,
                    job_new,
                    job_running,
                    job_succeeded,
                    job_failed_recent,
                    job_failed_old,
                    job_other_course_running,
                ]
            },
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = NULL WHERE id = ANY(:ids)"),
            {
                "ids": [
                    mat_visible_ready,
                    mat_invisible_ready,
                    mat_visible_processing,
                    mat_archived,
                    mat_other_course,
                    mat_reupload_in_flight,
                    mat_reupload_failed,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = ANY(:ids)"),
            {
                "ids": [
                    ver_v1_ready,
                    ver_v2_ready_current,
                    ver_invisible_ready,
                    ver_processing,
                    ver_archived_cancelled,
                    ver_other_course,
                    ver_in_flight_old_ready,
                    ver_in_flight_new_pending,
                    ver_failed_old_ready,
                    ver_failed_new,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = ANY(:ids)"),
            {
                "ids": [
                    mat_visible_ready,
                    mat_invisible_ready,
                    mat_visible_processing,
                    mat_archived,
                    mat_other_course,
                    mat_reupload_in_flight,
                    mat_reupload_failed,
                ]
            },
        )
        await conn.execute(text("DELETE FROM storage_objects WHERE id = :id"), {"id": storage_id})
        await conn.execute(
            text("DELETE FROM lessons WHERE id = ANY(:ids)"),
            {"ids": [lesson_id, other_lesson_id]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [module_id, other_module_id]},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [course_id, other_course_id]},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_published_excludes_unready(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        materials = await list_visible_materials(session, fixture_data["lesson_id"])
    ids = {m.id for m in materials}
    assert fixture_data["mat_visible_ready"] in ids
    assert fixture_data["mat_invisible_ready"] not in ids
    assert fixture_data["mat_visible_processing"] not in ids


async def test_published_excludes_invisible(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        materials = await list_visible_materials(session, fixture_data["lesson_id"])
    ids = {m.id for m in materials}
    assert fixture_data["mat_invisible_ready"] not in ids


async def test_published_excludes_processing(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        materials = await list_visible_materials(session, fixture_data["lesson_id"])
    ids = {m.id for m in materials}
    assert fixture_data["mat_visible_processing"] not in ids


async def test_get_visible_material_404_for_processing(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        ready = await get_visible_material(session, fixture_data["mat_visible_ready"])
        processing = await get_visible_material(session, fixture_data["mat_visible_processing"])
        invisible = await get_visible_material(session, fixture_data["mat_invisible_ready"])
    assert ready is not None
    assert ready.id == fixture_data["mat_visible_ready"]
    assert processing is None
    assert invisible is None


async def test_get_latest_ready_version_returns_current(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        version = await get_latest_ready_version(session, fixture_data["mat_visible_ready"])
    assert version is not None
    assert version.id == fixture_data["ver_v2_ready_current"]
    assert version.is_current is True


async def test_get_latest_ready_version_none_for_processing(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        version = await get_latest_ready_version(session, fixture_data["mat_visible_processing"])
    assert version is None


async def test_a_re_upload_in_flight_keeps_serving_the_last_ready_version(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """Students keep reading v1 while v2 is still in the pipeline.

    ``complete_upload`` promotes the new version before the ingest runs, so
    pairing ``current_version_id`` with ``processing_status='ready'`` — which
    every learner read used to do — left no row satisfying both. The material
    dropped out of the lesson for the whole of processing, which for a large
    PDF is minutes.
    """
    async with session_factory() as session:
        materials = await list_visible_materials(session, fixture_data["lesson_id"])
        single = await get_visible_material(session, fixture_data["mat_reupload_in_flight"])
        version = await get_latest_ready_version(session, fixture_data["mat_reupload_in_flight"])

    assert fixture_data["mat_reupload_in_flight"] in {m.id for m in materials}
    assert single is not None
    assert version is not None
    assert version.id == fixture_data["ver_in_flight_old_ready"]


async def test_a_failed_re_upload_does_not_take_the_material_dark(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """The permanent case: the new version's ingest failed and never recovers.

    Nothing in the pipeline restores ``is_current`` to the previous version, so
    before the fallback this material was unreadable until a teacher noticed
    and called ``rollback_to_version`` by hand.
    """
    async with session_factory() as session:
        materials = await list_visible_materials(session, fixture_data["lesson_id"])
        single = await get_visible_material(session, fixture_data["mat_reupload_failed"])
        version = await get_latest_ready_version(session, fixture_data["mat_reupload_failed"])

    assert fixture_data["mat_reupload_failed"] in {m.id for m in materials}
    assert single is not None
    assert version is not None
    assert version.id == fixture_data["ver_failed_old_ready"]


async def test_the_current_version_still_wins_when_it_is_ready(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """The fallback must not override a deliberate rollback.

    ``mat_visible_ready`` holds two ready versions, v1 and the current v2.
    Ordering on ``is_current`` before ``version_no`` is what makes this work in
    the rollback direction too: ``rollback_to_version`` refuses a target that
    is not ready, so a rolled-back-to v1 would be current AND ready, win the
    sort, and keep the newer v2 out of the learner's way.
    """
    async with session_factory() as session:
        version = await get_latest_ready_version(session, fixture_data["mat_visible_ready"])

    assert version is not None
    assert version.id == fixture_data["ver_v2_ready_current"]
    assert version.is_current is True


async def test_a_first_upload_still_processing_has_nothing_to_fall_back_to(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """A material that has never finished a version stays invisible."""
    async with session_factory() as session:
        materials = await list_visible_materials(session, fixture_data["lesson_id"])
        version = await get_latest_ready_version(session, fixture_data["mat_visible_processing"])

    assert fixture_data["mat_visible_processing"] not in {m.id for m in materials}
    assert version is None


async def test_authoring_returns_all_states(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        materials = await list_all_materials(session, fixture_data["lesson_id"])
    ids = {m.id for m in materials}
    assert fixture_data["mat_visible_ready"] in ids
    assert fixture_data["mat_invisible_ready"] in ids
    assert fixture_data["mat_visible_processing"] in ids
    assert fixture_data["mat_archived"] not in ids


async def test_authoring_include_archived_flag(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        without_archived = await list_all_materials(session, fixture_data["lesson_id"])
        with_archived = await list_all_materials(
            session, fixture_data["lesson_id"], include_archived=True
        )
    ids_without = {m.id for m in without_archived}
    ids_with = {m.id for m in with_archived}
    assert fixture_data["mat_archived"] not in ids_without
    assert fixture_data["mat_archived"] in ids_with


async def test_get_material_with_versions_eager_loads(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        result = await get_material_with_versions(session, fixture_data["mat_visible_ready"])
    assert result is not None
    material, versions = result
    assert material.id == fixture_data["mat_visible_ready"]
    version_ids = [v.id for v in versions]
    assert version_ids == [
        fixture_data["ver_v2_ready_current"],
        fixture_data["ver_v1_ready"],
    ]


async def test_get_material_for_authoring_returns_processing(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        material = await get_material_for_authoring(session, fixture_data["mat_visible_processing"])
    assert material is not None
    assert material.id == fixture_data["mat_visible_processing"]


async def test_processing_get_latest_job_orders_by_created_desc(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        job = await get_latest_processing_job(session, fixture_data["ver_v1_ready"])
    assert job is not None
    assert job.id == fixture_data["job_new"]


async def test_jobs_in_progress_filters_by_status(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        jobs = await list_jobs_in_progress(session)
    ids = {j.id for j in jobs}
    assert fixture_data["job_old"] in ids
    assert fixture_data["job_mid"] in ids
    assert fixture_data["job_new"] in ids
    assert fixture_data["job_running"] in ids
    assert fixture_data["job_other_course_running"] in ids
    assert fixture_data["job_succeeded"] not in ids
    assert fixture_data["job_failed_recent"] not in ids


async def test_jobs_in_progress_filters_by_course_id(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        jobs = await list_jobs_in_progress(session, course_id=fixture_data["course_id"])
    ids = {j.id for j in jobs}
    assert fixture_data["job_old"] in ids
    assert fixture_data["job_mid"] in ids
    assert fixture_data["job_new"] in ids
    assert fixture_data["job_running"] in ids
    assert fixture_data["job_other_course_running"] not in ids


async def test_list_failed_jobs_since(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    cutoff = fixture_data["now"] - timedelta(days=1)
    async with session_factory() as session:
        jobs = await list_failed_jobs_recent(session, cutoff)
    ids = {j.id for j in jobs}
    assert fixture_data["job_failed_recent"] in ids
    assert fixture_data["job_failed_old"] not in ids


def test_no_mechanism_split() -> None:
    queries_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "materials"
        / "queries"
    )
    subdirs = {p.name for p in queries_dir.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert "orm" not in subdirs
    assert "raw" not in subdirs
