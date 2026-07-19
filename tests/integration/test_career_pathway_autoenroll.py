"""Integration tests for W3 career-pathway auto-enroll + "prepared" milestone.

Covers:
* Enrolling a student in a career auto-creates ``course_enrollments`` for its
  **required** courses only; idempotent; reactivates a dropped row.
* Adding a required course to a career backfills active enrollees.
* Removing a course from the career leaves ``course_enrollments`` intact and
  drops it from derived progress (no in-flight conflict).
* Completing every course flips the career enrollment to ``completed``
  ("prepared") and surfaces `is_prepared`/`overall_percent`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
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

import abridgeai.features.access_control.models  # noqa: F401  -- FK targets
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.enrollments.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.security import CurrentUser
from abridgeai.features.career_paths.services import authoring as authoring_service
from abridgeai.features.career_paths.services import enrollment as enrollment_service


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


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def _course_with_lesson(
    conn, *, org: uuid.UUID, owner: uuid.UUID, slug: str
) -> tuple[uuid.UUID, uuid.UUID]:
    course_id, module_id, lesson_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
            "VALUES (:id, :org, :owner, :slug, :slug, 'published')"
        ),
        {"id": course_id, "org": org, "owner": owner, "slug": slug},
    )
    await conn.execute(
        text(
            "INSERT INTO modules (id, course_id, title, position, status) "
            "VALUES (:id, :cid, 'M', 1, 'published')"
        ),
        {"id": module_id, "cid": course_id},
    )
    await conn.execute(
        text(
            "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
            "VALUES (:id, :mid, :slug, 'L', 'published', 'video')"
        ),
        {"id": lesson_id, "mid": module_id, "slug": f"{slug}-l1"},
    )
    return course_id, lesson_id


@pytest_asyncio.fixture
async def seed(engine: AsyncEngine) -> AsyncIterator[dict]:
    """Org + manager + student; a REQUIRED and an OPTIONAL published course in
    one published path."""
    s = uuid.uuid4().hex[:8]
    org, manager, student = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    path_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org, "slug": f"cp-{s}", "name": "CP Org"},
        )
        for uid, email in ((manager, f"mgr-{s}@t.local"), (student, f"stu-{s}@t.local")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": email},
            )
        req_course, req_lesson = await _course_with_lesson(
            conn, org=org, owner=manager, slug=f"req-{s}"
        )
        opt_course, opt_lesson = await _course_with_lesson(
            conn, org=org, owner=manager, slug=f"opt-{s}"
        )
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, 'Path', 'published')"
            ),
            {"id": path_id, "org": org, "slug": f"path-{s}"},
        )
        for pos, (cid, req) in enumerate(
            ((req_course, True), (opt_course, False)), start=1
        ):
            await conn.execute(
                text(
                    "INSERT INTO career_course_items "
                    "(career_path_id, course_id, position, is_required) "
                    "VALUES (:pid, :cid, :pos, :req)"
                ),
                {"pid": path_id, "cid": cid, "pos": pos, "req": req},
            )

    yield {
        "org": org, "manager": manager, "student": student, "path_id": path_id,
        "req_course": req_course, "opt_course": opt_course,
        "req_lesson": req_lesson, "opt_lesson": opt_lesson,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE student_id = :s"), {"s": student}
        )
        await conn.execute(
            text("DELETE FROM lesson_progress WHERE user_id = :s"), {"s": student}
        )
        await conn.execute(
            text("DELETE FROM student_career_enrollments WHERE career_path_id = :p"),
            {"p": path_id},
        )
        await conn.execute(
            text("DELETE FROM career_course_items WHERE career_path_id = :p"), {"p": path_id}
        )
        await conn.execute(text("DELETE FROM career_paths WHERE id = :p"), {"p": path_id})
        await conn.execute(
            text("DELETE FROM lessons WHERE module_id IN "
                 "(SELECT id FROM modules WHERE course_id = ANY(CAST(:ids AS uuid[])))"),
            {"ids": [str(req_course), str(opt_course)]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE course_id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(req_course), str(opt_course)]},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(req_course), str(opt_course)]},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(manager), str(student)]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org})


async def _enrollment_status(engine: AsyncEngine, student: uuid.UUID, course: uuid.UUID):
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT status FROM course_enrollments "
                    "WHERE student_id = :s AND course_id = :c"
                ),
                {"s": student, "c": course},
            )
        ).scalar_one_or_none()


async def _mark_lesson_complete(engine: AsyncEngine, student: uuid.UUID, lesson: uuid.UUID):
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lesson_progress (user_id, lesson_id, status, completion_percent) "
                "VALUES (:u, :l, 'completed', 100)"
            ),
            {"u": student, "l": lesson},
        )


@pytest.mark.asyncio
async def test_enroll_fans_out_required_courses_only(
    engine, session_factory, seed
) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"
    assert await _enrollment_status(engine, seed["student"], seed["opt_course"]) is None


@pytest.mark.asyncio
async def test_reenroll_reactivates_and_is_idempotent(engine, session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    # Drop career + course enrollment, then re-enroll → both reactivate, no dup.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE course_enrollments SET status='dropped' WHERE student_id=:s"),
            {"s": seed["student"]},
        )
    async with session_factory() as db:
        await enrollment_service.unenroll_student(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await enrollment_service.enroll_student_in_path(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"
    async with engine.connect() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM course_enrollments "
                    "WHERE student_id=:s AND course_id=:c"
                ),
                {"s": seed["student"], "c": seed["req_course"]},
            )
        ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_add_required_course_backfills_active_students(engine, session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    # A new required published course, added after the student was enrolled.
    async with engine.begin() as conn:
        new_course, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"late-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db, seed["path_id"], new_course, position=None, is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], new_course) == "active"


@pytest.mark.asyncio
async def test_remove_course_is_non_destructive(engine, session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    async with session_factory() as db:
        await authoring_service.remove_course_from_path(
            db, seed["path_id"], seed["req_course"], actor=_actor(seed["manager"])
        )
        await db.commit()

    # course_enrollment untouched...
    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"
    # ...but the course no longer counts toward the path.
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
    assert seed["req_course"] not in {c.course_id for c in progress.courses}


@pytest.mark.asyncio
async def test_completion_flips_to_prepared(engine, session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    await _mark_lesson_complete(engine, seed["student"], seed["req_lesson"])
    await _mark_lesson_complete(engine, seed["student"], seed["opt_lesson"])

    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        assert progress.overall_percent == 100
        flipped = await enrollment_service.sync_enrollment_completion(
            db, career_path_id=seed["path_id"], student_id=seed["student"],
            overall_percent=progress.overall_percent,
        )
        await db.commit()
    assert flipped is True

    async with session_factory() as db:
        rows = await enrollment_service.list_my_career_enrollments(db, seed["student"])
        await db.commit()
    row = next(r for r in rows if r.career_path_id == seed["path_id"])
    assert row.status == "completed"
    assert row.is_prepared is True
    assert row.overall_percent == 100

    async with engine.connect() as conn:
        status, completed_at = (
            await conn.execute(
                text(
                    "SELECT status, completed_at FROM student_career_enrollments "
                    "WHERE career_path_id=:p AND student_id=:s"
                ),
                {"p": seed["path_id"], "s": seed["student"]},
            )
        ).one()
    assert status == "completed"
    assert completed_at is not None
