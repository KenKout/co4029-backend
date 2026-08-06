"""Integration tests for career-pathway course access + "prepared" milestone.

**Rewritten for Pattern B (lazy enrollment, migration 0070.)** The eager
auto-enroll this file was originally built around is gone: assigning a student
to a path no longer fans out ``course_enrollments`` for its required courses,
and adding a required course no longer backfills existing enrollees. Three of
the original five tests asserted exactly that fan-out and were invalid the
moment ``_autoenroll_required_courses`` was deleted; they are replaced by their
Pattern-B counterparts (assert NO fan-out) plus the Start endpoint, which now
owns the idempotency/reactivation behaviour those tests were guarding.

Stage-level rules (unlock, latch, cap, cross-stage move, two-class validation)
live in ``test_career_path_stages.py``.

Covers:
* Assigning a path grants access to the PATH only — no course fan-out.
* Adding a required course does NOT backfill active enrollees.
* ``ensure_course_enrollment`` remains an idempotent primitive (KEPT, not
  rewritten) and Start is the flow that uses it.
* Removing a course from the path leaves ``course_enrollments`` intact and
  drops it from derived progress.
* Completing every course flips the career enrollment to ``completed``
  ("prepared") and surfaces ``is_prepared`` / ``overall_percent``.
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
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.security import CurrentUser
from abridgeai.features.career_paths.services import authoring as authoring_service
from abridgeai.features.career_paths.services import enrollment as enrollment_service
from abridgeai.features.enrollments.api import public as enrollments_api


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
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "migrations"))
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
    one 'always'-unlocked stage of a published path."""
    s = uuid.uuid4().hex[:8]
    org, manager, student = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    path_id, stage_id = uuid.uuid4(), uuid.uuid4()

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
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, career_path_id, position, unlock_policy, enforcement) "
                "VALUES (:sid, :pid, 1, 'always', 'advisory')"
            ),
            {"sid": stage_id, "pid": path_id},
        )
        for pos, (cid, req) in enumerate(((req_course, True), (opt_course, False)), start=1):
            await conn.execute(
                text(
                    "INSERT INTO career_course_items "
                    "(career_path_id, course_id, stage_id, position, is_required) "
                    "VALUES (:pid, :cid, :sid, :pos, :req)"
                ),
                {"pid": path_id, "cid": cid, "sid": stage_id, "pos": pos, "req": req},
            )

    yield {
        "org": org,
        "manager": manager,
        "student": student,
        "path_id": path_id,
        "stage_id": stage_id,
        "req_course": req_course,
        "opt_course": opt_course,
        "req_lesson": req_lesson,
        "opt_lesson": opt_lesson,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM student_stage_progress WHERE enrollment_id IN "
                "(SELECT id FROM student_career_enrollments WHERE career_path_id = :p)"
            ),
            {"p": path_id},
        )
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE student_id = :s"), {"s": student}
        )
        await conn.execute(text("DELETE FROM lesson_progress WHERE user_id = :s"), {"s": student})
        await conn.execute(
            text("DELETE FROM student_career_enrollments WHERE career_path_id = :p"),
            {"p": path_id},
        )
        await conn.execute(
            text("DELETE FROM career_course_items WHERE career_path_id = :p"), {"p": path_id}
        )
        await conn.execute(
            text("DELETE FROM career_path_stages WHERE career_path_id = :p"), {"p": path_id}
        )
        await conn.execute(text("DELETE FROM career_paths WHERE id = :p"), {"p": path_id})
        # Scoped by organization_id (not the fixed [req_course, opt_course] ids)
        # so it also sweeps up any course a test creates mid-run, e.g. the
        # backfill test's `new_course` -- otherwise that row survives and its
        # courses.owner_user_id FK blocks the `users` delete below.
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id IN "
                "(SELECT id FROM courses WHERE organization_id = :org))"
            ),
            {"org": org},
        )
        await conn.execute(
            text(
                "DELETE FROM modules WHERE course_id IN "
                "(SELECT id FROM courses WHERE organization_id = :org)"
            ),
            {"org": org},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE organization_id = :org"),
            {"org": org},
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
                    "SELECT status FROM course_enrollments WHERE student_id = :s AND course_id = :c"
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
async def test_path_assignment_does_not_fan_out_to_courses(engine, session_factory, seed) -> None:
    """Pattern B: assigning a path grants access to the PATH only.

    Replaces the original ``test_enroll_fans_out_required_courses_only``. The
    eager fan-out it asserted enrolled a student in every required course of
    every stage at once — including stages still locked to them — which is the
    behaviour Pattern B removes. Course enrollments now come from Start.
    """
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) is None
    assert await _enrollment_status(engine, seed["student"], seed["opt_course"]) is None

    # The PATH enrollment itself exists and is active.
    async with engine.connect() as conn:
        status = (
            await conn.execute(
                text(
                    "SELECT status FROM student_career_enrollments "
                    "WHERE career_path_id = :p AND student_id = :s"
                ),
                {"p": seed["path_id"], "s": seed["student"]},
            )
        ).scalar_one()
    assert status == "active"


@pytest.mark.asyncio
async def test_single_course_enrollment_does_not_cascade_to_path(
    engine, session_factory, seed
) -> None:
    """The reverse direction: enrolling a student in ONE course of a path
    must NOT enroll them in the path itself nor in its other courses.
    Only the career-enrollment entry point fans out (required courses)."""
    async with session_factory() as db:
        await enrollments_api.ensure_course_enrollment(
            db,
            student_id=seed["student"],
            course_id=seed["req_course"],
            actor_id=seed["manager"],
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"
    assert await _enrollment_status(engine, seed["student"], seed["opt_course"]) is None
    async with engine.connect() as conn:
        count = (
            await conn.execute(
                text("SELECT COUNT(*) FROM student_career_enrollments WHERE student_id=:s"),
                {"s": seed["student"]},
            )
        ).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_reenroll_reactivates_path_without_touching_courses(
    engine, session_factory, seed
) -> None:
    """Re-assigning a dropped path enrollment reactivates the PATH row.

    Replaces the original test's course-fan-out half. The idempotency /
    reactivate-a-dropped-row behaviour it guarded now belongs to the Start
    endpoint (``ensure_course_enrollment`` was KEPT as the primitive, not
    rewritten) and is covered in ``test_career_path_stages.py``.
    """
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    # The student starts one course, then drops off the path entirely.
    async with session_factory() as db:
        await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    async with session_factory() as db:
        await enrollment_service.unenroll_student(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    # Path reactivated, exactly one row.
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT status FROM student_career_enrollments "
                        "WHERE career_path_id = :p AND student_id = :s"
                    ),
                    {"p": seed["path_id"], "s": seed["student"]},
                )
            )
            .scalars()
            .all()
        )
    assert list(rows) == ["active"]

    # The course enrollment the student already had is left alone — dropping
    # off a path must not revoke work in flight, and re-assigning must not
    # silently re-create anything either.
    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"
    assert await _enrollment_status(engine, seed["student"], seed["opt_course"]) is None


@pytest.mark.asyncio
async def test_add_required_course_does_not_backfill_active_students(
    engine, session_factory, seed
) -> None:
    """Pattern B: adding a required course must NOT backfill enrollees.

    Replaces ``test_add_required_course_backfills_active_students``. The
    backfill it asserted was the second eager fan-out (``add_course_to_path``'s
    enrollee loop) and is deleted: a manager attaching a course must not
    silently create enrollments for everyone already on the path, least of all
    into a stage they haven't unlocked. The student picks it up via Start.
    """
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
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
            db,
            seed["path_id"],
            new_course,
            stage_id=seed["stage_id"],
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], new_course) is None

    # ...and Start is what grants it (the stage is 'always' unlocked).
    async with session_factory() as db:
        result = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=new_course,
            student_id=seed["student"],
        )
        await db.commit()
    assert result.created is True
    assert await _enrollment_status(engine, seed["student"], new_course) == "active"


@pytest.mark.asyncio
async def test_remove_course_is_non_destructive(engine, session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    # Pattern B: the student has to start the course for an enrollment to exist.
    async with session_factory() as db:
        await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
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
        await db.commit()
    assert seed["req_course"] not in {c.course_id for c in progress.courses}


@pytest.mark.asyncio
async def test_completion_flips_to_prepared(engine, session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    # Pattern B: start both courses, then finish them.
    async with session_factory() as db:
        for course in (seed["req_course"], seed["opt_course"]):
            await enrollment_service.start_course_in_path(
                db,
                career_path_id=seed["path_id"],
                course_id=course,
                student_id=seed["student"],
            )
        await db.commit()

    await _mark_lesson_complete(engine, seed["student"], seed["req_lesson"])
    await _mark_lesson_complete(engine, seed["student"], seed["opt_lesson"])
    # The D2 writer normally fires from the tracking service; these tests
    # insert lesson_progress directly, so drive it explicitly to promote the
    # course enrollments to 'completed'.
    async with session_factory() as db:
        for course in (seed["req_course"], seed["opt_course"]):
            await enrollments_api.sync_course_completion(
                db, course_id=course, student_id=seed["student"]
            )
        await db.commit()

    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        assert progress.overall_percent == 100
        flipped = await enrollment_service.sync_enrollment_completion(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
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
