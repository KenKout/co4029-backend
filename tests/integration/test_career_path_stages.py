"""Integration tests for career-path STAGES (migration 0070 / rev 3 plan).

Covers, in the order the plan's review points are numbered:

* #1 creating an EMPTY stage on a PUBLISHED path succeeds (two-class
  validation: completeness is a publish-gate rule, not a mutation rule)
* #5 a cross-stage move offsets BOTH position sequences
* #3 the completion writer fires on ``mark_lesson_complete``, not only on a
  lazy read
* rev3 ``unmark_lesson_complete`` demotes the course but does NOT un-latch
  the stage
* #6 reordering a stage into position 1 warns and keeps the stored policy
* #7 deleting a stage with latched progress is blocked (409)
* #2 a snapshot written while the setting still reads 1 is stamped 1, not 2
* D2 ``satisfied`` ⟺ ``course_enrollments.status='completed'``
* unlock matrix, min-optional quota, zero-total stage, cap warns-never-blocks,
  Pattern B (no eager fan-out) and the Start endpoint's guards
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

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
from abridgeai.core.exceptions import AppError, ConflictError, ForbiddenError
from abridgeai.core.runtime_settings import invalidate_settings_cache
from abridgeai.core.security import CurrentUser
from abridgeai.features.career_paths.schemas import (
    CareerPathStageCreate,
    CareerPathStageUpdate,
)
from abridgeai.features.career_paths.services import authoring as authoring_service
from abridgeai.features.career_paths.services import enrollment as enrollment_service
from abridgeai.features.career_paths.services import readiness as readiness_service
from abridgeai.features.career_paths.services import stages as stage_service
from abridgeai.features.enrollments.api import public as enrollments_api
from abridgeai.features.progress.services import tracking as tracking_service


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
    """Org + manager + student + a published path with ONE backfilled stage
    holding a required and an optional course (what migration 0070 leaves).
    """
    s = uuid.uuid4().hex[:8]
    org, manager, student = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    path_id, stage1 = uuid.uuid4(), uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org, "slug": f"cps-{s}", "name": "CP Stage Org"},
        )
        for uid, email in ((manager, f"mgr-{s}@t.local"), (student, f"stu-{s}@t.local")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": email},
            )
        # Career-path enrolments are student-only: enroll_student_in_path
        # (enrollment.py) rejects users without the 'student' role. Give the
        # fixture student the org-scoped role, mirroring test_enrollments.py.
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id) "
                "VALUES (:uid, (SELECT id FROM roles WHERE code = 'student' "
                "AND deleted_at IS NULL), 'organization', :org)"
            ),
            {"uid": student, "org": org},
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
        # Gap 3 (0074): stages/items hang off a VERSION. The fixture path is
        # 'published' (for the path-status guards) but its v1 is a DRAFT —
        # the pre-fork authoring surface the tests mutate directly.
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
                "VALUES (:id, :vid, 1, 'always', 'soft')"
            ),
            {"id": stage1, "vid": version_id},
        )
        for pos, (cid, req) in enumerate(((req_course, True), (opt_course, False)), start=1):
            await conn.execute(
                text(
                    "INSERT INTO career_course_items "
                    "(version_id, course_id, stage_id, position, is_required) "
                    "VALUES (:vid, :cid, :sid, :pos, :req)"
                ),
                {"vid": version_id, "cid": cid, "sid": stage1, "pos": pos, "req": req},
            )

    yield {
        "org": org,
        "manager": manager,
        "student": student,
        "path_id": path_id,
        "version_id": version_id,
        "stage1": stage1,
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
            text("DELETE FROM career_readiness_snapshots WHERE career_path_id = :p"),
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
        await conn.execute(text("DELETE FROM career_path_versions WHERE career_path_id = :p"), {"p": path_id})
        await conn.execute(text("DELETE FROM career_paths WHERE id = :p"), {"p": path_id})
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
        await conn.execute(text("DELETE FROM courses WHERE organization_id = :org"), {"org": org})
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id = :s"),
            {"s": student},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(manager), str(student)]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org})


@pytest_asyncio.fixture
async def formula_v2(engine: AsyncEngine) -> AsyncIterator[None]:
    """Flip ``careerpath.progress_formula_version`` to 2 globally."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO system_settings (setting_key, setting_value_json, organization_id) "
                "VALUES ('careerpath.progress_formula_version', :v, NULL)"
            ),
            {"v": json.dumps(2)},
        )
    invalidate_settings_cache()
    yield
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM system_settings "
                "WHERE setting_key='careerpath.progress_formula_version'"
            )
        )
    invalidate_settings_cache()


async def _enroll(session_factory, seed) -> None:
    async with session_factory() as db:
        await enrollment_service.enroll_student_in_path(
            db,
            career_path_id=seed["path_id"],
            student_id=seed["student"],
            actor=_actor(seed["manager"]),
        )
        await db.commit()


async def _enrollment_status(engine, student, course):
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT status FROM course_enrollments WHERE student_id = :s AND course_id = :c"
                ),
                {"s": student, "c": course},
            )
        ).scalar_one_or_none()


async def _latched_count(engine, path_id) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM student_stage_progress ssp "
                    "JOIN student_career_enrollments sce ON sce.id = ssp.enrollment_id "
                    "WHERE sce.career_path_id = :p"
                ),
                {"p": path_id},
            )
        ).scalar_one()


async def _new_stage(session_factory, seed, **kwargs) -> uuid.UUID:
    async with session_factory() as db:
        stage = await authoring_service.create_stage(
            db,
            seed["path_id"],
            CareerPathStageCreate(**kwargs),
            _actor(seed["manager"]),
        )
        await db.commit()
    return stage.id


# --- #1 two-class validation ------------------------------------------


@pytest.mark.asyncio
async def test_create_empty_stage_on_published_path_succeeds(session_factory, seed) -> None:
    """Guards #1: the authoring flow is create-then-fill. If "every stage has
    >= 1 course" ran on the mutation path you could never add a second stage
    to a published path."""
    stage_id = await _new_stage(session_factory, seed, title="Stage 2")
    async with session_factory() as db:
        stages = await authoring_service.list_path_stages(db, seed["path_id"])
    assert stage_id in {s.id for s in stages}
    assert next(s for s in stages if s.id == stage_id).course_count == 0


@pytest.mark.asyncio
async def test_publish_gate_rejects_empty_stage(session_factory, seed) -> None:
    """The same rule DOES apply at the publish gate."""
    await _new_stage(session_factory, seed, title="Empty")
    async with session_factory() as db:
        with pytest.raises(AppError, match="stage_has_no_courses"):
            await authoring_service.publish_path(db, seed["path_id"], _actor(seed["manager"]))


@pytest.mark.asyncio
async def test_min_optional_above_optional_count_is_rejected_on_mutation(
    session_factory, seed
) -> None:
    """INTEGRITY class: an unsatisfiable quota makes the path unfinishable, so
    it is caught on every mutation rather than at the gate."""
    async with session_factory() as db:
        with pytest.raises(AppError, match="min_optional"):
            await authoring_service.update_stage(
                db,
                seed["path_id"],
                seed["stage1"],
                CareerPathStageUpdate(min_optional_to_complete=5),
                _actor(seed["manager"]),
            )


# --- #5 cross-stage move ----------------------------------------------


@pytest.mark.asyncio
async def test_cross_stage_move_offsets_both_sequences(engine, session_factory, seed) -> None:
    """Guards #5: moving between stages mutates TWO (stage_id, position)
    sequences. Offsetting only one leaves a hole or collides."""
    stage2 = await _new_stage(session_factory, seed, title="S2")
    # Give stage 2 its own course so BOTH sequences are non-empty.
    async with engine.begin() as conn:
        extra, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"x-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            extra,
            stage_id=stage2,
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()

    # Move the required course from stage 1 into stage 2 at position 1.
    async with session_factory() as db:
        await authoring_service.move_course_to_stage(
            db, seed["path_id"], seed["req_course"], stage_id=stage2, position=1
        )
        await db.commit()

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT stage_id, course_id, position FROM career_course_items "
                    "WHERE version_id IN (SELECT id FROM career_path_versions "
                    "WHERE career_path_id = :p) ORDER BY stage_id, position"
                ),
                {"p": seed["path_id"]},
            )
        ).all()

    by_stage: dict[uuid.UUID, list[tuple[uuid.UUID, int]]] = {}
    for stage_id, course_id, position in rows:
        by_stage.setdefault(stage_id, []).append((course_id, position))

    # Source stage reindexed with NO hole: the optional course is now 1, not 2.
    assert by_stage[seed["stage1"]] == [(seed["opt_course"], 1)]
    # Target stage contiguous 1..2 with the moved course first.
    assert by_stage[stage2] == [(seed["req_course"], 1), (extra, 2)]


# --- #3 synchronous completion writer ---------------------------------


@pytest.mark.asyncio
async def test_completion_writer_fires_on_mark_lesson_complete(
    engine, session_factory, seed
) -> None:
    """Guards #3: the D2 writer must run at the mark-complete call site, not
    only on a lazy read. Nothing here ever calls get_my_path_progress."""
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await enrollments_api.ensure_course_enrollment(
            db,
            student_id=seed["student"],
            course_id=seed["req_course"],
            actor_id=seed["manager"],
        )
        await db.commit()
    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"

    async with session_factory() as db:
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()

    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "completed"


@pytest.mark.asyncio
async def test_unmark_demotes_course_but_does_not_unlatch_stage(
    engine, session_factory, seed
) -> None:
    """rev3: completion can go BACKWARD. The course demotes so `satisfied`
    stays honest; the stage latch is append-only and must survive."""
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        for course in (seed["req_course"], seed["opt_course"]):
            await enrollments_api.ensure_course_enrollment(
                db, student_id=seed["student"], course_id=course, actor_id=seed["manager"]
            )
        await db.commit()

    async with session_factory() as db:
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()
    # Stage 1 has 1 required + min_optional 0 → complete → latch written.
    async with session_factory() as db:
        await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    assert await _latched_count(engine, seed["path_id"]) == 1

    async with session_factory() as db:
        await tracking_service.unmark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()

    # Course demoted...
    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"
    # ...stage still latched, and still reported complete.
    assert await _latched_count(engine, seed["path_id"]) == 1
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    stage = next(s for s in progress.stages if s.stage_id == seed["stage1"])
    assert stage.latched is True
    assert stage.complete is True
    assert stage.satisfied_required == 0  # the asymmetry, made visible


# --- D2 satisfied semantics -------------------------------------------


@pytest.mark.asyncio
async def test_satisfied_is_enrollment_status_not_lesson_percent(session_factory, seed) -> None:
    """D2: lesson progress alone does not satisfy a course. Without an
    enrollment row there is no status to be 'completed'."""
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()

    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    row = next(c for c in progress.courses if c.course_id == seed["req_course"])
    assert row.completion_percent == 100
    assert row.is_enrolled is False
    assert row.satisfied is False


# --- unlock matrix ----------------------------------------------------


@pytest.mark.asyncio
async def test_unlock_after_previous_blocks_until_stage_complete(
    session_factory, seed, engine
) -> None:
    stage2 = await _new_stage(session_factory, seed, title="S2", unlock_policy="after_previous")
    async with engine.begin() as conn:
        c2, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"s2-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            c2,
            stage_id=stage2,
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    await _enroll(session_factory, seed)

    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    s1 = next(s for s in progress.stages if s.position == 1)
    s2 = next(s for s in progress.stages if s.position == 2)
    assert s1.unlocked is True  # stage 1 always unlocked
    assert s2.unlocked is False  # gated on stage 1

    # Satisfy stage 1's required course → stage 2 unlocks.
    async with session_factory() as db:
        await enrollments_api.ensure_course_enrollment(
            db,
            student_id=seed["student"],
            course_id=seed["req_course"],
            actor_id=seed["manager"],
        )
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    assert next(s for s in progress.stages if s.position == 2).unlocked is True


@pytest.mark.asyncio
async def test_stage_one_is_unlocked_even_when_policy_says_after_previous(
    session_factory, seed, engine
) -> None:
    """D5: position 1's policy is inert — a locked first stage could never be
    started. The stored value is preserved, not normalised."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE career_path_stages SET unlock_policy='after_previous_required' "
                "WHERE id = :sid"
            ),
            {"sid": seed["stage1"]},
        )
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    stage = next(s for s in progress.stages if s.position == 1)
    assert stage.unlocked is True
    assert stage.unlock_policy == "after_previous_required"  # preserved, not rewritten


# --- #6 reorder warns -------------------------------------------------


@pytest.mark.asyncio
async def test_reorder_into_position_one_warns_and_keeps_policy(session_factory, seed) -> None:
    """Guards #6: reorder must WARN, not silently rewrite manager intent."""
    stage2 = await _new_stage(session_factory, seed, title="S2", unlock_policy="after_previous")
    async with session_factory() as db:
        result = await authoring_service.reorder_stages(
            db, seed["path_id"], [stage2, seed["stage1"]], _actor(seed["manager"])
        )
        await db.commit()

    codes = {w.code for w in result.warnings}
    assert "stage_becomes_implicitly_unlocked" in codes
    moved = next(s for s in result.stages if s.id == stage2)
    assert moved.position == 1
    # Policy preserved verbatim — that is the whole point of D5.
    assert moved.unlock_policy == "after_previous"


# --- #7 deletion guards ----------------------------------------------


@pytest.mark.asyncio
async def test_delete_stage_with_courses_is_blocked(session_factory, seed) -> None:
    async with session_factory() as db:
        with pytest.raises(ConflictError, match="stage_in_use"):
            await authoring_service.delete_stage(
                db, seed["path_id"], seed["stage1"], _actor(seed["manager"])
            )


@pytest.mark.asyncio
async def test_delete_stage_with_latched_progress_is_blocked(engine, session_factory, seed) -> None:
    """Guards #7: emptying a stage is not enough. Deleting a stage students
    have LATCHED orphans the latch and moves their progress bar."""
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await enrollments_api.ensure_course_enrollment(
            db,
            student_id=seed["student"],
            course_id=seed["req_course"],
            actor_id=seed["manager"],
        )
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()
    async with session_factory() as db:
        await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    assert await _latched_count(engine, seed["path_id"]) == 1

    # Move every course out, so ONLY the latch blocks the delete.
    stage2 = await _new_stage(session_factory, seed, title="S2")
    async with session_factory() as db:
        for course in (seed["req_course"], seed["opt_course"]):
            await authoring_service.move_course_to_stage(
                db, seed["path_id"], course, stage_id=stage2, position=None
            )
        await db.commit()

    async with session_factory() as db:
        with pytest.raises(ConflictError, match="stage_in_use"):
            await authoring_service.delete_stage(
                db, seed["path_id"], seed["stage1"], _actor(seed["manager"])
            )


# --- #2 formula version stamping --------------------------------------


@pytest.mark.asyncio
async def test_snapshot_in_dark_window_is_stamped_1_not_2(engine, session_factory, seed) -> None:
    """Guards #2: while the setting still reads 1, snapshots must be stamped
    1. A column default of 2 would mislabel the entire dark window."""
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await readiness_service.snapshot_enrollment(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()

    async with engine.connect() as conn:
        versions = (
            (
                await conn.execute(
                    text(
                        "SELECT formula_version FROM career_readiness_snapshots "
                        "WHERE career_path_id = :p"
                    ),
                    {"p": seed["path_id"]},
                )
            )
            .scalars()
            .all()
        )
    assert list(versions) == [1]


@pytest.mark.asyncio
async def test_snapshot_after_cutover_is_stamped_2(
    engine, session_factory, seed, formula_v2
) -> None:
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await readiness_service.snapshot_enrollment(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()

    async with engine.connect() as conn:
        versions = (
            (
                await conn.execute(
                    text(
                        "SELECT formula_version FROM career_readiness_snapshots "
                        "WHERE career_path_id = :p"
                    ),
                    {"p": seed["path_id"]},
                )
            )
            .scalars()
            .all()
        )
    assert list(versions) == [2]


# --- progress denominator --------------------------------------------


@pytest.mark.asyncio
async def test_denominator_counts_required_plus_min_optional(
    engine, session_factory, seed, formula_v2
) -> None:
    """2 required + "min 1 of 3 optional" → stage_total 3, not 5."""
    async with engine.begin() as conn:
        extra_req, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"r2-{uuid.uuid4().hex[:6]}"
        )
        opt2, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"o2-{uuid.uuid4().hex[:6]}"
        )
        opt3, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"o3-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            extra_req,
            stage_id=seed["stage1"],
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        for cid in (opt2, opt3):
            await authoring_service.add_course_to_path(
                db,
                seed["path_id"],
                cid,
                stage_id=seed["stage1"],
                position=None,
                is_required=False,
                actor=_actor(seed["manager"]),
            )
        await authoring_service.update_stage(
            db,
            seed["path_id"],
            seed["stage1"],
            CareerPathStageUpdate(min_optional_to_complete=1),
            _actor(seed["manager"]),
        )
        await db.commit()

    await _enroll(session_factory, seed)
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    stage = next(s for s in progress.stages if s.stage_id == seed["stage1"])
    assert stage.required_count == 2
    assert stage.optional_count == 3
    assert stage.stage_total == 3  # 2 required + min_optional 1
    assert progress.formula_version == 2


@pytest.mark.asyncio
async def test_zero_total_stage_is_complete_and_excluded(
    session_factory, seed, formula_v2, engine
) -> None:
    """A stage with no required courses and no quota is complete by
    definition and must not sit in the denominator dragging the path down."""
    stage2 = await _new_stage(session_factory, seed, title="Electives", unlock_policy="always")
    async with engine.begin() as conn:
        opt, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"e-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            opt,
            stage_id=stage2,
            position=None,
            is_required=False,
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    await _enroll(session_factory, seed)

    # Satisfy stage 1's only required course → path should read 100%.
    async with session_factory() as db:
        await enrollments_api.ensure_course_enrollment(
            db,
            student_id=seed["student"],
            course_id=seed["req_course"],
            actor_id=seed["manager"],
        )
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()

    zero_stage = next(s for s in progress.stages if s.stage_id == stage2)
    assert zero_stage.stage_total == 0
    assert zero_stage.complete is True
    assert progress.overall_percent == 100.0


# --- cap: warns, never blocks ----------------------------------------


@pytest.mark.asyncio
async def test_concurrency_cap_warns_but_never_blocks(engine, session_factory, seed) -> None:
    """D4: the cap is an attention signal. Even at/over the cap the Start
    endpoint succeeds — `hard` enforcement governs stage lock only."""
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE career_paths SET max_concurrent = 1 WHERE id = :p"),
            {"p": seed["path_id"]},
        )
        await conn.execute(
            text(
                "UPDATE career_path_stages SET enforcement='hard' WHERE version_id IN "
                "(SELECT id FROM career_path_versions WHERE career_path_id = :p)"
            ),
            {"p": seed["path_id"]},
        )
    await _enroll(session_factory, seed)

    async with session_factory() as db:
        first = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert first.created is True

    # Second start puts the student at 2 active > cap 1: still succeeds.
    async with session_factory() as db:
        second = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["opt_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert second.created is True
    assert second.over_concurrency_cap is True
    assert await _enrollment_status(engine, seed["student"], seed["opt_course"]) == "active"


# --- Pattern B + Start endpoint --------------------------------------


@pytest.mark.asyncio
async def test_path_enrollment_no_longer_fans_out_courses(engine, session_factory, seed) -> None:
    """Pattern B: assigning a path grants access to the PATH only."""
    await _enroll(session_factory, seed)
    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) is None
    assert await _enrollment_status(engine, seed["student"], seed["opt_course"]) is None


@pytest.mark.asyncio
async def test_start_is_idempotent_and_reactivates_dropped(engine, session_factory, seed) -> None:
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        first = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert first.created is True

    async with session_factory() as db:
        again = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert again.created is False

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE course_enrollments SET status='dropped' "
                "WHERE student_id=:s AND course_id=:c"
            ),
            {"s": seed["student"], "c": seed["req_course"]},
        )
    async with session_factory() as db:
        revived = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert revived.created is True
    assert await _enrollment_status(engine, seed["student"], seed["req_course"]) == "active"

    async with engine.connect() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM course_enrollments WHERE student_id=:s AND course_id=:c"
                ),
                {"s": seed["student"], "c": seed["req_course"]},
            )
        ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_start_requires_active_path_enrollment(session_factory, seed) -> None:
    """The carve-out rests on a manager-made assignment: no path enrollment,
    no self-enroll."""
    async with session_factory() as db:
        with pytest.raises(ForbiddenError, match="active_path"):
            await enrollment_service.start_course_in_path(
                db,
                career_path_id=seed["path_id"],
                course_id=seed["req_course"],
                student_id=seed["student"],
            )


@pytest.mark.asyncio
async def test_start_refuses_course_in_locked_stage(session_factory, seed, engine) -> None:
    """A LOCKED stage blocks Start only under ``enforcement='hard'``.

    Previously this test created a stage with the DDL default (`soft`) and
    asserted the block, which encoded the bug: Start tested `not
    target.unlocked` directly and blocked every enforcement level, so a
    manager who picked "Show a warning, still allow" got a hard block.
    """
    stage2 = await _new_stage(
        session_factory,
        seed,
        title="S2",
        unlock_policy="after_previous",
        enforcement="hard",
    )
    async with engine.begin() as conn:
        c2, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"lk-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            c2,
            stage_id=stage2,
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    await _enroll(session_factory, seed)

    async with session_factory() as db:
        with pytest.raises(ForbiddenError, match="stage_locked"):
            await enrollment_service.start_course_in_path(
                db, career_path_id=seed["path_id"], course_id=c2, student_id=seed["student"]
            )


@pytest.mark.parametrize("enforcement", ["soft", "advisory"])
@pytest.mark.asyncio
async def test_start_allows_locked_stage_under_non_hard_enforcement(
    session_factory, seed, engine, enforcement
) -> None:
    """`soft` and `advisory` must ALLOW the Start and warn instead.

    This is the behaviour test the helper's unit test could not give: it
    asserted `stage_is_hard_locked()` in isolation while Start never called
    the helper at all. Driven through the ROUTE so the response the student
    actually receives is what is pinned.

    `soft` is the DDL default, so before the fix every stage a manager created
    through the UI hard-blocked students while the settings popover said it
    only warns.
    """
    stage2 = await _new_stage(
        session_factory,
        seed,
        title="S2",
        unlock_policy="after_previous",
        enforcement=enforcement,
    )
    async with engine.begin() as conn:
        c2, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"sf-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            c2,
            stage_id=stage2,
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    await _enroll(session_factory, seed)

    # Stage 2 really is locked: stage 1's required course is unfinished.
    async with session_factory() as db:
        evals = await stage_service.evaluate_stages(
            db,
            version_id=seed["version_id"],
            student_id=seed["student"],
            enrollment_id=None,
        )
    target = next(ev for ev in evals if ev.stage.id == stage2)
    assert target.unlocked is False
    assert stage_service.stage_is_hard_locked(target) is False

    from abridgeai.features.career_paths.routers import learner as learner_router  # noqa: PLC0415

    async with session_factory() as db:
        result = await learner_router.start_course_in_path(
            seed["path_id"],
            c2,
            _actor(seed["student"]),
            db,
        )

    # Allowed...
    assert result.created is True
    assert result.stage_id == stage2
    # ...and the student is TOLD they are working ahead.
    assert result.stage_locked_warning is True

    # The enrollment was really written, not just reported.
    assert await _enrollment_status(engine, seed["student"], c2) == "active"


@pytest.mark.asyncio
async def test_start_in_unlocked_stage_sets_no_locked_warning(session_factory, seed) -> None:
    """The warning must not cry wolf on a normal Start."""
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        result = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert result.stage_locked_warning is False


@pytest.mark.asyncio
async def test_start_reports_active_in_path_count(session_factory, seed) -> None:
    """Start must report the count its own cap warning is phrased around.

    The FE toast interpolates `{{count}}`; StartCourseResult carried no
    number, so the FE hardcoded 0 and the warning read "you have 0 courses
    open in this path". The service already computes `active_in_path` right
    there — it just was not returned.

    Counted AFTER this Start, so the first Start reports 1, not 0.
    """
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        first = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert first.active_in_path == 1, "the course just started must be counted"

    async with session_factory() as db:
        second = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["opt_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert second.active_in_path == 2
    # No cap configured on the seed path, so nothing to warn about.
    assert second.max_concurrent is None
    assert second.over_concurrency_cap is False


@pytest.mark.asyncio
async def test_start_reports_cap_alongside_the_count(engine, session_factory, seed) -> None:
    """With a cap set, the warning has both numbers it needs."""
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE career_paths SET max_concurrent = 1 WHERE id = :p"),
            {"p": seed["path_id"]},
        )
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["req_course"],
            student_id=seed["student"],
        )
        await db.commit()
    async with session_factory() as db:
        second = await enrollment_service.start_course_in_path(
            db,
            career_path_id=seed["path_id"],
            course_id=seed["opt_course"],
            student_id=seed["student"],
        )
        await db.commit()
    assert second.max_concurrent == 1
    assert second.active_in_path == 2
    # Advisory only — the Start still succeeded.
    assert second.over_concurrency_cap is True
    assert second.created is True


@pytest.mark.asyncio
async def test_start_refuses_course_outside_the_path(session_factory, seed, engine) -> None:
    """A student cannot name an arbitrary course id."""
    async with engine.begin() as conn:
        outside, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"out-{uuid.uuid4().hex[:6]}"
        )
    await _enroll(session_factory, seed)
    from abridgeai.core.exceptions import NotFoundError  # noqa: PLC0415

    async with session_factory() as db:
        with pytest.raises(NotFoundError):
            await enrollment_service.start_course_in_path(
                db,
                career_path_id=seed["path_id"],
                course_id=outside,
                student_id=seed["student"],
            )


# --- latch survives manager edits ------------------------------------


@pytest.mark.asyncio
async def test_latch_survives_elective_removal(engine, session_factory, seed) -> None:
    """The latch's other job: a manager raising the bar after a student
    passed must not un-complete them."""
    async with session_factory() as db:
        await authoring_service.update_stage(
            db,
            seed["path_id"],
            seed["stage1"],
            CareerPathStageUpdate(min_optional_to_complete=1),
            _actor(seed["manager"]),
        )
        await db.commit()
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        for course, lesson in (
            (seed["req_course"], seed["req_lesson"]),
            (seed["opt_course"], seed["opt_lesson"]),
        ):
            await enrollments_api.ensure_course_enrollment(
                db, student_id=seed["student"], course_id=course, actor_id=seed["manager"]
            )
            await tracking_service.mark_lesson_complete(
                db, user_id=seed["student"], lesson_id=lesson
            )
        await db.commit()
    async with session_factory() as db:
        progress = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    assert next(s for s in progress.stages if s.stage_id == seed["stage1"]).complete is True
    assert await _latched_count(engine, seed["path_id"]) == 1

    # Manager removes the elective the student used to meet the quota.
    async with session_factory() as db:
        await authoring_service.remove_course_from_path(
            db, seed["path_id"], seed["opt_course"], _actor(seed["manager"])
        )
        await db.commit()

    async with session_factory() as db:
        after = await enrollment_service.get_my_path_progress(
            db, career_path_id=seed["path_id"], student_id=seed["student"]
        )
        await db.commit()
    stage = next(s for s in after.stages if s.stage_id == seed["stage1"])
    assert stage.latched is True
    assert stage.complete is True


# --- evaluator unit-ish checks ---------------------------------------


@pytest.mark.asyncio
async def test_progress_endpoint_commits_the_latch_unconditionally(
    engine, session_factory, seed
) -> None:
    """Regression: the progress GET must commit even when the ENROLLMENT did
    not flip.

    The route used to commit only ``if flipped`` (enrollment reached 100%),
    which silently rolled back the stage latch written moments earlier by
    ``get_my_path_progress``. A stage then reported ``complete`` in the
    response while having no latch row — and a manager could delete a stage
    students had genuinely finished. Caught by live API verification, not by
    the service-level tests (they commit explicitly).

    Here stage 1 completes but the PATH does not (stage 2 is unfinished), so
    ``sync_enrollment_completion`` returns False — exactly the case the old
    code failed to persist.
    """
    stage2 = await _new_stage(session_factory, seed, title="S2", unlock_policy="after_previous")
    async with engine.begin() as conn:
        c2, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"pc-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            c2,
            stage_id=stage2,
            position=None,
            is_required=True,
            actor=_actor(seed["manager"]),
        )
        await db.commit()
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        await enrollments_api.ensure_course_enrollment(
            db,
            student_id=seed["student"],
            course_id=seed["req_course"],
            actor_id=seed["manager"],
        )
        await tracking_service.mark_lesson_complete(
            db, user_id=seed["student"], lesson_id=seed["req_lesson"]
        )
        await db.commit()

    # Drive the ROUTE, not the service, so the route's commit policy is what
    # is under test.
    from abridgeai.features.career_paths.routers import learner as learner_router  # noqa: PLC0415

    async with session_factory() as db:
        progress = await learner_router.get_my_career_path_progress(
            seed["path_id"],
            SimpleNamespace(user_id=seed["student"]),
            db,
        )
    assert next(s for s in progress.stages if s.stage_id == seed["stage1"]).complete is True
    # The path is NOT complete, so the enrollment did not flip...
    assert progress.overall_percent < 100
    # ...and the latch must still have been persisted.
    assert await _latched_count(engine, seed["path_id"]) == 1


@pytest.mark.asyncio
async def test_hard_enforcement_only_blocks_when_locked(session_factory, seed) -> None:
    await _enroll(session_factory, seed)
    async with session_factory() as db:
        evals = await stage_service.evaluate_stages(
            db,
            version_id=seed["version_id"],
            student_id=seed["student"],
            enrollment_id=None,
        )
    # Stage 1 is unlocked, so even 'hard' must not block.
    assert stage_service.stage_is_hard_locked(evals[0]) is False


# --- required/optional is editable after attach -----------------------


@pytest.mark.asyncio
async def test_update_path_course_flips_required_to_optional(
    session_factory, seed
) -> None:
    """`is_required` used to be write-once at attach time, which meant a stage
    could never hold an optional course and `min_optional_to_complete` could
    only ever be 0. It is now patchable in place."""
    async with session_factory() as db:
        rows = await authoring_service.update_path_course(
            db, seed["path_id"], seed["req_course"], is_required=False
        )
        await db.commit()
    link = next(r for r in rows if r.course_id == seed["req_course"])
    assert link.is_required is False
    # Position is preserved — the old remove + re-add workaround re-appended.
    assert link.stage_id == seed["stage1"]


@pytest.mark.asyncio
async def test_update_path_course_omitted_field_is_untouched(
    session_factory, seed
) -> None:
    """A patch carrying only `is_required` must not reset `satisfied_by`."""
    async with session_factory() as db:
        rows = await authoring_service.update_path_course(
            db, seed["path_id"], seed["req_course"], is_required=False
        )
        await db.commit()
    link = next(r for r in rows if r.course_id == seed["req_course"])
    assert link.satisfied_by == "completion"


@pytest.mark.asyncio
async def test_flip_to_required_rejected_when_it_breaks_optional_quota(
    engine, session_factory, seed
) -> None:
    """Flipping optional -> required LOWERS optional_count. When that pushes
    the count below `min_optional_to_complete` the stage could never complete,
    so the patch must be rejected rather than leaving it uncompletable."""
    async with engine.begin() as conn:
        opt, _ = await _course_with_lesson(
            conn, org=seed["org"], owner=seed["manager"], slug=f"q-{uuid.uuid4().hex[:6]}"
        )
    async with session_factory() as db:
        await authoring_service.add_course_to_path(
            db,
            seed["path_id"],
            opt,
            stage_id=seed["stage1"],
            position=None,
            is_required=False,
            actor=_actor(seed["manager"]),
        )
        await authoring_service.update_stage(
            db,
            seed["path_id"],
            seed["stage1"],
            # The stage holds TWO optional courses (the fixture's opt_course
            # plus the one added below); min=1 would still be satisfiable
            # after a single flip, so the quota must demand both.
            CareerPathStageUpdate(min_optional_to_complete=2),
            _actor(seed["manager"]),
        )
        await db.commit()

    # The last remaining optional course cannot become required: min is 2.
    async with session_factory() as db:
        with pytest.raises(AppError, match="min_optional"):
            await authoring_service.update_path_course(
                db, seed["path_id"], opt, is_required=True
            )
