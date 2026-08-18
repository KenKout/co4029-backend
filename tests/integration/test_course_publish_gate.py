"""The course publish gate: a course with no gradeable unit cannot publish.

A gradeable unit is a published lesson, a published quiz, or a published
interview config. A course with zero of them can never be completed by any
student — the completion writer refuses to promote an empty course — so as a
required course on a career path it locks its stage, and every stage behind
it, permanently.

That rule already guarded career-path publication, but only once someone put
the course on a path: possibly weeks after the course went live, with an error
naming the path rather than the course. These tests pin the gate at course
publish, which is the first moment the system can tell the manager.

Both doors into `published` are covered. `POST /publish` is the obvious one;
`PATCH {"status": "published"}` is the one that makes a gate applied to only
the first door decorative.
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
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.enrollments.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import ConflictError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.schemas import CourseUpdate
from abridgeai.features.courses.services import authoring as authoring_service

pytestmark = pytest.mark.asyncio


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
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
async def seed(engine: AsyncEngine) -> AsyncIterator[dict]:
    """An org, a manager, and a DRAFT course with one module and no items."""
    s = uuid.uuid4().hex[:8]
    org, manager = uuid.uuid4(), uuid.uuid4()
    course_id, module_id = uuid.uuid4(), uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org, "slug": f"pg-{s}", "name": "Publish Gate Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": manager, "email": f"mgr-{s}@t.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Gate Course', 'draft')"
            ),
            {"id": course_id, "org": org, "owner": manager, "slug": f"gate-{s}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, 'M1', 1, 'published')"
            ),
            {"id": module_id, "cid": course_id},
        )
        # This file proves the CONTENT / OUTCOME gates in isolation, so it
        # disables the teacher-staffing dimension: the org's min is 1 and the
        # manager is staffed as the Course Instructor, leaving publish to hinge
        # only on the units/outcomes gates under test.
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, course_id, "
                "granted_by, course_role) "
                "SELECT :aid, :uid, r.id, 'course', :org, :cid, :uid, "
                "'course_instructor' FROM roles r WHERE r.code = 'teacher'"
            ),
            {
                "aid": uuid.uuid4(),
                "uid": manager,
                "org": org,
                "cid": course_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO system_settings (organization_id, setting_key, setting_value_json) "
                "VALUES (:org, 'courses.min_teachers_per_course', '1')"
            ),
            {"org": org},
        )

    yield {
        "org": org,
        "manager": manager,
        "course_id": course_id,
        "module_id": module_id,
        "slug_suffix": s,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lessons WHERE module_id IN "
                 "(SELECT id FROM modules WHERE course_id = :c)"),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :c"),
            {"c": course_id},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE course_id = :c"),
            {"c": course_id},
        )
        await conn.execute(
            text("DELETE FROM system_settings WHERE organization_id = :o"),
            {"o": org},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": manager})
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def _add_published_lesson(engine: AsyncEngine, seed: dict) -> uuid.UUID:
    lesson_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:id, :mid, :slug, 'L1', 'published', 'video')"
            ),
            {"id": lesson_id, "mid": seed["module_id"], "slug": f"l-{seed['slug_suffix']}"},
        )
    return lesson_id


async def _add_outcome(engine: AsyncEngine, seed: dict) -> uuid.UUID:
    """Satisfy the learning-outcome gate.

    Kept separate from :func:`_add_published_lesson` on purpose: this file
    exists to prove each gate independently, and a helper that quietly
    satisfied both would make every "blocked" test pass for the wrong reason.
    """
    outcome_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO course_learning_outcomes "
                "(id, course_id, position, outcome_text) "
                "VALUES (:id, :cid, 1, 'State an outcome')"
            ),
            {"id": outcome_id, "cid": seed["course_id"]},
        )
    return outcome_id


async def _status(engine: AsyncEngine, course_id: uuid.UUID) -> str:
    async with engine.begin() as conn:
        return (
            await conn.execute(
                text("SELECT status FROM courses WHERE id = :id"), {"id": course_id}
            )
        ).scalar_one()


async def test_publish_refuses_course_with_no_gradeable_units(
    session_factory, seed, engine
) -> None:
    """The gate itself. An empty course is unpublishable."""
    async with session_factory() as db:
        with pytest.raises(ConflictError, match="course_has_no_gradeable_units"):
            await authoring_service.publish_course(
                db, seed["course_id"], _actor(seed["manager"])
            )
    # And nothing was written on the way out.
    assert await _status(engine, seed["course_id"]) == "draft"


async def test_publish_allows_course_with_one_published_lesson(
    session_factory, seed, engine
) -> None:
    """One unit is enough — the gate is a floor, not a quality bar."""
    await _add_published_lesson(engine, seed)
    await _add_outcome(engine, seed)
    async with session_factory() as db:
        course = await authoring_service.publish_course(
            db, seed["course_id"], _actor(seed["manager"])
        )
        await db.commit()
    assert course.status == "published"
    assert await _status(engine, seed["course_id"]) == "published"


async def test_a_draft_lesson_is_not_a_gradeable_unit(session_factory, seed, engine) -> None:
    """Draft content cannot satisfy the gate: a student cannot complete it."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:id, :mid, :slug, 'Draft L', 'draft', 'video')"
            ),
            {
                "id": uuid.uuid4(),
                "mid": seed["module_id"],
                "slug": f"dl-{seed['slug_suffix']}",
            },
        )
    async with session_factory() as db:
        with pytest.raises(ConflictError, match="course_has_no_gradeable_units"):
            await authoring_service.publish_course(
                db, seed["course_id"], _actor(seed["manager"])
            )


async def test_patch_status_published_applies_the_same_gate(
    session_factory, seed, engine
) -> None:
    """The second door.

    `POST /publish` is not the only way into `published`; `PATCH {"status":
    "published"}` reaches the same state. A gate on only one of them is
    decorative — the manager just uses the other door.
    """
    async with session_factory() as db:
        with pytest.raises(ConflictError, match="course_has_no_gradeable_units"):
            await authoring_service.update_course(
                db,
                seed["course_id"],
                CourseUpdate(status="published"),
                _actor(seed["manager"]),
            )
    assert await _status(engine, seed["course_id"]) == "draft"


async def test_patch_status_published_succeeds_once_a_unit_exists(
    session_factory, seed, engine
) -> None:
    await _add_published_lesson(engine, seed)
    async with session_factory() as db:
        course = await authoring_service.update_course(
            db,
            seed["course_id"],
            CourseUpdate(status="published"),
            _actor(seed["manager"]),
        )
        await db.commit()
    assert course.status == "published"


async def test_republishing_a_published_course_skips_the_gate(
    session_factory, seed, engine
) -> None:
    """Re-publish is a no-op and must not fail.

    Otherwise an already-live course whose only lesson was later unpublished
    would start throwing on an operation that changes nothing.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET status = 'published' WHERE id = :id"),
            {"id": seed["course_id"]},
        )
    async with session_factory() as db:
        course = await authoring_service.publish_course(
            db, seed["course_id"], _actor(seed["manager"])
        )
        await db.commit()
    assert course.status == "published"


async def test_patching_other_fields_on_a_draft_never_hits_the_gate(
    session_factory, seed
) -> None:
    """The gate is about publishing, not editing. Authoring a draft course with
    no content yet is the normal state and must stay unobstructed."""
    async with session_factory() as db:
        course = await authoring_service.update_course(
            db,
            seed["course_id"],
            CourseUpdate(description="still drafting"),
            _actor(seed["manager"]),
        )
        await db.commit()
    assert course.description == "still drafting"
    assert course.status == "draft"


async def test_publish_blocked_when_content_exists_but_no_outcome(
    session_factory, seed, engine
) -> None:
    """The outcome gate is independent of the content gate.

    Content present, outcomes absent: the failure must name outcomes. A merged
    check would either let this through or blame the wrong thing, and the
    manager would go looking in the curriculum editor for a problem that lives
    on the settings screen.
    """
    await _add_published_lesson(engine, seed)
    async with session_factory() as db:
        with pytest.raises(ConflictError, match="course_has_no_learning_outcomes"):
            await authoring_service.publish_course(
                db, seed["course_id"], _actor(seed["manager"])
            )
    assert await _status(engine, seed["course_id"]) == "draft"


async def test_content_gate_reports_first_when_both_are_unmet(
    session_factory, seed, engine
) -> None:
    """With neither gate met, the manager hears about content first.

    Order matters for a checklist a human reads one 409 at a time: no content
    means no student can ever finish the course, while no outcomes means it is
    merely undocumented. Fix the blocking one first.
    """
    await _add_outcome(engine, seed)
    async with session_factory() as db:
        with pytest.raises(ConflictError, match="course_has_no_gradeable_units"):
            await authoring_service.publish_course(
                db, seed["course_id"], _actor(seed["manager"])
            )
    assert await _status(engine, seed["course_id"]) == "draft"
