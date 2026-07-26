"""T7 characterization tests for ``features/courses/routers/_deps.py``.

Asserts the 5 ORM-migrated sub-resource → course walks return the same
course id as the legacy raw ``text()`` SELECTs they replaced, and that
the auto-applied soft-delete loader filter correctly hides tombstoned
lessons.

Fixtures are inserted via raw ``text()`` (T4 lint allows it for test
fixtures) so we can pre-stamp ``deleted_at`` and bypass the soft-delete
loader for the setup phase. Assertions go through the migrated ORM
helpers (``_resolve_*``) which DO respect the loader filter.
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

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
# The ``_resolve_*`` helpers moved out of ``routers._deps`` into the public
# ``queries.resolution`` module (dropping the underscore) when the deps
# layer was split; the snapshot semantics are unchanged.
from abridgeai.features.courses.queries.resolution import (
    resolve_lesson_to_course,
    resolve_module_item_to_course,
    resolve_module_to_course,
    resolve_outcome_to_course,
    resolve_resource_to_course,
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
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Build a 5-lesson fixture across 2 courses with 1 soft-deleted leaf.

    Layout:
      org_t7
      ├── course_alpha (live, owner_alpha)
      │   └── module_a1
      │       ├── lesson_a1 (live)         -- happy-path target
      │       │   └── resource_a1 (live)
      │       ├── lesson_a2 (live)
      │       └── lesson_a3 (deleted_at=NOW())  -- soft-delete probe
      │       module_a1.module_item_a1 -> lesson_a1
      │       outcome_a1 -> course_alpha
      └── course_beta (live, owner_beta)
          └── module_b1
              └── lesson_b1 (live)         -- separate-course probe
    """
    suffix = uuid.uuid4().hex[:8]
    org_id = uuid.uuid4()
    owner_alpha = uuid.uuid4()
    owner_beta = uuid.uuid4()
    course_alpha = uuid.uuid4()
    course_beta = uuid.uuid4()
    module_a1 = uuid.uuid4()
    module_b1 = uuid.uuid4()
    lesson_a1 = uuid.uuid4()
    lesson_a2 = uuid.uuid4()
    lesson_a3_softdel = uuid.uuid4()
    lesson_b1 = uuid.uuid4()
    resource_a1 = uuid.uuid4()
    module_item_a1 = uuid.uuid4()
    outcome_a1 = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) "
                "VALUES (:a, :ea, 'active'), (:b, :eb, 'active')"
            ),
            {
                "a": owner_alpha,
                "b": owner_beta,
                "ea": f"t7-alpha-{suffix}@abridgeai.local",
                "eb": f"t7-beta-{suffix}@abridgeai.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {"id": org_id, "slug": f"t7-org-{suffix}", "name": "T7 Snapshot Org"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES "
                "(:ca, :org, :oa, :sa, 'Course Alpha', 'draft'), "
                "(:cb, :org, :ob, :sb, 'Course Beta', 'draft')"
            ),
            {
                "ca": course_alpha,
                "cb": course_beta,
                "org": org_id,
                "oa": owner_alpha,
                "ob": owner_beta,
                "sa": f"t7-alpha-{suffix}",
                "sb": f"t7-beta-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, position, title, status) VALUES "
                "(:m1, :ca, 1, 'Module A1', 'draft'), "
                "(:m2, :cb, 1, 'Module B1', 'draft')"
            ),
            {"m1": module_a1, "m2": module_b1, "ca": course_alpha, "cb": course_beta},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, lesson_type) VALUES "
                "(:la1, :ma, :sa1, 'Lesson A1', 'video'), "
                "(:la2, :ma, :sa2, 'Lesson A2', 'reading'), "
                "(:lb1, :mb, :sb1, 'Lesson B1', 'video')"
            ),
            {
                "la1": lesson_a1,
                "la2": lesson_a2,
                "lb1": lesson_b1,
                "ma": module_a1,
                "mb": module_b1,
                "sa1": f"less-a1-{suffix}",
                "sa2": f"less-a2-{suffix}",
                "sb1": f"less-b1-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, lesson_type, deleted_at) "
                "VALUES (:la3, :ma, :sa3, 'Lesson A3 (soft-deleted)', 'video', NOW())"
            ),
            {
                "la3": lesson_a3_softdel,
                "ma": module_a1,
                "sa3": f"less-a3-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lesson_resources "
                "(id, lesson_id, title, resource_type, position) "
                "VALUES (:rid, :lid, 'Resource A1', 'pdf', 1)"
            ),
            {"rid": resource_a1, "lid": lesson_a1},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items "
                "(id, module_id, item_type, lesson_id, position) "
                "VALUES (:mid, :ma, 'lesson', :lid, 1)"
            ),
            {"mid": module_item_a1, "ma": module_a1, "lid": lesson_a1},
        )
        await conn.execute(
            text(
                "INSERT INTO course_learning_outcomes "
                "(id, course_id, position, outcome_text) "
                "VALUES (:oid, :cid, 1, 'Outcome A1')"
            ),
            {"oid": outcome_a1, "cid": course_alpha},
        )

    data: dict[str, uuid.UUID] = {
        "org_id": org_id,
        "owner_alpha": owner_alpha,
        "owner_beta": owner_beta,
        "course_alpha": course_alpha,
        "course_beta": course_beta,
        "module_a1": module_a1,
        "module_b1": module_b1,
        "lesson_a1": lesson_a1,
        "lesson_a2": lesson_a2,
        "lesson_a3_softdel": lesson_a3_softdel,
        "lesson_b1": lesson_b1,
        "resource_a1": resource_a1,
        "module_item_a1": module_item_a1,
        "outcome_a1": outcome_a1,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE id = :id"), {"id": outcome_a1}
        )
        await conn.execute(text("DELETE FROM module_items WHERE id = :id"), {"id": module_item_a1})
        await conn.execute(text("DELETE FROM lesson_resources WHERE id = :id"), {"id": resource_a1})
        await conn.execute(
            text("DELETE FROM lessons WHERE id = ANY(:ids)"),
            {"ids": [lesson_a1, lesson_a2, lesson_a3_softdel, lesson_b1]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [module_a1, module_b1]},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [course_alpha, course_beta]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [owner_alpha, owner_beta]},
        )


@pytest.mark.asyncio
async def test_module_walk_returns_course(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict[str, uuid.UUID]
) -> None:
    async with session_factory() as session:
        result = await resolve_module_to_course(session, scenario["module_a1"])
    assert result is not None
    course_id, owner_user_id = result
    assert course_id == scenario["course_alpha"]
    assert owner_user_id == scenario["owner_alpha"]


@pytest.mark.asyncio
async def test_lesson_walk_returns_course(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict[str, uuid.UUID]
) -> None:
    async with session_factory() as session:
        a1 = await resolve_lesson_to_course(session, scenario["lesson_a1"])
        a2 = await resolve_lesson_to_course(session, scenario["lesson_a2"])
        b1 = await resolve_lesson_to_course(session, scenario["lesson_b1"])
    assert a1 is not None
    assert a1[0] == scenario["course_alpha"]
    assert a1[1] == scenario["owner_alpha"]
    assert a2 is not None
    assert a2[0] == scenario["course_alpha"]
    assert b1 is not None
    assert b1[0] == scenario["course_beta"]
    assert b1[1] == scenario["owner_beta"]


@pytest.mark.asyncio
async def test_resource_walk_returns_course(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict[str, uuid.UUID]
) -> None:
    async with session_factory() as session:
        result = await resolve_resource_to_course(session, scenario["resource_a1"])
    assert result is not None
    assert result[0] == scenario["course_alpha"]
    assert result[1] == scenario["owner_alpha"]


@pytest.mark.asyncio
async def test_module_item_walk_returns_course(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict[str, uuid.UUID]
) -> None:
    async with session_factory() as session:
        result = await resolve_module_item_to_course(session, scenario["module_item_a1"])
    assert result is not None
    assert result[0] == scenario["course_alpha"]
    assert result[1] == scenario["owner_alpha"]


@pytest.mark.asyncio
async def test_outcome_walk_returns_course(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict[str, uuid.UUID]
) -> None:
    async with session_factory() as session:
        result = await resolve_outcome_to_course(session, scenario["outcome_a1"])
    assert result is not None
    assert result[0] == scenario["course_alpha"]
    assert result[1] == scenario["owner_alpha"]


@pytest.mark.asyncio
async def test_softdeleted_lesson_excluded_by_loader_filter(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict[str, uuid.UUID]
) -> None:
    """Soft-deleted lesson must be invisible to the migrated walk.

    The do_orm_execute listener auto-applies ``deleted_at IS NULL`` to
    every SELECT touching a SoftDeleteMixin table, so the walk should
    return ``None`` even though the row physically exists.
    """
    async with session_factory() as session:
        result = await resolve_lesson_to_course(session, scenario["lesson_a3_softdel"])
    assert result is None


@pytest.mark.asyncio
async def test_unknown_lesson_returns_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        result = await resolve_lesson_to_course(session, uuid.uuid4())
    assert result is None
