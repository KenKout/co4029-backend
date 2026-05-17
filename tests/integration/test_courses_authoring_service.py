"""Integration tests for ``features.courses.services.authoring`` (T3.5).

Covers the locked-decision invariants:

* §A5 — :func:`add_lesson` auto-creates a ``ModuleItem`` row.
* §A6 — :func:`reorder_module_items` two-phase swap pattern handles
  full-cycle reorder without violating ``uq_module_items_position``.
* :func:`delete_lesson_resource` uses
  :func:`abridgeai.core.db.recursive_delete.soft_delete_cascade`.
* CRUD lifecycle: create → publish → archive transitions.

Tests use raw SQL for setup / teardown (avoiding cross-feature ORM
collisions during ``Base.metadata`` flush) and consume the service
layer directly. ``soft_delete_cascade`` operates on attached ORM
instances so :func:`delete_lesson_resource` is exercised end-to-end.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio
from alembic import command
from alembic.config import Config

# Register stub tables for FK targets owned by features not yet ported
# (learning_materials -> Phase 4; quizzes -> Phase 5; interview_configs -> Phase 6;
# storage_objects already declared in identity). The ORM unit-of-work walks every
# FK target at flush time; if the target Table is not in Base.metadata it raises
# NoReferencedTableError. Stub Tables keep the dependency graph resolvable.
from sqlalchemy import (  # noqa: E402  -- stub registration must follow ORM imports
    Column,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID  # noqa: E402
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.schemas import (
    CourseCreate,
    LessonCreate,
    LessonResourceCreate,
)
from abridgeai.features.courses.services import authoring as authoring_service

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
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"auth-{suffix}", "name": "Authoring Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"auth-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Authoring Test Course', 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module 1', 1, 'draft')"
            ),
            {"id": module_id, "course": course_id},
        )

    yield {
        "owner_id": owner_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM lesson_resources "
                "WHERE lesson_id IN (SELECT id FROM lessons WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE course_id = :c"),
            {"c": course_id},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE organization_id = :o"),
            {"o": org_id},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def test_add_lesson_auto_creates_module_item(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        owner = _actor(scenario["owner_id"])
        payload = LessonCreate(
            module_id=scenario["module_id"],
            slug="lesson-auto-mi",
            title="Auto-MI Lesson",
        )
        lesson = await authoring_service.add_lesson(session, scenario["module_id"], payload, owner)
        await session.commit()

        rows = (
            (
                await session.execute(
                    text(
                        "SELECT id, item_type, lesson_id, position FROM module_items "
                        "WHERE module_id = :m"
                    ),
                    {"m": scenario["module_id"]},
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["item_type"] == "lesson"
    assert row["lesson_id"] == lesson.id
    assert row["position"] == 1


async def test_reorder_module_items_two_phase_swap(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        owner = _actor(scenario["owner_id"])
        lessons = []
        for idx in range(3):
            payload = LessonCreate(
                module_id=scenario["module_id"],
                slug=f"lesson-{idx}",
                title=f"Lesson {idx}",
            )
            lesson = await authoring_service.add_lesson(
                session, scenario["module_id"], payload, owner
            )
            lessons.append(lesson)
        await session.commit()

        items_before = (
            (
                await session.execute(
                    text(
                        "SELECT id, position FROM module_items "
                        "WHERE module_id = :m ORDER BY position"
                    ),
                    {"m": scenario["module_id"]},
                )
            )
            .mappings()
            .all()
        )
        item_ids = [row["id"] for row in items_before]
        assert [row["position"] for row in items_before] == [1, 2, 3]

        reordered = [item_ids[2], item_ids[0], item_ids[1]]
        await authoring_service.reorder_module_items(
            session, scenario["module_id"], reordered, owner
        )
        await session.commit()

        items_after = (
            (
                await session.execute(
                    text(
                        "SELECT id, position FROM module_items "
                        "WHERE module_id = :m ORDER BY position"
                    ),
                    {"m": scenario["module_id"]},
                )
            )
            .mappings()
            .all()
        )

    assert [row["id"] for row in items_after] == reordered
    assert [row["position"] for row in items_after] == [1, 2, 3]


async def test_delete_lesson_resource_uses_soft_delete_cascade(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        owner = _actor(scenario["owner_id"])
        lesson = await authoring_service.add_lesson(
            session,
            scenario["module_id"],
            LessonCreate(
                module_id=scenario["module_id"],
                slug="lesson-with-res",
                title="Lesson w/ Resource",
            ),
            owner,
        )
        storage_object_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO storage_objects "
                "(id, bucket, object_key, mime_type, size_bytes) "
                "VALUES (:id, 'b', :k, 'application/pdf', 0)"
            ),
            {"id": storage_object_id, "k": f"k-{storage_object_id.hex[:6]}"},
        )
        resource = await authoring_service.add_lesson_resource(
            session,
            lesson.id,
            LessonResourceCreate(
                lesson_id=lesson.id,
                title="Doc",
                resource_type="pdf",
                storage_object_id=storage_object_id,
                position=1,
            ),
            owner,
        )
        await session.commit()

        await authoring_service.delete_lesson_resource(session, resource.id, owner)
        await session.commit()

        result = (
            (
                await session.execute(
                    text("SELECT deleted_at, deleted_by FROM lesson_resources WHERE id = :id"),
                    {"id": resource.id},
                )
            )
            .mappings()
            .one_or_none()
        )

    assert result is not None
    assert result["deleted_at"] is not None
    assert result["deleted_by"] == owner.user_id

    async with session_factory() as session:
        absent = (
            await session.execute(
                text("SELECT id FROM lesson_resources WHERE id = :id"),
                {"id": resource.id},
            )
        ).scalar_one_or_none()
    assert absent is not None


async def test_publish_course_widens_status(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        owner = _actor(scenario["owner_id"])
        suffix = uuid.uuid4().hex[:8]
        new_course = await authoring_service.create_course(
            session,
            CourseCreate(
                organization_id=scenario["org_id"],
                owner_user_id=scenario["owner_id"],
                slug=f"create-{suffix}",
                title="Lifecycle Course",
            ),
            owner,
        )
        await session.commit()
        assert new_course.status == "draft"

        published = await authoring_service.publish_course(session, new_course.id, owner)
        await session.commit()
        assert published.status == "published"


async def test_archive_course_sets_status_archived(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        owner = _actor(scenario["owner_id"])
        archived = await authoring_service.archive_course(session, scenario["course_id"], owner)
        await session.commit()
    assert archived.status == "archived"


async def test_set_module_prerequisites_clears_existing(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
    engine: AsyncEngine,
) -> None:
    other_module = uuid.uuid4()
    third_module = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Module 2', 2, 'draft')"
            ),
            {"m": other_module, "c": scenario["course_id"]},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Module 3', 3, 'draft')"
            ),
            {"m": third_module, "c": scenario["course_id"]},
        )

    try:
        async with session_factory() as session:
            owner = _actor(scenario["owner_id"])
            await authoring_service.set_module_prerequisites(
                session, scenario["module_id"], [other_module], owner
            )
            await session.commit()

            await authoring_service.set_module_prerequisites(
                session, scenario["module_id"], [third_module], owner
            )
            await session.commit()

            rows = (
                (
                    await session.execute(
                        text(
                            "SELECT prerequisite_module_id FROM module_prerequisites "
                            "WHERE module_id = :m"
                        ),
                        {"m": scenario["module_id"]},
                    )
                )
                .scalars()
                .all()
            )
        assert list(rows) == [third_module]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM module_prerequisites WHERE module_id = :m"),
                {"m": scenario["module_id"]},
            )
            await conn.execute(
                text("DELETE FROM modules WHERE id IN (:m1, :m2)"),
                {"m1": other_module, "m2": third_module},
            )


def test_no_sqlalchemy_imports_in_services() -> None:
    services_dir = (
        Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "courses" / "services"
    )
    pattern = re.compile(r"^(from|import) sqlalchemy")
    for path in services_dir.glob("*.py"):
        for line in path.read_text().splitlines():
            assert not pattern.match(line), f"sqlalchemy import found in {path}: {line!r}"
