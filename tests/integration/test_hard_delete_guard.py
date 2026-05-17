from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.identity.models  # noqa: F401  -- register users / auth_sessions FK targets
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_configs
import abridgeai.features.materials.models  # noqa: F401  -- register learning_materials
import abridgeai.features.quizzes.models  # noqa: F401  -- register quizzes
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.courses.models import Course, Lesson, Module
from abridgeai.features.identity.models import AuthSession


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def org_owner(engine: AsyncEngine):
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {
                "id": org_id,
                "slug": f"hdg-{org_id.hex[:8]}",
                "name": "Hard Delete Guard Test Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"hdg-{owner_id.hex[:8]}@test.local"},
        )
    yield org_id, owner_id
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = :uid"),
            {"uid": owner_id},
        )
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id IN "
                "(SELECT id FROM courses WHERE organization_id = :org))"
            ),
            {"org": org_id},
        )
        await conn.execute(
            text(
                "DELETE FROM modules WHERE course_id IN "
                "(SELECT id FROM courses WHERE organization_id = :org)"
            ),
            {"org": org_id},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE organization_id = :org"),
            {"org": org_id},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def _seed_course(session: AsyncSession, org_id: uuid.UUID, owner_id: uuid.UUID) -> Course:
    course = Course(
        organization_id=org_id,
        owner_user_id=owner_id,
        slug=f"c-{uuid.uuid4().hex[:8]}",
        title="HDG Course",
    )
    session.add(course)
    await session.flush()
    return course


async def test_db_delete_softdelete_row_raises(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course = await _seed_course(session, org_id, owner_id)
        await session.delete(course)
        with pytest.raises(RuntimeError, match="soft_delete_cascade"):
            await session.flush()
        await session.rollback()


async def test_db_delete_non_softdelete_row_succeeds(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    session_id = uuid.uuid4()
    async with session_factory() as session:
        auth = AuthSession(
            id=session_id,
            user_id=owner_id,
            refresh_token_hash=f"hash-{session_id.hex}",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        session.add(auth)
        await session.flush()

        await session.delete(auth)
        await session.flush()
        await session.commit()

    async with session_factory() as session:
        rows = (
            await session.execute(select(AuthSession).where(AuthSession.id == session_id))
        ).all()
        assert rows == []


async def test_soft_delete_cascade_still_works(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course = await _seed_course(session, org_id, owner_id)
        module = Module(course=course, title="HDG Module", position=1)
        lesson = Lesson(module=module, slug=f"l-{uuid.uuid4().hex[:8]}", title="HDG Lesson")
        session.add(module)
        session.add(lesson)
        await session.flush()
        course_id = course.id

        result = await soft_delete_cascade(session, course, actor_id=owner_id)
        await session.flush()
        await session.commit()

    assert result.count == 3
    assert {tbl for (tbl, _id) in result.affected} == {"courses", "modules", "lessons"}

    async with session_factory() as session:
        row = (
            await session.execute(
                select(Course).where(Course.id == course_id).execution_options(include_deleted=True)
            )
        ).scalar_one()
        assert row.deleted_at is not None
        assert row.deleted_by == owner_id


async def test_cascade_does_not_propagate_delete_after_narrowing(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course = await _seed_course(session, org_id, owner_id)
        module = Module(course=course, title="HDG Module", position=1)
        session.add(module)
        await session.flush()
        await session.commit()

    from sqlalchemy import inspect as sa_inspect

    course_mapper = sa_inspect(Course)
    modules_rel = course_mapper.relationships["modules"]
    assert modules_rel.cascade.save_update is True, "save-update must be kept (db.add cascade)"
    assert modules_rel.cascade.merge is True
    assert modules_rel.cascade.refresh_expire is True
    assert modules_rel.cascade.expunge is True
    assert modules_rel.cascade.delete is False, (
        "delete cascade must be absent so db.delete(course) does NOT propagate "
        "the unit-of-work delete to children"
    )
    assert modules_rel.cascade.delete_orphan is False, (
        "delete-orphan must be absent so detaching a module does not trigger DELETE"
    )

    module_mapper = sa_inspect(Module)
    for rel_key in ("lessons", "items"):
        rel = module_mapper.relationships[rel_key]
        assert rel.cascade.save_update is True
        assert rel.cascade.delete is False, f"Module.{rel_key} must not cascade delete"
        assert rel.cascade.delete_orphan is False

    lesson_mapper = sa_inspect(Lesson)
    resources_rel = lesson_mapper.relationships["resources"]
    assert resources_rel.cascade.save_update is True
    assert resources_rel.cascade.delete is False
    assert resources_rel.cascade.delete_orphan is False
