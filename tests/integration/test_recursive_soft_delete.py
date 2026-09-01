from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.identity.models  # noqa: F401, E402  -- register users / storage_objects FK targets
import abridgeai.features.interviews.models  # noqa: F401, E402  -- register interview_configs FK target
import abridgeai.features.materials.models  # noqa: F401, E402  -- register learning_materials FK target
import abridgeai.features.quizzes.models  # noqa: F401, E402  -- register quizzes FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.access_control.models import OrgUnit  # noqa: E402
from abridgeai.features.courses.models import Course, Lesson, Module  # noqa: E402


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
                "slug": f"rsd-{org_id.hex[:8]}",
                "name": "Recursive Soft Delete Test Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"rsd-{owner_id.hex[:8]}@test.local"},
        )
    yield org_id, owner_id
    async with engine.begin() as conn:
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
        await conn.execute(
            text("DELETE FROM org_units WHERE organization_id = :org"),
            {"org": org_id},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def _seed_chain(
    session: AsyncSession, org_id: uuid.UUID, owner_id: uuid.UUID
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    course = Course(
        organization_id=org_id,
        owner_user_id=owner_id,
        slug=f"c-{uuid.uuid4().hex[:8]}",
        title="Test Course",
    )
    module = Module(course=course, title="Test Module", position=1)
    lesson = Lesson(module=module, slug=f"l-{uuid.uuid4().hex[:8]}", title="Test Lesson")
    session.add(course)
    await session.flush()
    return course.id, module.id, lesson.id


async def test_simple_cascade(session_factory: async_sessionmaker[AsyncSession], org_owner) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course_id, module_id, lesson_id = await _seed_chain(session, org_id, owner_id)
        course = await session.get(Course, course_id)
        assert course is not None
        result = await soft_delete_cascade(session, course)
        await session.commit()

    assert result.count == 3
    assert {tbl for (tbl, _id) in result.affected} == {"courses", "modules", "lessons"}

    async with session_factory() as session:
        active = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one_or_none()
        assert active is None
        active_module = (
            await session.execute(select(Module).where(Module.id == module_id))
        ).scalar_one_or_none()
        assert active_module is None
        active_lesson = (
            await session.execute(select(Lesson).where(Lesson.id == lesson_id))
        ).scalar_one_or_none()
        assert active_lesson is None

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Course).where(Course.id == course_id).execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        assert rows[0][0].deleted_at is not None


async def test_query_filter_respects_cascade(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course_id, module_id, lesson_id = await _seed_chain(session, org_id, owner_id)
        course = await session.get(Course, course_id)
        assert course is not None
        await soft_delete_cascade(session, course)
        await session.commit()

    async with session_factory() as session:
        lessons = (
            (await session.execute(select(Lesson).where(Lesson.module_id == module_id)))
            .scalars()
            .all()
        )
        assert lessons == []


async def test_atomicity_on_exception(
    session_factory: async_sessionmaker[AsyncSession],
    org_owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course_id, module_id, lesson_id = await _seed_chain(session, org_id, owner_id)
        await session.commit()

    async with session_factory() as session:
        course = await session.get(Course, course_id)
        assert course is not None

        original_refresh = session.refresh
        call_count = {"n": 0}

        async def patched_refresh(obj, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise RuntimeError("simulated mid-traverse failure")
            return await original_refresh(obj, *args, **kwargs)

        monkeypatch.setattr(session, "refresh", patched_refresh)

        with pytest.raises(RuntimeError, match="simulated"):
            await soft_delete_cascade(session, course)
        await session.rollback()

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Course, Module, Lesson)
                .join(Module, Module.course_id == Course.id)
                .join(Lesson, Lesson.module_id == Module.id)
                .where(Course.id == course_id)
                .execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        c, m, le = rows[0]
        assert c.deleted_at is None
        assert m.deleted_at is None
        assert le.deleted_at is None


@pytest.mark.skip(
    reason=(
        "0094_flat_faculties made this scenario unconstructable. "
        "ck_org_units_live_faculty_root requires every LIVE org_unit to be "
        "a top-level faculty with parent_unit_id IS NULL, so a cycle can no "
        "longer be seeded — the CHECK rejects both the non-faculty insert and "
        "the parent UPDATE. org_units is also the only self-referential table "
        "soft_delete_cascade traverses (course_learning_outcomes has the "
        "parent column but no `children` relationship), so there is nowhere "
        "to port this to. The cycle guard remains in soft_delete_cascade as "
        "defence; this test is kept, skipped, rather than deleted so the "
        "coverage gap stays visible."
    )
)
async def test_cycle_safety(
    session_factory: async_sessionmaker[AsyncSession], engine: AsyncEngine, org_owner
) -> None:
    org_id, _owner_id = org_owner
    unit_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name) "
                "VALUES (:id, :org, 'department', 'Self-loop unit')"
            ),
            {"id": unit_id, "org": org_id},
        )
        await conn.execute(
            text("UPDATE org_units SET parent_unit_id = :id WHERE id = :id"),
            {"id": unit_id},
        )

    async with session_factory() as session:
        unit = await session.get(OrgUnit, unit_id)
        assert unit is not None
        result = await soft_delete_cascade(session, unit)
        await session.commit()

    assert result.count == 1
    assert result.affected == [("org_units", unit_id)]

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(OrgUnit).where(OrgUnit.id == unit_id).execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        assert rows[0][0].deleted_at is not None


async def test_dry_run(session_factory: async_sessionmaker[AsyncSession], org_owner) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course_id, module_id, lesson_id = await _seed_chain(session, org_id, owner_id)
        await session.commit()

    async with session_factory() as session:
        course = await session.get(Course, course_id)
        assert course is not None
        result = await soft_delete_cascade(session, course, dry_run=True)
        await session.commit()

    assert result.count == 3
    assert {tbl for (tbl, _id) in result.affected} == {"courses", "modules", "lessons"}

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Course, Module, Lesson)
                .join(Module, Module.course_id == Course.id)
                .join(Lesson, Lesson.module_id == Module.id)
                .where(Course.id == course_id)
                .execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        c, m, le = rows[0]
        assert c.deleted_at is None
        assert m.deleted_at is None
        assert le.deleted_at is None


async def test_actor_id_propagation(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course_id, module_id, lesson_id = await _seed_chain(session, org_id, owner_id)
        course = await session.get(Course, course_id)
        assert course is not None
        await soft_delete_cascade(session, course, actor_id=owner_id)
        await session.commit()

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Course, Module, Lesson)
                .join(Module, Module.course_id == Course.id)
                .join(Lesson, Lesson.module_id == Module.id)
                .where(Course.id == course_id)
                .execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        c, m, le = rows[0]
        assert c.deleted_by == owner_id
        assert m.deleted_by == owner_id
        assert le.deleted_by == owner_id


_ = IntegrityError
