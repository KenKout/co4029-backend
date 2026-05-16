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

from abridgeai.core.audit.context import current_actor_var
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade

from ._test_models import (
    _Course,
    _Lesson,
    _LessonResource,
    _Module,
    _ModuleItem,
    _OrgUnit,
)


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
                "slug": f"sdc-{org_id.hex[:8]}",
                "name": "Soft Delete Cascade Test Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"sdc-{owner_id.hex[:8]}@test.local"},
        )
    yield org_id, owner_id
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = :id"),
            {"id": owner_id},
        )
        await conn.execute(
            text(
                "DELETE FROM lesson_resources WHERE lesson_id IN "
                "(SELECT id FROM lessons WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id IN "
                "(SELECT id FROM courses WHERE organization_id = :org)))"
            ),
            {"org": org_id},
        )
        await conn.execute(
            text(
                "DELETE FROM module_items WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id IN "
                "(SELECT id FROM courses WHERE organization_id = :org))"
            ),
            {"org": org_id},
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
        await conn.execute(
            text("DELETE FROM org_units WHERE organization_id = :org"),
            {"org": org_id},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def _seed_full_chain(
    session: AsyncSession,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    *,
    slug: str | None = None,
    resource_count: int = 1,
) -> dict[str, uuid.UUID | list[uuid.UUID]]:
    course = _Course(
        organization_id=org_id,
        owner_user_id=owner_id,
        slug=slug or f"c-{uuid.uuid4().hex[:8]}",
        title="Test Course",
    )
    module = _Module(course=course, title="Test Module", position=1)
    lesson = _Lesson(module=module, slug=f"l-{uuid.uuid4().hex[:8]}", title="Test Lesson")
    resources = [
        _LessonResource(
            lesson=lesson,
            title=f"Resource {i}",
            resource_type="link",
            position=i,
        )
        for i in range(1, resource_count + 1)
    ]
    session.add(course)
    await session.flush()
    item = _ModuleItem(
        module_id=module.id,
        item_type="lesson",
        lesson_id=lesson.id,
        position=1,
    )
    session.add(item)
    await session.flush()
    return {
        "course_id": course.id,
        "module_id": module.id,
        "lesson_id": lesson.id,
        "module_item_id": item.id,
        "resource_ids": [r.id for r in resources],
    }


async def test_nested_cascade_4_levels(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=2)
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        result = await soft_delete_cascade(session, course)
        await session.commit()

    assert result.count == 6
    affected_tables = {tbl for (tbl, _id) in result.affected}
    assert affected_tables == {
        "courses",
        "modules",
        "lessons",
        "lesson_resources",
        "module_items",
    }
    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert ids["course_id"] in affected_ids
    assert ids["module_id"] in affected_ids
    assert ids["lesson_id"] in affected_ids
    assert ids["module_item_id"] in affected_ids
    for rid in ids["resource_ids"]:
        assert rid in affected_ids

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(_LessonResource)
                    .where(_LessonResource.lesson_id == ids["lesson_id"])
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
        assert all(r.deleted_at is not None for r in rows)


async def test_partial_cascade_does_not_cross_siblings(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        course = _Course(
            organization_id=org_id,
            owner_user_id=owner_id,
            slug=f"c-{uuid.uuid4().hex[:8]}",
            title="Two Modules",
        )
        module_a = _Module(course=course, title="Module A", position=1)
        module_b = _Module(course=course, title="Module B", position=2)
        lesson_a = _Lesson(module=module_a, slug=f"la-{uuid.uuid4().hex[:8]}", title="Lesson A")
        lesson_b = _Lesson(module=module_b, slug=f"lb-{uuid.uuid4().hex[:8]}", title="Lesson B")
        for i in range(1, 4):
            _LessonResource(
                lesson=lesson_a,
                title=f"A-Res {i}",
                resource_type="link",
                position=i,
            )
        session.add(course)
        await session.flush()
        lesson_a_id = lesson_a.id
        lesson_b_id = lesson_b.id
        module_b_id = module_b.id
        await session.commit()

    async with session_factory() as session:
        lesson = await session.get(_Lesson, lesson_a_id)
        assert lesson is not None
        result = await soft_delete_cascade(session, lesson)
        await session.commit()

    assert result.count == 4
    assert {tbl for (tbl, _id) in result.affected} == {
        "lessons",
        "lesson_resources",
    }

    async with session_factory() as session:
        sibling = (
            await session.execute(select(_Lesson).where(_Lesson.id == lesson_b_id))
        ).scalar_one_or_none()
        assert sibling is not None
        assert sibling.deleted_at is None

        sibling_module = (
            await session.execute(select(_Module).where(_Module.id == module_b_id))
        ).scalar_one_or_none()
        assert sibling_module is not None
        assert sibling_module.deleted_at is None


async def test_uniqueness_reuse_after_soft_delete(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    target_slug = f"reuse-{uuid.uuid4().hex[:8]}"
    duplicate_slug = f"dup-{uuid.uuid4().hex[:8]}"

    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, slug=target_slug, resource_count=1)
        await session.commit()
        first_course_id = ids["course_id"]

    async with session_factory() as session:
        course = await session.get(_Course, first_course_id)
        assert course is not None
        await soft_delete_cascade(session, course)
        await session.commit()

    async with session_factory() as session:
        recreated = _Course(
            organization_id=org_id,
            owner_user_id=owner_id,
            slug=target_slug,
            title="Recreated",
        )
        session.add(recreated)
        await session.commit()
        recreated_id = recreated.id

    async with session_factory() as session:
        active_with_slug = (
            (await session.execute(select(_Course).where(_Course.slug == target_slug)))
            .scalars()
            .all()
        )
        assert len(active_with_slug) == 1
        assert active_with_slug[0].id == recreated_id

        all_with_slug = (
            (
                await session.execute(
                    select(_Course)
                    .where(_Course.slug == target_slug)
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(all_with_slug) == 2

    async with session_factory() as session:
        first_dup = _Course(
            organization_id=org_id,
            owner_user_id=owner_id,
            slug=duplicate_slug,
            title="Active Y",
        )
        session.add(first_dup)
        await session.commit()

    async with session_factory() as session:
        second_dup = _Course(
            organization_id=org_id,
            owner_user_id=owner_id,
            slug=duplicate_slug,
            title="Should Conflict",
        )
        session.add(second_dup)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


async def test_query_filter_respects_cascade(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=2)
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        await soft_delete_cascade(session, course)
        await session.commit()

    async with session_factory() as session:
        active_courses = (
            (await session.execute(select(_Course).where(_Course.id == ids["course_id"])))
            .scalars()
            .all()
        )
        assert active_courses == []

        active_lessons = (
            (await session.execute(select(_Lesson).where(_Lesson.module_id == ids["module_id"])))
            .scalars()
            .all()
        )
        assert active_lessons == []

        active_resources = (
            (
                await session.execute(
                    select(_LessonResource).where(_LessonResource.lesson_id == ids["lesson_id"])
                )
            )
            .scalars()
            .all()
        )
        assert active_resources == []


async def test_opt_out_admin_query_shows_descendants(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=2)
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        await soft_delete_cascade(session, course)
        await session.commit()

    async with session_factory() as session:
        course_rows = (
            (
                await session.execute(
                    select(_Course)
                    .where(_Course.id == ids["course_id"])
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(course_rows) == 1
        assert course_rows[0].deleted_at is not None

        lesson_rows = (
            (
                await session.execute(
                    select(_Lesson)
                    .where(_Lesson.module_id == ids["module_id"])
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(lesson_rows) == 1
        assert lesson_rows[0].deleted_at is not None

        resource_rows = (
            (
                await session.execute(
                    select(_LessonResource)
                    .where(_LessonResource.lesson_id == ids["lesson_id"])
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(resource_rows) == 2
        assert all(r.deleted_at is not None for r in resource_rows)


async def test_audit_trail_deleted_by_populated(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner

    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=1)
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        await soft_delete_cascade(session, course, actor_id=owner_id)
        await session.commit()

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(_Course, _Module, _Lesson, _LessonResource)
                .join(_Module, _Module.course_id == _Course.id)
                .join(_Lesson, _Lesson.module_id == _Module.id)
                .join(
                    _LessonResource,
                    _LessonResource.lesson_id == _Lesson.id,
                )
                .where(_Course.id == ids["course_id"])
                .execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        c, m, le, r = rows[0]
        assert c.deleted_by == owner_id
        assert m.deleted_by == owner_id
        assert le.deleted_by == owner_id
        assert r.deleted_by == owner_id

    async with session_factory() as session:
        ids2 = await _seed_full_chain(session, org_id, owner_id, resource_count=1)
        await session.commit()

    token = current_actor_var.set(owner_id)
    try:
        async with session_factory() as session:
            course2 = await session.get(_Course, ids2["course_id"])
            assert course2 is not None
            await soft_delete_cascade(session, course2)
            await session.commit()
    finally:
        current_actor_var.reset(token)

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(_Course, _Module, _Lesson, _LessonResource)
                .join(_Module, _Module.course_id == _Course.id)
                .join(_Lesson, _Lesson.module_id == _Module.id)
                .join(
                    _LessonResource,
                    _LessonResource.lesson_id == _Lesson.id,
                )
                .where(_Course.id == ids2["course_id"])
                .execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        c, m, le, r = rows[0]
        assert c.deleted_by == owner_id
        assert m.deleted_by == owner_id
        assert le.deleted_by == owner_id
        assert r.deleted_by == owner_id


async def test_dry_run_then_real_cascade(
    session_factory: async_sessionmaker[AsyncSession], org_owner
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=2)
        await session.commit()

    async with session_factory() as session:
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        plan = await soft_delete_cascade(session, course, dry_run=True)
        await session.commit()

    assert plan.count == 6
    assert {tbl for (tbl, _id) in plan.affected} == {
        "courses",
        "modules",
        "lessons",
        "lesson_resources",
        "module_items",
    }

    async with session_factory() as session:
        active_course = (
            await session.execute(select(_Course).where(_Course.id == ids["course_id"]))
        ).scalar_one_or_none()
        assert active_course is not None
        assert active_course.deleted_at is None

    async with session_factory() as session:
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        real = await soft_delete_cascade(session, course)
        await session.commit()

    assert real.count == 6
    async with session_factory() as session:
        active_course = (
            await session.execute(select(_Course).where(_Course.id == ids["course_id"]))
        ).scalar_one_or_none()
        assert active_course is None


async def test_cycle_safety_two_node_ring(
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    org_owner,
) -> None:
    org_id, _owner_id = org_owner
    unit_a_id = uuid.uuid4()
    unit_b_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name) "
                "VALUES (:id, :org, 'department', 'Ring A')"
            ),
            {"id": unit_a_id, "org": org_id},
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, parent_unit_id, unit_type, name) "
                "VALUES (:id, :org, :pid, 'department', 'Ring B')"
            ),
            {"id": unit_b_id, "org": org_id, "pid": unit_a_id},
        )
        await conn.execute(
            text("UPDATE org_units SET parent_unit_id = :pid WHERE id = :id"),
            {"id": unit_a_id, "pid": unit_b_id},
        )

    async with session_factory() as session:
        unit_a = await session.get(_OrgUnit, unit_a_id)
        assert unit_a is not None
        result = await soft_delete_cascade(session, unit_a)
        await session.commit()

    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert affected_ids == {unit_a_id, unit_b_id}
    assert result.count == 2

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(_OrgUnit)
                    .where(_OrgUnit.id.in_([unit_a_id, unit_b_id]))
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
        assert all(r.deleted_at is not None for r in rows)


async def test_atomicity_flush_failure_rolls_back(
    session_factory: async_sessionmaker[AsyncSession],
    org_owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, owner_id = org_owner
    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=2)
        await session.commit()

    async with session_factory() as session:
        course = await session.get(_Course, ids["course_id"])
        assert course is not None

        original_flush = session.flush
        flush_calls = {"n": 0}

        async def patched_flush(*args, **kwargs):
            flush_calls["n"] += 1
            if flush_calls["n"] >= 1:
                raise RuntimeError("simulated flush failure")
            return await original_flush(*args, **kwargs)

        monkeypatch.setattr(session, "flush", patched_flush)

        with pytest.raises(RuntimeError, match="simulated flush"):
            await soft_delete_cascade(session, course)
        await session.rollback()

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(_Course, _Module, _Lesson)
                .join(_Module, _Module.course_id == _Course.id)
                .join(_Lesson, _Lesson.module_id == _Module.id)
                .where(_Course.id == ids["course_id"])
                .execution_options(include_deleted=True)
            )
        ).all()
        assert len(rows) == 1
        c, m, le = rows[0]
        assert c.deleted_at is None
        assert m.deleted_at is None
        assert le.deleted_at is None

        resource_rows = (
            (
                await session.execute(
                    select(_LessonResource)
                    .where(_LessonResource.lesson_id == ids["lesson_id"])
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(resource_rows) == 2
        assert all(r.deleted_at is None for r in resource_rows)


async def test_cascade_does_not_touch_hard_delete_table(
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    org_owner,
) -> None:
    org_id, owner_id = org_owner
    session_id = uuid.uuid4()
    refresh_hash = f"hash-{uuid.uuid4().hex}"
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :hash, NOW() + INTERVAL '1 day')"
            ),
            {"id": session_id, "uid": owner_id, "hash": refresh_hash},
        )

    async with session_factory() as session:
        ids = await _seed_full_chain(session, org_id, owner_id, resource_count=1)
        course = await session.get(_Course, ids["course_id"])
        assert course is not None
        result = await soft_delete_cascade(session, course)
        await session.commit()

    affected_tables = {tbl for (tbl, _id) in result.affected}
    assert "auth_sessions" not in affected_tables
    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert session_id not in affected_ids

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT id, user_id, refresh_token_hash, revoked_at "
                    "FROM auth_sessions WHERE id = :id"
                ),
                {"id": session_id},
            )
        ).one_or_none()
        assert row is not None
        assert row.id == session_id
        assert row.user_id == owner_id
        assert row.refresh_token_hash == refresh_hash
        assert row.revoked_at is None

        col_check = (
            await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'auth_sessions' AND column_name = 'deleted_at'"
                )
            )
        ).all()
        assert col_check == []
