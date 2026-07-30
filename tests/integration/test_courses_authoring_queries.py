from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
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
from abridgeai.features.courses.queries import (
    get_course_with_content_tree,
    get_course_for_authoring,
    list_all_lesson_resources,
    list_courses_for_owner,
    list_lessons_for_authoring,
    list_modules_for_authoring,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace(
            "postgresql+psycopg://", "postgresql+psycopg_async://", 1
        )
    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://", "postgresql+psycopg_async://", 1
        )
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
    eng = create_async_engine(
        _async_url(get_settings().database_url), pool_pre_ping=True
    )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def fixture_data(engine: AsyncEngine) -> AsyncIterator[dict]:
    org = uuid.uuid4()
    owner = uuid.uuid4()
    other_owner = uuid.uuid4()
    pub_course = uuid.uuid4()
    draft_course = uuid.uuid4()
    archived_course = uuid.uuid4()
    other_owner_course = uuid.uuid4()
    pub_module = uuid.uuid4()
    pub_lesson = uuid.uuid4()
    draft_lesson = uuid.uuid4()
    item_published = uuid.uuid4()
    item_draft_target = uuid.uuid4()
    resource_visible = uuid.uuid4()
    resource_hidden = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Authoring Org', 'active')"
            ),
            {"id": org, "slug": f"auth-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) VALUES "
                "(:o, :oe, 'active'), (:p, :pe, 'active')"
            ),
            {
                "o": owner,
                "p": other_owner,
                "oe": f"owner-{owner.hex[:8]}@test.local",
                "pe": f"peer-{other_owner.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) VALUES "
                "(:c1, :o, :u, :s1, 'Pub', 'published'), "
                "(:c2, :o, :u, :s2, 'Draft', 'draft'), "
                "(:c3, :o, :u, :s3, 'Archived', 'archived'), "
                "(:c4, :o, :p, :s4, 'Other Owner', 'published')"
            ),
            {
                "c1": pub_course,
                "c2": draft_course,
                "c3": archived_course,
                "c4": other_owner_course,
                "o": org,
                "u": owner,
                "p": other_owner,
                "s1": f"pub-{pub_course.hex[:6]}",
                "s2": f"drf-{draft_course.hex[:6]}",
                "s3": f"arc-{archived_course.hex[:6]}",
                "s4": f"oth-{other_owner_course.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Pub Module', 1, 'published')"
            ),
            {"m": pub_module, "c": pub_course},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) VALUES "
                "(:l1, :m, 'pub-lesson', 'Pub Lesson', 'published'), "
                "(:l2, :m, 'draft-lesson', 'Draft Lesson', 'draft')"
            ),
            {"l1": pub_lesson, "l2": draft_lesson, "m": pub_module},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items "
                "(id, module_id, item_type, lesson_id, position) VALUES "
                "(:i1, :m, 'lesson', :l1, 1), "
                "(:i2, :m, 'lesson', :l2, 2)"
            ),
            {
                "i1": item_published,
                "i2": item_draft_target,
                "m": pub_module,
                "l1": pub_lesson,
                "l2": draft_lesson,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lesson_resources "
                "(id, lesson_id, title, resource_type, position, visible_to_students) VALUES "
                "(:r1, :l, 'Visible', 'pdf', 1, TRUE), "
                "(:r2, :l, 'Hidden', 'pdf', 2, FALSE)"
            ),
            {"r1": resource_visible, "r2": resource_hidden, "l": pub_lesson},
        )

    data = {
        "org": org,
        "owner": owner,
        "other_owner": other_owner,
        "pub_course": pub_course,
        "draft_course": draft_course,
        "archived_course": archived_course,
        "other_owner_course": other_owner_course,
        "pub_module": pub_module,
        "pub_lesson": pub_lesson,
        "draft_lesson": draft_lesson,
        "item_published": item_published,
        "item_draft_target": item_draft_target,
        "resource_visible": resource_visible,
        "resource_hidden": resource_hidden,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_resources WHERE lesson_id = :l"),
            {"l": pub_lesson},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = :m"),
            {"m": pub_module},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE id = ANY(:ids)"),
            {"ids": [pub_lesson, draft_lesson]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = :m"), {"m": pub_module}
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {
                "ids": [
                    pub_course,
                    draft_course,
                    archived_course,
                    other_owner_course,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [owner, other_owner]},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"), {"id": org}
        )


async def test_get_course_for_authoring_returns_drafts(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        course = await get_course_for_authoring(session, fixture_data["draft_course"])
    assert course is not None
    assert course.status == "draft"


async def test_get_course_content_authoring_returns_drafts(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    # ``get_course_content_authoring`` became ``get_course_with_content_tree``
    # in the ORM content-tree refactor: it now returns a hydrated Course
    # (modules -> items -> lesson/quiz/interview) instead of a dict tree.
    async with session_factory() as session:
        course = await get_course_with_content_tree(session, fixture_data["pub_course"])
        assert course is not None
        items = [item for module in course.modules for item in module.items]
        item_ids = {str(item.id) for item in items}
        assert str(fixture_data["item_published"]) in item_ids
        assert str(fixture_data["item_draft_target"]) in item_ids
        lesson_statuses = {
            item.lesson.status for item in items if item.lesson is not None
        }
    assert "draft" in lesson_statuses
    assert "published" in lesson_statuses


async def test_authoring_returns_archived_when_flag_set(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        without_flag = await list_courses_for_owner(
            session, fixture_data["owner"], include_archived=False
        )
        with_flag = await list_courses_for_owner(
            session, fixture_data["owner"], include_archived=True
        )
    assert fixture_data["archived_course"] not in {c.id for c in without_flag}
    assert fixture_data["archived_course"] in {c.id for c in with_flag}


async def test_authoring_content_archived_flag_filters_archived_course(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        excluded = await get_course_with_content_tree(
            session, fixture_data["archived_course"], include_archived=False
        )
        included = await get_course_with_content_tree(
            session, fixture_data["archived_course"], include_archived=True
        )
        assert excluded is None
        assert included is not None
        assert included.status == "archived"


async def test_list_courses_for_owner_excludes_other_owners(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        rows = await list_courses_for_owner(session, fixture_data["owner"])
    ids = {c.id for c in rows}
    assert fixture_data["pub_course"] in ids
    assert fixture_data["draft_course"] in ids
    assert fixture_data["other_owner_course"] not in ids


async def test_list_modules_for_authoring_returns_all_statuses(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
    engine: AsyncEngine,
) -> None:
    extra_module = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Draft Module', 2, 'draft')"
            ),
            {"m": extra_module, "c": fixture_data["pub_course"]},
        )
    try:
        async with session_factory() as session:
            modules = await list_modules_for_authoring(
                session, fixture_data["pub_course"]
            )
        ids = {m.id for m in modules}
        assert fixture_data["pub_module"] in ids
        assert extra_module in ids
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM modules WHERE id = :m"), {"m": extra_module}
            )


async def test_list_lessons_for_authoring_returns_all_statuses(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        lessons = await list_lessons_for_authoring(
            session, fixture_data["pub_module"]
        )
    ids = {lesson.id for lesson in lessons}
    assert fixture_data["pub_lesson"] in ids
    assert fixture_data["draft_lesson"] in ids


async def test_list_all_lesson_resources_includes_invisible(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        resources = await list_all_lesson_resources(
            session, fixture_data["pub_lesson"]
        )
    ids = {r.id for r in resources}
    assert fixture_data["resource_visible"] in ids
    assert fixture_data["resource_hidden"] in ids
