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
    get_published_course_by_id,
    get_published_course_by_slug,
    get_published_course_content,
    list_published_courses,
    list_published_lessons,
    list_published_modules,
    list_visible_lesson_resources,
    list_visible_module_items,
)

# Import sibling feature models so SQLAlchemy can resolve the string-name
# relationships on ModuleItem ('Quiz', interview config) when this file runs
# standalone — mapper configuration needs every referenced class registered.
from abridgeai.features.interviews import models as _interviews_models  # noqa: F401
from abridgeai.features.quizzes import models as _quizzes_models  # noqa: F401


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
async def fixture_data(engine: AsyncEngine) -> AsyncIterator[dict]:
    """Seed a minimal courses tree for visibility tests.

    Layout:
        org A: published course "C-pub" + draft course "C-draft"
        org B: published course "C-pub" (slug collision across orgs)
        course C-pub:
            published module M1
                module_item I-vis -> published lesson L-pub
                module_item I-hidden -> draft lesson L-draft
                module_item I-iv-vis -> published interview config IC-pub
                module_item I-iv-hidden -> draft interview config IC-draft
            draft module M-draft (with 1 published lesson) — entire subtree hidden
        lesson L-pub has 2 resources: 1 visible, 1 hidden
    """
    org_a = uuid.uuid4()
    org_b = uuid.uuid4()
    owner = uuid.uuid4()
    pub_course = uuid.uuid4()
    draft_course = uuid.uuid4()
    org_b_course = uuid.uuid4()
    pub_module = uuid.uuid4()
    draft_module = uuid.uuid4()
    pub_lesson = uuid.uuid4()
    draft_lesson = uuid.uuid4()
    draft_module_lesson = uuid.uuid4()
    item_visible = uuid.uuid4()
    item_hidden = uuid.uuid4()
    resource_visible = uuid.uuid4()
    resource_hidden = uuid.uuid4()
    interview_pub = uuid.uuid4()
    interview_draft = uuid.uuid4()
    item_interview_visible = uuid.uuid4()
    item_interview_hidden = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) VALUES "
                "(:id_a, :slug_a, 'Org A', 'active'), "
                "(:id_b, :slug_b, 'Org B', 'active')"
            ),
            {
                "id_a": org_a,
                "id_b": org_b,
                "slug_a": f"orga-{org_a.hex[:8]}",
                "slug_b": f"orgb-{org_b.hex[:8]}",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :email, 'active')"),
            {"id": owner, "email": f"owner-{owner.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) VALUES "
                "(:c1, :oa, :uid, 'shared-slug', 'Pub', 'published'), "
                "(:c2, :oa, :uid, 'only-draft', 'Draft', 'draft'), "
                "(:c3, :ob, :uid, 'shared-slug', 'Pub B', 'published')"
            ),
            {
                "c1": pub_course,
                "c2": draft_course,
                "c3": org_b_course,
                "oa": org_a,
                "ob": org_b,
                "uid": owner,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) VALUES "
                "(:m1, :c, 'Pub Module', 1, 'published'), "
                "(:m2, :c, 'Draft Module', 2, 'draft')"
            ),
            {"m1": pub_module, "m2": draft_module, "c": pub_course},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) VALUES "
                "(:l1, :m1, 'pub-lesson', 'Pub Lesson', 'published'), "
                "(:l2, :m1, 'draft-lesson', 'Draft Lesson', 'draft'), "
                "(:l3, :m2, 'orphan', 'Orphan', 'published')"
            ),
            {
                "l1": pub_lesson,
                "l2": draft_lesson,
                "l3": draft_module_lesson,
                "m1": pub_module,
                "m2": draft_module,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO module_items (id, module_id, item_type, lesson_id, position) VALUES "
                "(:i1, :m, 'lesson', :l1, 1), "
                "(:i2, :m, 'lesson', :l2, 2)"
            ),
            {
                "i1": item_visible,
                "i2": item_hidden,
                "m": pub_module,
                "l1": pub_lesson,
                "l2": draft_lesson,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status) VALUES "
                "(:ic1, :c, :m, 'Pub Interview', 'published'), "
                "(:ic2, :c, :m, 'Draft Interview', 'draft')"
            ),
            {
                "ic1": interview_pub,
                "ic2": interview_draft,
                "c": pub_course,
                "m": pub_module,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO module_items "
                "(id, module_id, item_type, interview_config_id, position) VALUES "
                "(:i3, :m, 'interview', :ic1, 3), "
                "(:i4, :m, 'interview', :ic2, 4)"
            ),
            {
                "i3": item_interview_visible,
                "i4": item_interview_hidden,
                "m": pub_module,
                "ic1": interview_pub,
                "ic2": interview_draft,
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
        "org_a": org_a,
        "org_b": org_b,
        "owner": owner,
        "pub_course": pub_course,
        "draft_course": draft_course,
        "org_b_course": org_b_course,
        "pub_module": pub_module,
        "draft_module": draft_module,
        "pub_lesson": pub_lesson,
        "draft_lesson": draft_lesson,
        "item_visible": item_visible,
        "item_hidden": item_hidden,
        "resource_visible": resource_visible,
        "resource_hidden": resource_hidden,
        "interview_pub": interview_pub,
        "interview_draft": interview_draft,
        "item_interview_visible": item_interview_visible,
        "item_interview_hidden": item_interview_hidden,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_resources WHERE lesson_id = :l"),
            {"l": pub_lesson},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = ANY(:ids)"),
            {"ids": [pub_module, draft_module]},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE id = ANY(:ids)"),
            {"ids": [interview_pub, interview_draft]},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE id = ANY(:ids)"),
            {"ids": [pub_lesson, draft_lesson, draft_module_lesson]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [pub_module, draft_module]},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [pub_course, draft_course, org_b_course]},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner})
        await conn.execute(
            text("DELETE FROM organizations WHERE id = ANY(:ids)"),
            {"ids": [org_a, org_b]},
        )


async def test_list_published_courses_excludes_drafts(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        page = await list_published_courses(session, organization_id=fixture_data["org_a"])
    ids = [c.id for c in page.items]
    assert fixture_data["pub_course"] in ids
    assert fixture_data["draft_course"] not in ids
    assert fixture_data["org_b_course"] not in ids


async def test_get_published_course_content_excludes_draft_lesson_item(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        tree = await get_published_course_content(session, fixture_data["pub_course"])
    assert tree is not None
    item_ids = {item["id"] for item in tree["items"]}
    assert fixture_data["item_visible"] in item_ids
    assert fixture_data["item_hidden"] not in item_ids


async def test_get_published_course_content_filters_interview_items(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """FR-3.6: published interview config visible + hydrated; draft excluded.

    Exercises the real CASE + outer-join path — a missing/draft config
    leaves the join NULL which must exclude the row, not surface it.
    """
    async with session_factory() as session:
        tree = await get_published_course_content(session, fixture_data["pub_course"])
    assert tree is not None
    items_by_id = {str(item["id"]): item for item in tree["items"]}
    assert str(fixture_data["item_interview_visible"]) in items_by_id
    assert str(fixture_data["item_interview_hidden"]) not in items_by_id
    visible = items_by_id[str(fixture_data["item_interview_visible"])]
    assert visible["interview"] is not None
    assert visible["interview"].id == fixture_data["interview_pub"]
    assert visible["interview"].title == "Pub Interview"


async def test_list_visible_module_items_hydrates_interview_target(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        items = await list_visible_module_items(session, fixture_data["pub_module"])
    by_id = {str(item["id"]): item for item in items}
    assert str(fixture_data["item_interview_visible"]) in by_id
    assert str(fixture_data["item_interview_hidden"]) not in by_id
    target = by_id[str(fixture_data["item_interview_visible"])]["target"]
    assert target is not None
    assert target.id == fixture_data["interview_pub"]


async def test_get_published_course_content_excludes_draft_module(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        tree = await get_published_course_content(session, fixture_data["pub_course"])
    assert tree is not None
    module_ids = {m.id for m in tree["modules"]}
    assert fixture_data["pub_module"] in module_ids
    assert fixture_data["draft_module"] not in module_ids


async def test_get_published_course_by_slug_org_scoped(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        course_a = await get_published_course_by_slug(session, "shared-slug", fixture_data["org_a"])
        course_b = await get_published_course_by_slug(session, "shared-slug", fixture_data["org_b"])
        course_other = await get_published_course_by_slug(session, "shared-slug", uuid.uuid4())
    assert course_a is not None
    assert course_a.id == fixture_data["pub_course"]
    assert course_b is not None
    assert course_b.id == fixture_data["org_b_course"]
    assert course_other is None


async def test_list_visible_lesson_resources_filters_invisible(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        resources = await list_visible_lesson_resources(session, fixture_data["pub_lesson"])
    ids = {r.id for r in resources}
    assert fixture_data["resource_visible"] in ids
    assert fixture_data["resource_hidden"] not in ids


async def test_list_published_modules_returns_only_published(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        modules = await list_published_modules(session, fixture_data["pub_course"])
    ids = {m.id for m in modules}
    assert fixture_data["pub_module"] in ids
    assert fixture_data["draft_module"] not in ids


async def test_list_published_lessons_returns_only_published(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        lessons = await list_published_lessons(session, fixture_data["pub_module"])
    ids = {lesson.id for lesson in lessons}
    assert fixture_data["pub_lesson"] in ids
    assert fixture_data["draft_lesson"] not in ids


async def test_get_published_course_by_id_rejects_drafts(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        published = await get_published_course_by_id(session, fixture_data["pub_course"])
        draft = await get_published_course_by_id(session, fixture_data["draft_course"])
    assert published is not None
    assert draft is None


async def test_cursor_pagination_round_trip(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """Seed extra published courses, paginate with limit=2, walk all pages."""
    extra = [uuid.uuid4() for _ in range(3)]
    async with engine.begin() as conn:
        for i, cid in enumerate(extra):
            await conn.execute(
                text(
                    "INSERT INTO courses "
                    "(id, organization_id, owner_user_id, slug, title, status) "
                    "VALUES (:id, :org, :uid, :slug, :title, 'published')"
                ),
                {
                    "id": cid,
                    "org": fixture_data["org_a"],
                    "uid": fixture_data["owner"],
                    "slug": f"extra-{i}-{cid.hex[:6]}",
                    "title": f"Extra {i}",
                },
            )

    try:
        async with session_factory() as session:
            page1 = await list_published_courses(
                session, organization_id=fixture_data["org_a"], limit=2
            )
            assert len(page1.items) == 2
            assert page1.next_cursor is not None
            page2 = await list_published_courses(
                session,
                organization_id=fixture_data["org_a"],
                limit=2,
                cursor=page1.next_cursor,
            )
            seen = {c.id for c in page1.items} | {c.id for c in page2.items}
            assert len(seen) == len(page1.items) + len(page2.items)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = ANY(:ids)"), {"ids": extra})


def test_no_mechanism_split() -> None:
    from pathlib import Path

    queries_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "courses"
        / "queries"
    )
    subdirs = {p.name for p in queries_dir.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert "orm" not in subdirs
    assert "raw" not in subdirs
    assert "sql" in subdirs
