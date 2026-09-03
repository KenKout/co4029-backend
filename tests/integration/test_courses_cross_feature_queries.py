"""Coverage for ``courses/queries/cross_feature.py``.

This module is the read surface sibling features go through instead of
reaching into the courses feature's own query modules. That makes its
FILTERS the contract: every caller inherits them, and a filter that quietly
drifts changes what another feature sees without anything in that feature
changing.

The three that carry real weight, and what each protects:

* ``list_courses_for_teacher`` — a teacher is a ``user_role_assignments`` row
  with ``scope_kind='course'`` and role ``teacher``. It must not count an
  expired assignment, a soft-deleted one, or a non-teacher role on the same
  course, or the manager's user-detail page credits people with courses they
  do not teach.
* ``list_courses_by_org`` — deliberately has NO status filter, because career
  path authoring needs the whole catalogue including drafts. A well-meaning
  ``status == 'published'`` here would empty the course picker.
* ``get_user_primary_org_id`` — excludes ``scope_kind='global'`` so a platform
  admin resolves to no org rather than borrowing whichever tenant sorts first.

Isolation: the suite shares one Postgres, so every test builds its own
organization and tears the whole graph down.
"""

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
from abridgeai.features.courses.queries import cross_feature as q


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
async def db(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


class Graph:
    """org → 2 courses → module → lesson → resource, plus a teacher."""

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex[:10]
        self.org_id = uuid.uuid4()
        self.other_org_id = uuid.uuid4()
        self.teacher_id = uuid.uuid4()
        self.course_a = uuid.uuid4()  # teacher assigned, published
        self.course_b = uuid.uuid4()  # draft, not assigned
        self.foreign_course = uuid.uuid4()
        self.module_pub = uuid.uuid4()
        self.module_draft = uuid.uuid4()
        self.lesson_pub = uuid.uuid4()
        self.lesson_draft = uuid.uuid4()
        self.lesson_in_draft_module = uuid.uuid4()
        self.resource_id = uuid.uuid4()


@pytest_asyncio.fixture
async def graph(engine: AsyncEngine) -> AsyncIterator[Graph]:
    g = Graph()
    async with engine.begin() as conn:
        for org, name in ((g.org_id, "Cross Feature Org"), (g.other_org_id, "Cross Feature Other")):
            await conn.execute(
                text(
                    "INSERT INTO organizations (id, slug, name, status) "
                    "VALUES (:id, :slug, :name, 'active')"
                ),
                {"id": org, "slug": f"xf-{org.hex[:10]}", "name": name},
            )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": g.teacher_id, "email": f"xf-{g.tag}@test.local"},
        )
        for cid, org, title, status in (
            (g.course_a, g.org_id, "Course A", "published"),
            (g.course_b, g.org_id, "Course B", "draft"),
            (g.foreign_course, g.other_org_id, "Foreign", "published"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO courses "
                    "(id, organization_id, owner_user_id, slug, title, status) "
                    "VALUES (:id, :org, :owner, :slug, :title, :status)"
                ),
                {
                    "id": cid,
                    "org": org,
                    "owner": g.teacher_id,
                    "slug": f"xf-{cid.hex[:10]}",
                    "title": title,
                    "status": status,
                },
            )
        for mid, cid, title, pos, status in (
            (g.module_pub, g.course_a, "Published module", 1, "published"),
            (g.module_draft, g.course_a, "Draft module", 2, "draft"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO modules (id, course_id, title, position, status) "
                    "VALUES (:id, :course, :title, :pos, :status)"
                ),
                {"id": mid, "course": cid, "title": title, "pos": pos, "status": status},
            )
        for lid, mid, title, status in (
            (g.lesson_pub, g.module_pub, "Published lesson", "published"),
            (g.lesson_draft, g.module_pub, "Draft lesson", "draft"),
            (g.lesson_in_draft_module, g.module_draft, "Hidden by module", "published"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title, status) "
                    "VALUES (:id, :module, :slug, :title, :status)"
                ),
                {
                    "id": lid,
                    "module": mid,
                    "slug": f"l-{lid.hex[:10]}",
                    "title": title,
                    "status": status,
                },
            )
        await conn.execute(
            text(
                "INSERT INTO lesson_resources "
                "(id, lesson_id, title, resource_type, position) "
                "VALUES (:id, :lesson, 'A handout', 'pdf', 1)"
            ),
            {"id": g.resource_id, "lesson": g.lesson_pub},
        )
    yield g
    async with engine.begin() as conn:
        for stmt in (
            "DELETE FROM module_items WHERE module_id = ANY(:modules)",
            "DELETE FROM lesson_resources WHERE lesson_id = ANY(:lessons)",
            "DELETE FROM course_learning_outcomes WHERE course_id = ANY(:courses)",
            "DELETE FROM user_role_assignments WHERE user_id = :teacher",
            "DELETE FROM lessons WHERE module_id = ANY(:modules)",
            "DELETE FROM modules WHERE course_id = ANY(:courses)",
            "DELETE FROM courses WHERE id = ANY(:courses)",
            "DELETE FROM users WHERE id = :teacher",
            "DELETE FROM organizations WHERE id = ANY(:orgs)",
        ):
            await conn.execute(
                text(stmt),
                {
                    "modules": [g.module_pub, g.module_draft],
                    "lessons": [g.lesson_pub, g.lesson_draft, g.lesson_in_draft_module],
                    "courses": [g.course_a, g.course_b, g.foreign_course],
                    "teacher": g.teacher_id,
                    "orgs": [g.org_id, g.other_org_id],
                },
            )


async def _assign(
    engine: AsyncEngine,
    *,
    user_id: uuid.UUID,
    course_id: uuid.UUID,
    org_id: uuid.UUID,
    role_code: str = "teacher",
    scope_kind: str = "course",
    active_until: str | None = None,
    deleted: bool = False,
) -> None:
    """One user_role_assignments row, addressed the way the queries read it."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id, course_id, "
                " is_instructor, is_assistant, active_until, deleted_at) "
                "SELECT :uid, r.id, :scope, :org, :course, "
                "       :instructor, :assistant, "
                "       CAST(:until AS timestamptz), "
                "       CASE WHEN :deleted THEN NOW() ELSE NULL END "
                "FROM roles r WHERE r.code = :role"
            ),
            {
                "uid": user_id,
                "scope": scope_kind,
                "org": org_id,
                "course": course_id if scope_kind == "course" else None,
                # 0093_teacher_title_flags: a COURSE-scoped assignment must
                # carry one of the title flags or the CHECK rejects it.
                "instructor": scope_kind == "course",
                "assistant": False,
                "until": active_until,
                "deleted": deleted,
                "role": role_code,
            },
        )


# ---------------------------------------------------------------------------
# list_courses_for_teacher
# ---------------------------------------------------------------------------


async def test_lists_only_courses_the_teacher_is_assigned_to(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    await _assign(engine, user_id=graph.teacher_id, course_id=graph.course_a, org_id=graph.org_id)
    rows = await q.list_courses_for_teacher(db, graph.teacher_id)
    assert [c.id for c in rows] == [graph.course_a]


async def test_an_expired_assignment_does_not_count(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """A teacher removed from a course must stop being credited with it.

    Removal sets ``active_until`` rather than deleting the row, so a query
    that only checked ``deleted_at`` would keep listing every course the
    teacher has EVER taught.
    """
    await _assign(
        engine,
        user_id=graph.teacher_id,
        course_id=graph.course_a,
        org_id=graph.org_id,
        active_until="2020-01-01T00:00:00+00:00",
    )
    assert await q.list_courses_for_teacher(db, graph.teacher_id) == []


async def test_a_soft_deleted_assignment_does_not_count(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    await _assign(
        engine,
        user_id=graph.teacher_id,
        course_id=graph.course_a,
        org_id=graph.org_id,
        deleted=True,
    )
    assert await q.list_courses_for_teacher(db, graph.teacher_id) == []


async def test_a_non_teacher_role_on_a_course_does_not_count(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """Role is part of the predicate, not just the scope.

    Without the ``Role.code == 'teacher'`` join a course-scoped assistant or
    observer would appear on the teacher's own list.
    """
    await _assign(
        engine,
        user_id=graph.teacher_id,
        course_id=graph.course_a,
        org_id=graph.org_id,
        role_code="student",
    )
    assert await q.list_courses_for_teacher(db, graph.teacher_id) == []


async def test_an_org_scoped_assignment_is_not_a_course_assignment(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """Being a teacher IN an org is not teaching a specific course."""
    await _assign(
        engine,
        user_id=graph.teacher_id,
        course_id=graph.course_a,
        org_id=graph.org_id,
        scope_kind="organization",
    )
    assert await q.list_courses_for_teacher(db, graph.teacher_id) == []


# ---------------------------------------------------------------------------
# list_courses_by_org
# ---------------------------------------------------------------------------


async def test_org_listing_includes_drafts(db: AsyncSession, graph: Graph) -> None:
    """No status filter, on purpose.

    Career-path authoring lets a draft path carry draft courses — the publish
    gate re-checks every link — so narrowing this to published would empty the
    course picker for exactly the paths still being written.
    """
    ids = {c.id for c in await q.list_courses_by_org(db, graph.org_id)}
    assert graph.course_a in ids
    assert graph.course_b in ids, "a draft course must still be offered"


async def test_org_listing_excludes_other_tenants(db: AsyncSession, graph: Graph) -> None:
    ids = {c.id for c in await q.list_courses_by_org(db, graph.org_id)}
    assert graph.foreign_course not in ids


async def test_org_listing_is_newest_first(db: AsyncSession, graph: Graph) -> None:
    rows = await q.list_courses_by_org(db, graph.org_id)
    created = [c.created_at for c in rows]
    assert created == sorted(created, reverse=True)


# ---------------------------------------------------------------------------
# Single-row resolvers
# ---------------------------------------------------------------------------


async def test_get_course_and_org_and_slug(db: AsyncSession, graph: Graph) -> None:
    course = await q.get_course(db, graph.course_a)
    assert course is not None
    assert course.id == graph.course_a
    assert await q.get_course_org(db, graph.course_a) == graph.org_id
    assert await q.get_course_slug(db, graph.course_a) == course.slug


async def test_single_row_resolvers_return_none_when_absent(db: AsyncSession) -> None:
    """None, not an exception — callers branch on absence."""
    missing = uuid.uuid4()
    assert await q.get_course(db, missing) is None
    assert await q.get_course_org(db, missing) is None
    assert await q.get_course_slug(db, missing) is None
    assert await q.get_lesson(db, missing) is None
    assert await q.get_module(db, missing) is None
    assert await q.get_lesson_title(db, missing) is None
    assert await q.walk_resource_to_course(db, missing) is None


async def test_get_lesson_and_module_and_title(db: AsyncSession, graph: Graph) -> None:
    lesson = await q.get_lesson(db, graph.lesson_pub)
    assert lesson is not None
    assert lesson.title == "Published lesson"
    module = await q.get_module(db, graph.module_pub)
    assert module is not None
    assert module.title == "Published module"
    assert await q.get_lesson_title(db, graph.lesson_pub) == "Published lesson"


async def test_walk_resource_to_course_climbs_the_whole_chain(
    db: AsyncSession, graph: Graph
) -> None:
    """resource → lesson → module → course, in one statement.

    Four tables with no denormalised course id on the resource; if any join
    in the chain breaks, an uploaded file loses its owning course and the
    permission checks built on this walk fail open or closed silently.
    """
    course = await q.walk_resource_to_course(db, graph.resource_id)
    assert course is not None
    assert course.id == graph.course_a


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------


async def test_published_lessons_require_a_published_module(db: AsyncSession, graph: Graph) -> None:
    """BOTH levels must be published.

    A published lesson inside a draft module is not reachable by a learner,
    so returning it would let unfinished material into a learner-facing list.
    """
    ids = {row.id for row in await q.get_published_lessons_for_course(db, graph.course_a)}
    assert graph.lesson_pub in ids
    assert graph.lesson_draft not in ids, "draft lesson leaked"
    assert graph.lesson_in_draft_module not in ids, "draft module's lesson leaked"


async def test_list_lesson_ids_for_modules(db: AsyncSession, graph: Graph) -> None:
    ids = set(await q.list_lesson_ids_for_modules(db, [graph.module_pub]))
    assert ids == {graph.lesson_pub, graph.lesson_draft}


async def test_list_lesson_ids_short_circuits_on_an_empty_list(db: AsyncSession) -> None:
    """No modules selected must not become ``IN ()`` — that is a SQL error.

    The guard is the reason module-scoped generation with nothing selected
    returns an empty filter instead of failing the request.
    """
    assert await q.list_lesson_ids_for_modules(db, []) == []


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


async def test_course_outcomes_are_ordered_and_exclude_deleted(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    async with engine.begin() as conn:
        for pos, txt, deleted in (
            (2, "Second outcome", False),
            (1, "First outcome", False),
            (3, "Removed outcome", True),
        ):
            await conn.execute(
                text(
                    "INSERT INTO course_learning_outcomes "
                    "(id, course_id, position, outcome_text, deleted_at) "
                    "VALUES (gen_random_uuid(), :course, :pos, :txt, "
                    "        CASE WHEN :deleted THEN NOW() ELSE NULL END)"
                ),
                {"course": graph.course_a, "pos": pos, "txt": txt, "deleted": deleted},
            )

    rows = await q.list_course_outcomes(db, graph.course_a)
    assert [r.outcome_text for r in rows] == ["First outcome", "Second outcome"]


# ---------------------------------------------------------------------------
# Module items
# ---------------------------------------------------------------------------


async def test_module_item_positions_start_at_one_and_increment(
    db: AsyncSession, graph: Graph
) -> None:
    """The next position is derived, not assumed.

    ``COALESCE(MAX(position), 0) + 1`` is what makes the first insert land on
    1 rather than on 0 or NULL, and what stops two items sharing a slot.
    """
    assert await q.next_module_item_position(db, graph.module_pub) == 1

    first = await q.insert_module_item(
        db,
        module_id=graph.module_pub,
        item_type="lesson",
        position=await q.next_module_item_position(db, graph.module_pub),
        lesson_id=graph.lesson_pub,
    )
    assert first.position == 1
    await db.commit()

    assert await q.next_module_item_position(db, graph.module_pub) == 2


async def test_find_module_items_by_lesson(db: AsyncSession, graph: Graph) -> None:
    await q.insert_module_item(
        db,
        module_id=graph.module_pub,
        item_type="lesson",
        position=1,
        lesson_id=graph.lesson_pub,
    )
    await db.commit()

    found = await q.find_module_items_by_lesson(db, graph.lesson_pub)
    assert [item.lesson_id for item in found] == [graph.lesson_pub]
    assert await q.find_module_items_by_lesson(db, graph.lesson_draft) == []


# ---------------------------------------------------------------------------
# get_user_primary_org_id
# ---------------------------------------------------------------------------


async def test_primary_org_resolves_from_a_scoped_assignment(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    await _assign(
        engine,
        user_id=graph.teacher_id,
        course_id=graph.course_a,
        org_id=graph.org_id,
        scope_kind="organization",
    )
    assert await q.get_user_primary_org_id(db, graph.teacher_id) == graph.org_id


async def test_a_global_admin_has_no_primary_org(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """``scope_kind='global'`` is excluded deliberately.

    A platform admin belongs to no tenant. Letting a global row answer this
    would silently scope the whole catalogue to whichever organization
    happened to be attached to that assignment.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id) "
                "SELECT :uid, r.id, 'global', NULL FROM roles r WHERE r.code = 'admin'"
            ),
            {"uid": graph.teacher_id},
        )
    assert await q.get_user_primary_org_id(db, graph.teacher_id) is None


async def test_primary_org_ignores_an_expired_assignment(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    await _assign(
        engine,
        user_id=graph.teacher_id,
        course_id=graph.course_a,
        org_id=graph.org_id,
        scope_kind="organization",
        active_until="2020-01-01T00:00:00+00:00",
    )
    assert await q.get_user_primary_org_id(db, graph.teacher_id) is None


async def test_primary_org_is_none_for_an_unassigned_user(db: AsyncSession) -> None:
    assert await q.get_user_primary_org_id(db, uuid.uuid4()) is None
