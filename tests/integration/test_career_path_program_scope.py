"""The student career-path catalog is scoped to their learning program.

Bug (2026-08-31): ``/career-paths`` and ``/career-paths/{slug}`` were org-wide
catalog reads with no program awareness, while
``learning_programs.request_path_change`` only accepts a target the student's
pinned program version offers. A student could therefore open a path detail page
whose only action would have 409'd — it rendered with no button and no
explanation, so the page looked broken instead of simply not being on offer.

What is pinned down here:

* a student in a program sees ONLY that program version's paths, in both the
  list and the two detail reads (they must agree — a slim read that still serves
  an off-menu path would just move the dead end);
* a student in NO program keeps the whole org catalog. That is the pre-program
  behaviour and still correct for a directly-enrolled or browsing student, so the
  fix must not quietly gate them out;
* a path the student is genuinely ENROLLED on stays visible even when the
  program version does not list it — a live attempt must not be hidden by a
  curriculum edit;
* the scope is applied in SQL, not after the LIMIT: this list is cursor
  paginated, so post-filtering would return short pages and let ``next_cursor``
  skip past paths the student should see.

These go through the service layer rather than HTTP: the rule is query logic, and
the router adds only a 404 mapping that is already covered elsewhere.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.learning_programs.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.features.career_paths.services import enrollment as enrollment_service

pytestmark = pytest.mark.asyncio


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@dataclass(frozen=True)
class _Fixture:
    """Ids and slugs of the graph one test case needs."""

    org_id: uuid.UUID
    offered_id: uuid.UUID
    off_menu_id: uuid.UUID
    offered_slug: str
    off_menu_slug: str
    in_program: uuid.UUID
    no_program: uuid.UUID
    legacy: uuid.UUID


@pytest_asyncio.fixture
async def graph(engine: AsyncEngine) -> AsyncIterator[_Fixture]:
    """Two published paths, a program offering only ONE, and three students.

    Built from scratch rather than leaning on the seeded catalog so the
    assertions state exact set membership instead of "at least".
    """
    org_id = uuid.uuid4()
    unit_id = uuid.uuid4()
    offered_id, off_menu_id = uuid.uuid4(), uuid.uuid4()
    offered_ver, off_menu_ver = uuid.uuid4(), uuid.uuid4()
    program_id, program_version_id = uuid.uuid4(), uuid.uuid4()
    in_program, no_program, legacy = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    tag = uuid.uuid4().hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, name, slug) VALUES (:i, :n, :s)"
            ),
            {"i": org_id, "n": f"ScopeOrg {tag}", "s": f"scope-org-{tag}"},
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, name, unit_type) "
                "VALUES (:i, :o, :n, 'faculty')"
            ),
            {"i": unit_id, "o": org_id, "n": f"ScopeFaculty {tag}"},
        )
        for uid, email in (
            (in_program, f"scope-in-{tag}@test.local"),
            (no_program, f"scope-none-{tag}@test.local"),
            (legacy, f"scope-legacy-{tag}@test.local"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO users (id, primary_email, status) "
                    "VALUES (:i, :e, 'active')"
                ),
                {"i": uid, "e": email},
            )
            # Membership is what resolves the primary org; without it the
            # catalog returns empty for reasons unrelated to program scope.
            await conn.execute(
                text(
                    "INSERT INTO organization_memberships (id, organization_id, user_id, status) "
                    "VALUES (:i, :o, :u, 'active')"
                ),
                {"i": uuid.uuid4(), "o": org_id, "u": uid},
            )
        for pid, vid, slug in (
            (offered_id, offered_ver, f"scope-offered-{tag}"),
            (off_menu_id, off_menu_ver, f"scope-offmenu-{tag}"),
        ):
            await conn.execute(
                text(
                    # 0094_flat_faculties dropped career_paths.org_unit_id:
                    # "Career paths are organization-wide and intentionally
                    # have no faculty." The learning program below still
                    # carries one — that is what scopes a student's menu.
                    "INSERT INTO career_paths (id, organization_id, slug, name, status)"
                    " VALUES (:i, :o, :s, :n, 'published')"
                ),
                {"i": pid, "o": org_id, "s": slug, "n": slug},
            )
            await conn.execute(
                text(
                    "INSERT INTO career_path_versions (id, career_path_id, version_no, status, "
                    "published_at) VALUES (:i, :p, 1, 'published', now())"
                ),
                {"i": vid, "p": pid},
            )
        await conn.execute(
            text(
                "INSERT INTO learning_programs (id, organization_id, faculty_id, name, slug, "
                "status) VALUES (:i, :o, :ou, :n, :s, 'published')"
            ),
            {"i": program_id, "o": org_id, "ou": unit_id, "n": f"P {tag}", "s": f"p-{tag}"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_program_versions (id, learning_program_id, version_no, "
                "status, max_path_switches, published_at) "
                "VALUES (:i, :p, 1, 'published', 3, now())"
            ),
            {"i": program_version_id, "p": program_id},
        )
        # The program offers ONLY `offered`; `off_menu` is published in the same
        # org but never attached — the exact shape that produced the dead page.
        await conn.execute(
            text(
                "INSERT INTO learning_program_version_paths (program_version_id, career_path_id, "
                "career_path_version_id, position) VALUES (:v, :p, :pv, 1)"
            ),
            {"v": program_version_id, "p": offered_id, "pv": offered_ver},
        )
        for student in (in_program, legacy):
            await conn.execute(
                text(
                    "INSERT INTO program_enrollments (id, learning_program_id, program_version_id, "
                    "student_id, status) VALUES (:i, :p, :v, :s, 'active')"
                ),
                {"i": uuid.uuid4(), "p": program_id, "v": program_version_id, "s": student},
            )
        # `legacy` additionally holds path access to the OFF-MENU path, standing
        # in for a direct/legacy enrolment or a path dropped from the version.
        await conn.execute(
            text(
                "INSERT INTO student_career_enrollments (id, student_id, career_path_id, "
                "version_id, status, started_at) VALUES (:i, :s, :p, :v, 'active', now())"
            ),
            {"i": uuid.uuid4(), "s": legacy, "p": off_menu_id, "v": off_menu_ver},
        )

    yield _Fixture(
        org_id=org_id,
        offered_id=offered_id,
        off_menu_id=off_menu_id,
        offered_slug=f"scope-offered-{tag}",
        off_menu_slug=f"scope-offmenu-{tag}",
        in_program=in_program,
        no_program=no_program,
        legacy=legacy,
    )

    async with engine.begin() as conn:
        for stmt, params in (
            ("DELETE FROM student_career_enrollments WHERE student_id = ANY(:u)", None),
            ("DELETE FROM program_enrollments WHERE student_id = ANY(:u)", None),
            (
                "DELETE FROM learning_program_version_paths WHERE program_version_id = :pv",
                {"pv": program_version_id},
            ),
            (
                "DELETE FROM learning_program_versions WHERE id = :pv",
                {"pv": program_version_id},
            ),
            ("DELETE FROM learning_programs WHERE id = :p", {"p": program_id}),
            (
                "DELETE FROM career_path_versions WHERE career_path_id = ANY(:p)",
                {"p": [offered_id, off_menu_id]},
            ),
            ("DELETE FROM career_paths WHERE id = ANY(:p)", {"p": [offered_id, off_menu_id]}),
            (
                "DELETE FROM organization_memberships WHERE organization_id = :o",
                {"o": org_id},
            ),
            ("DELETE FROM users WHERE id = ANY(:u)", None),
            ("DELETE FROM org_units WHERE id = :ou", {"ou": unit_id}),
            ("DELETE FROM organizations WHERE id = :o", {"o": org_id}),
        ):
            await conn.execute(
                text(stmt), params or {"u": [in_program, no_program, legacy]}
            )


async def _slugs(session_factory: async_sessionmaker[AsyncSession], user_id: uuid.UUID) -> set[str]:
    async with session_factory() as session:
        page = await enrollment_service.list_published_paths_for_user(
            session, user_id=user_id, limit=100, cursor=None
        )
    return {item.slug for item in page.items}


async def test_program_student_sees_only_their_program_paths(
    session_factory: async_sessionmaker[AsyncSession], graph: _Fixture
) -> None:
    """The headline fix: the off-menu path is gone from the catalog list."""
    assert await _slugs(session_factory, graph.in_program) == {graph.offered_slug}


async def test_off_menu_path_detail_reads_404_for_a_program_student(
    session_factory: async_sessionmaker[AsyncSession], graph: _Fixture
) -> None:
    """Both detail reads must refuse it, or the dead page just moves.

    404 rather than an empty-but-200 page: the student was never offered this
    path, and answering "not found" also stops the org catalog being an
    existence oracle for other faculties' paths.
    """
    async with session_factory() as session:
        detail = await enrollment_service.get_published_path_detail_for_user(
            session, slug=graph.off_menu_slug, user_id=graph.in_program
        )
        plain = await enrollment_service.get_published_path_for_user(
            session, slug=graph.off_menu_slug, user_id=graph.in_program
        )
    assert detail is None
    assert plain is None


async def test_offered_path_is_still_readable(
    session_factory: async_sessionmaker[AsyncSession], graph: _Fixture
) -> None:
    """The scope must not over-reach: the program's own path still resolves."""
    async with session_factory() as session:
        detail = await enrollment_service.get_published_path_detail_for_user(
            session, slug=graph.offered_slug, user_id=graph.in_program
        )
        plain = await enrollment_service.get_published_path_for_user(
            session, slug=graph.offered_slug, user_id=graph.in_program
        )
    assert detail is not None
    assert plain is not None
    assert detail.slug == graph.offered_slug


async def test_student_without_a_program_keeps_the_whole_catalog(
    session_factory: async_sessionmaker[AsyncSession], graph: _Fixture
) -> None:
    """No program means no restriction — NOT an empty catalog.

    The regression this guards is treating "no pinned paths" as "nothing
    visible", which would blank the catalog for every student not yet in a
    program (and for directly-enrolled ones).
    """
    assert await _slugs(session_factory, graph.no_program) == {
        graph.offered_slug,
        graph.off_menu_slug,
    }
    async with session_factory() as session:
        assert (
            await enrollment_service.get_published_path_detail_for_user(
                session, slug=graph.off_menu_slug, user_id=graph.no_program
            )
            is not None
        )


async def test_enrolled_path_stays_visible_even_when_off_menu(
    session_factory: async_sessionmaker[AsyncSession], graph: _Fixture
) -> None:
    """A live enrolment outranks the program pin.

    Otherwise a curriculum edit that drops a path would hide the page of a
    student who is actively working through it — losing access to your own
    in-progress path is a worse failure than seeing one extra entry.
    """
    assert await _slugs(session_factory, graph.legacy) == {
        graph.offered_slug,
        graph.off_menu_slug,
    }
    async with session_factory() as session:
        assert (
            await enrollment_service.get_published_path_detail_for_user(
                session, slug=graph.off_menu_slug, user_id=graph.legacy
            )
            is not None
        )


async def test_scope_is_applied_before_the_limit(
    session_factory: async_sessionmaker[AsyncSession], graph: _Fixture
) -> None:
    """A page of size 1 must return the OFFERED path, not an empty page.

    With a post-fetch filter the query would take the newest path in the org
    (the off-menu one, created last), drop it, and hand back an empty page with
    a cursor — hiding a path the student is entitled to see.
    """
    async with session_factory() as session:
        page = await enrollment_service.list_published_paths_for_user(
            session, user_id=graph.in_program, limit=1, cursor=None
        )
    assert [item.slug for item in page.items] == [graph.offered_slug]
