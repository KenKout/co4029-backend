"""Reassigning a course's owning faculty, and labelling it in list reads.

``faculty_id`` was set once at creation and then frozen, so every course created
before the faculty feature sat at NULL with no route to fix it — and a "filter by
faculty" view could never match them. It is now PATCHable, which makes the
tenancy check the load-bearing part: without it a manager could move a course
into another organization's faculty by sending any UUID.

The list reads also gained ``faculty_name``, because a row carrying only
``faculty_id`` forces the SPA to render a UUID or issue one request per row.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import AppError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.schemas.request import CourseUpdate
from abridgeai.features.courses.services import assignment as assignment_service
from abridgeai.features.courses.services import authoring as authoring_service


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
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """A course in the seeded org, a faculty beside it, and a FOREIGN faculty."""
    suffix = uuid.uuid4().hex[:8]
    course_id = uuid.uuid4()
    home_faculty = uuid.uuid4()
    foreign_org = uuid.uuid4()
    foreign_faculty = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": home_faculty,
                "org": seeded_users.organization_id,
                "name": f"ZZ Home Faculty {suffix}",
                "code": f"zzhome{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {
                "id": foreign_org,
                "slug": f"zz-foreign-{suffix}",
                "name": f"ZZ Foreign Org {suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', :name, :code)"
            ),
            {
                "id": foreign_faculty,
                "org": foreign_org,
                "name": f"ZZ Foreign Faculty {suffix}",
                "code": f"zzforeign{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.teacher_id,
                "slug": f"zz-faculty-course-{suffix}",
                "title": f"ZZ Faculty Course {suffix}",
            },
        )

    yield {
        "course_id": course_id,
        "home_faculty": home_faculty,
        "foreign_faculty": foreign_faculty,
        "foreign_org": foreign_org,
    }

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM org_units WHERE id = ANY(:ids)"),
            {"ids": [home_faculty, foreign_faculty]},
        )
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :o"), {"o": foreign_org}
        )


def _actor(seeded_users: SeededUsers) -> CurrentUser:
    return CurrentUser(user_id=seeded_users.manager_id, session_id=uuid.uuid4())


async def test_faculty_can_be_assigned_after_creation(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The gap this closes: a NULL-faculty course could never be fixed."""
    async with session_factory() as session:
        dto = await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(faculty_id=scenario["home_faculty"]),
            _actor(seeded_users),
        )
        assert dto.faculty_id == scenario["home_faculty"]
        await session.commit()


async def test_a_foreign_faculty_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The reason the check exists: no moving a course out of its tenant.

    A PATCH body is just a UUID; without validation this would succeed and the
    course would silently belong to another organization.
    """
    async with session_factory() as session:
        with pytest.raises(AppError, match="live top-level faculty"):
            await authoring_service.update_course(
                session,
                scenario["course_id"],
                CourseUpdate(faculty_id=scenario["foreign_faculty"]),
                _actor(seeded_users),
            )


async def test_a_nonexistent_faculty_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    async with session_factory() as session:
        with pytest.raises(AppError, match="live top-level faculty"):
            await authoring_service.update_course(
                session,
                scenario["course_id"],
                CourseUpdate(faculty_id=uuid.uuid4()),
                _actor(seeded_users),
            )


async def test_explicit_null_unassigns_rather_than_inferring(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """``faculty_id: null`` means UNASSIGN, not "pick one for me".

    Creation infers a faculty from the actor's affiliations when none is given.
    Reusing that logic here would make clearing the field silently move the
    course to whatever faculty the CALLER belongs to — an edit nobody requested.
    """
    async with session_factory() as session:
        await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(faculty_id=scenario["home_faculty"]),
            _actor(seeded_users),
        )
        await session.commit()

        dto = await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(faculty_id=None),
            _actor(seeded_users),
        )
        assert dto.faculty_id is None
        await session.commit()


async def test_a_patch_that_omits_faculty_leaves_it_alone(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Omitted != null. Editing the description must not clear the faculty."""
    async with session_factory() as session:
        await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(faculty_id=scenario["home_faculty"]),
            _actor(seeded_users),
        )
        await session.commit()

        dto = await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(description="unrelated edit"),
            _actor(seeded_users),
        )
        assert dto.faculty_id == scenario["home_faculty"]
        await session.commit()


async def test_list_reads_carry_the_faculty_name(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The worklist needs a NAME; a UUID is not renderable.

    Also pins that an unassigned course reports ``None`` rather than inheriting a
    neighbour's label — the batched lookup is keyed per course, not per page.
    """
    async with session_factory() as session:
        await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(faculty_id=scenario["home_faculty"]),
            _actor(seeded_users),
        )
        await session.commit()

        rows = await assignment_service.list_courses_for_organization(
            session, seeded_users.organization_id
        )
        mine = next(r for r in rows if r.id == scenario["course_id"])
        assert mine.faculty_name is not None
        assert mine.faculty_name.startswith("ZZ Home Faculty")

        unassigned = [r for r in rows if r.faculty_id is None]
        assert all(r.faculty_name is None for r in unassigned)


async def test_a_retired_faculty_reads_as_unassigned(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """A course pointing at a soft-deleted faculty must not name it.

    Naming a retired faculty would present it as current; the id stays on the row
    for history, but the label reads as unassigned.
    """
    async with session_factory() as session:
        await authoring_service.update_course(
            session,
            scenario["course_id"],
            CourseUpdate(faculty_id=scenario["home_faculty"]),
            _actor(seeded_users),
        )
        await session.commit()

    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE org_units SET deleted_at = NOW() WHERE id = :id"),
            {"id": scenario["home_faculty"]},
        )

    async with session_factory() as session:
        rows = await assignment_service.list_courses_for_organization(
            session, seeded_users.organization_id
        )
        mine = next(r for r in rows if r.id == scenario["course_id"])
        assert mine.faculty_id == scenario["home_faculty"]
        assert mine.faculty_name is None
