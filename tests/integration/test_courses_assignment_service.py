"""Integration tests for ``features.courses.services.assignment`` (T3.5).

Covers the locked-decision invariant: :func:`assign_teacher_to_course`
creates a :class:`UserRoleAssignment` row with ``role_code='teacher'``,
``scope_kind='course'``, ``course_id`` set, and ``granted_by`` set to
the actor user id (plan §4243).

Also exercises :func:`list_teachers_for_course`,
:func:`remove_teacher_from_course` (soft-revoke via ``active_until``),
and :func:`list_courses_in_faculty` for Faculty Dean overview.
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

import abridgeai.features.access_control.models  # noqa: F401  -- register orgs/roles FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.services import assignment as assignment_service


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
    org_unit_id = uuid.uuid4()
    actor_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    course_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"asg-{suffix}", "name": "Assignment Test Org"},
        )
        await conn.execute(
            text(
                "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'department', :name, :code)"
            ),
            {
                "id": org_unit_id,
                "org": org_id,
                "name": "Test Department",
                "code": f"D-{suffix}",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:a, :ae), (:t, :te)"),
            {
                "a": actor_id,
                "ae": f"actor-{suffix}@test.local",
                "t": teacher_id,
                "te": f"teacher-{suffix}@test.local",
            },
        )
        await conn.execute(
            text("INSERT INTO user_profiles (user_id, display_name) VALUES (:t, :dn)"),
            {"t": teacher_id, "dn": "Teacher Display"},
        )
        # Assignment requires the assignee to be a member of the course's
        # organization, enforced server-side rather than trusted from the
        # client. Real teachers all carry this row; the fixture must too.
        await conn.execute(
            text(
                "INSERT INTO organization_memberships (user_id, organization_id, status) "
                "VALUES (:t, :org, 'active')"
            ),
            {"t": teacher_id, "org": org_id},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, faculty_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :unit, :owner, :slug, 'Assigned Course', 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "unit": org_unit_id,
                "owner": actor_id,
                "slug": f"course-{suffix}",
            },
        )

    yield {
        "actor_id": actor_id,
        "teacher_id": teacher_id,
        "course_id": course_id,
        "org_id": org_id,
        "org_unit_id": org_unit_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE user_id IN (:a, :t) OR course_id = :c"),
            {"a": actor_id, "t": teacher_id, "c": course_id},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE user_id = :t"), {"t": teacher_id}
        )
        await conn.execute(text("DELETE FROM user_profiles WHERE user_id = :t"), {"t": teacher_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [actor_id, teacher_id]},
        )
        await conn.execute(text("DELETE FROM org_units WHERE id = :id"), {"id": org_unit_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def test_assign_teacher_creates_role_assignment(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        actor = _actor(scenario["actor_id"])
        result = await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        rows = (
            (
                await session.execute(
                    text(
                        """
                    SELECT ura.id, ura.scope_kind, ura.course_id, ura.granted_by,
                           r.code AS role_code
                    FROM user_role_assignments ura
                    JOIN roles r ON r.id = ura.role_id
                    WHERE ura.user_id = :u AND ura.course_id = :c
                    """
                    ),
                    {"u": scenario["teacher_id"], "c": scenario["course_id"]},
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["role_code"] == "teacher"
    assert row["scope_kind"] == "course"
    assert row["course_id"] == scenario["course_id"]
    assert row["granted_by"] == scenario["actor_id"]
    assert result["id"] == row["id"]


async def test_assign_teacher_idempotent_active_assignment(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        actor = _actor(scenario["actor_id"])
        first = await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()
        second = await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        rows = (
            (
                await session.execute(
                    text(
                        "SELECT id FROM user_role_assignments WHERE user_id = :u AND course_id = :c"
                    ),
                    {"u": scenario["teacher_id"], "c": scenario["course_id"]},
                )
            )
            .scalars()
            .all()
        )

    assert first["id"] == second["id"]
    assert len(list(rows)) == 1


async def test_list_teachers_for_course_returns_active_only(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        actor = _actor(scenario["actor_id"])
        await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        teachers = await assignment_service.list_teachers_for_course(session, scenario["course_id"])
    assert len(teachers) == 1
    assert teachers[0].user_id == scenario["teacher_id"]
    assert teachers[0].display_name == "Teacher Display"


async def test_remove_teacher_soft_revokes_assignment(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        actor = _actor(scenario["actor_id"])
        assigned = await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        await assignment_service.remove_teacher_from_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        row = (
            (
                await session.execute(
                    text("SELECT active_until FROM user_role_assignments WHERE id = :id"),
                    {"id": assigned["id"]},
                )
            )
            .mappings()
            .one_or_none()
        )
    assert row is not None
    assert row["active_until"] is not None


async def test_list_courses_in_faculty_returns_owned_courses(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session:
        rows = await assignment_service.list_courses_in_faculty(
            session, scenario["org_unit_id"]
        )
    assert any(course.id == scenario["course_id"] for course in rows)


async def test_assign_teacher_to_draft_course_notifies(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """A DRAFT course assignment must notify — this is the normal case.

    Previously this test asserted `count == 0` for a draft, i.e. it encoded
    the bug. The manager flow is: create (draft) -> assign teacher -> teacher
    edits content -> manager publishes. Assignment therefore ALWAYS happens
    while the course is a draft, so gating the notification on
    `status == "published"` meant it never fired in the real flow and the
    teacher was handed work nobody told them about.

    The premise the old guard rested on — "a teacher can't act on a draft they
    can't yet see" — is false: `list_courses_assigned_to_teacher` applies only
    `_archived_filter`, with no status filter, which is exactly what makes the
    "teacher edits content" step possible.
    """
    async with session_factory() as session:
        actor = _actor(scenario["actor_id"])
        await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        row = (
            await session.execute(
                text(
                    "SELECT category, entity_type, entity_id FROM notifications "
                    "WHERE user_id = :u AND category = 'course_announcement'"
                ),
                {"u": scenario["teacher_id"]},
            )
        ).one()
    assert row.category == "course_announcement"
    assert row.entity_type == "course"
    assert row.entity_id == scenario["course_id"]


async def test_assign_teacher_to_archived_course_does_not_notify(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
    engine: AsyncEngine,
) -> None:
    """Archived is the one status with nothing to act on, so it stays silent."""
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET status = 'archived' WHERE id = :id"),
            {"id": scenario["course_id"]},
        )
    async with session_factory() as session:
        actor = _actor(scenario["actor_id"])
        await assignment_service.assign_teacher_to_course(
            session, scenario["course_id"], scenario["teacher_id"], actor
        )
        await session.commit()

        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM notifications "
                    "WHERE user_id = :u AND category = 'course_announcement'"
                ),
                {"u": scenario["teacher_id"]},
            )
        ).scalar_one()
    assert count == 0


async def test_assign_teacher_to_published_course_notifies(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
    engine: AsyncEngine,
) -> None:
    """Assigning a teacher to a PUBLISHED course creates a course_announcement
    notification deep-linking to the teacher course workspace."""
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET status = 'published' WHERE id = :id"),
            {"id": scenario["course_id"]},
        )

    try:
        async with session_factory() as session:
            actor = _actor(scenario["actor_id"])
            await assignment_service.assign_teacher_to_course(
                session, scenario["course_id"], scenario["teacher_id"], actor
            )
            await session.commit()

            row = (
                (
                    await session.execute(
                        text(
                            "SELECT category, entity_type, entity_id, action_url "
                            "FROM notifications WHERE user_id = :u "
                            "AND category = 'course_announcement'"
                        ),
                        {"u": scenario["teacher_id"]},
                    )
                )
                .mappings()
                .all()
            )
        assert len(row) == 1
        assert row[0]["entity_type"] == "course"
        assert str(row[0]["entity_id"]) == str(scenario["course_id"])
        assert row[0]["action_url"] == f"/teacher/courses/{scenario['course_id']}"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM notifications WHERE user_id = :u"),
                {"u": scenario["teacher_id"]},
            )

