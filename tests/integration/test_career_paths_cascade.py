from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import selectinload

import abridgeai.features.career_paths.models  # noqa: F401  -- register career_course_items FK target
import abridgeai.features.courses.models  # noqa: F401  -- register courses + organizations FKs
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.access_control.models import (
    CareerPath,
    StudentCareerEnrollment,
)
from abridgeai.features.career_paths.models import CareerPathCourse


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
async def seeded_scope(engine: AsyncEngine):
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"cp-cas-{suffix}", "name": "Career Path Cascade Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"cp-owner-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"cp-stu-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"cp-cas-course-{suffix}",
                "title": "Career Path Cascade Course",
            },
        )

    yield {
        "org_id": org_id,
        "owner_id": owner_id,
        "student_id": student_id,
        "course_id": course_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM student_career_enrollments WHERE career_path_id IN "
                "(SELECT id FROM career_paths WHERE organization_id = :org)"
            ),
            {"org": org_id},
        )
        await conn.execute(
            text(
                "DELETE FROM career_course_items WHERE career_path_id IN "
                "(SELECT id FROM career_paths WHERE organization_id = :org)"
            ),
            {"org": org_id},
        )
        await conn.execute(
            text("DELETE FROM career_paths WHERE organization_id = :org"),
            {"org": org_id},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [owner_id, student_id]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_career_path_enrollment_pair_loads_bidirectionally(
    session_factory: async_sessionmaker[AsyncSession], seeded_scope
) -> None:
    org_id = seeded_scope["org_id"]
    student_id = seeded_scope["student_id"]
    course_id = seeded_scope["course_id"]

    async with session_factory() as session:
        path = CareerPath(
            organization_id=org_id,
            slug=f"path-{uuid.uuid4().hex[:8]}",
            name="Test Path",
            status="draft",
        )
        session.add(path)
        await session.flush()

        enrollment = StudentCareerEnrollment(
            career_path_id=path.id,
            student_id=student_id,
            status="active",
        )
        session.add(enrollment)

        path_course = CareerPathCourse(
            career_path_id=path.id,
            course_id=course_id,
            position=1,
            is_required=True,
        )
        session.add(path_course)
        await session.commit()

        path_id = path.id
        enrollment_id = enrollment.id

    async with session_factory() as session:
        loaded_path = (
            await session.execute(
                select(CareerPath)
                .where(CareerPath.id == path_id)
                .options(selectinload(CareerPath.enrollments))
            )
        ).scalar_one()
        assert len(loaded_path.enrollments) == 1
        assert loaded_path.enrollments[0].id == enrollment_id

        loaded_enrollment = (
            await session.execute(
                select(StudentCareerEnrollment)
                .where(StudentCareerEnrollment.id == enrollment_id)
                .options(selectinload(StudentCareerEnrollment.career_path))
            )
        ).scalar_one()
        assert loaded_enrollment.career_path is not None
        assert loaded_enrollment.career_path.id == path_id

        loaded_path_course = (
            await session.execute(
                select(CareerPathCourse)
                .where(CareerPathCourse.career_path_id == path_id)
                .options(selectinload(CareerPathCourse.career_path))
            )
        ).scalar_one()
        assert loaded_path_course.career_path is not None
        assert loaded_path_course.career_path.id == path_id


async def test_soft_delete_cascade_walks_enrollments(
    session_factory: async_sessionmaker[AsyncSession], seeded_scope
) -> None:
    org_id = seeded_scope["org_id"]
    owner_id = seeded_scope["owner_id"]
    student_id = seeded_scope["student_id"]

    async with session_factory() as session:
        path = CareerPath(
            organization_id=org_id,
            slug=f"path-{uuid.uuid4().hex[:8]}",
            name="Cascade Test Path",
            status="draft",
        )
        session.add(path)
        await session.flush()

        enrollment = StudentCareerEnrollment(
            career_path_id=path.id,
            student_id=student_id,
            status="active",
        )
        session.add(enrollment)
        await session.commit()

        path_id = path.id
        enrollment_id = enrollment.id

    async with session_factory() as session:
        path = await session.get(CareerPath, path_id)
        assert path is not None
        result = await soft_delete_cascade(session, path, actor_id=owner_id)
        await session.commit()

    affected_tables = {tbl for (tbl, _id) in result.affected}
    assert affected_tables == {"career_paths", "student_career_enrollments"}
    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert path_id in affected_ids
    assert enrollment_id in affected_ids
    assert result.count == 2

    async with session_factory() as session:
        deleted_path = (
            await session.execute(
                select(CareerPath)
                .where(CareerPath.id == path_id)
                .execution_options(include_deleted=True)
            )
        ).scalar_one()
        assert deleted_path.deleted_at is not None
        assert deleted_path.deleted_by == owner_id

        deleted_enrollment = (
            await session.execute(
                select(StudentCareerEnrollment)
                .where(StudentCareerEnrollment.id == enrollment_id)
                .execution_options(include_deleted=True)
            )
        ).scalar_one()
        assert deleted_enrollment.deleted_at is not None
        assert deleted_enrollment.deleted_by == owner_id

        active_path = (
            await session.execute(select(CareerPath).where(CareerPath.id == path_id))
        ).scalar_one_or_none()
        assert active_path is None
