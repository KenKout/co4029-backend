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

import abridgeai.features.courses.models  # noqa: F401  -- register courses + organizations FKs
import abridgeai.features.enrollments.models  # noqa: F401  -- register enrollments tables
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
from abridgeai.features.enrollments.models import Enrollment, InvitationCode


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
            {"id": org_id, "slug": f"enr-cas-{suffix}", "name": "Enrollment Cascade Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"enr-owner-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"enr-stu-{suffix}@test.local"},
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
                "slug": f"enr-cas-course-{suffix}",
                "title": "Enrollment Cascade Course",
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
            text("DELETE FROM course_enrollments WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(
            text("DELETE FROM course_invitation_codes WHERE course_id = :cid"),
            {"cid": course_id},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [owner_id, student_id]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_invitation_code_enrollments_pair_loads(
    session_factory: async_sessionmaker[AsyncSession], seeded_scope
) -> None:
    org_id = seeded_scope["org_id"]
    student_id = seeded_scope["student_id"]
    course_id = seeded_scope["course_id"]

    async with session_factory() as session:
        code = InvitationCode(
            course_id=course_id,
            organization_id=org_id,
            code=f"INVITE-{uuid.uuid4().hex[:8]}",
            max_uses=10,
        )
        session.add(code)
        await session.flush()

        enrollment = Enrollment(
            course_id=course_id,
            student_id=student_id,
            source="invite_code",
            invitation_code_id=code.id,
        )
        session.add(enrollment)
        await session.commit()

        code_id = code.id
        enrollment_id = enrollment.id

    async with session_factory() as session:
        loaded_code = (
            await session.execute(
                select(InvitationCode)
                .where(InvitationCode.id == code_id)
                .options(selectinload(InvitationCode.enrollments))
            )
        ).scalar_one()
        assert len(loaded_code.enrollments) == 1
        assert loaded_code.enrollments[0].id == enrollment_id

        loaded_enrollment = (
            await session.execute(
                select(Enrollment)
                .where(Enrollment.id == enrollment_id)
                .options(selectinload(Enrollment.invitation_code))
            )
        ).scalar_one()
        assert loaded_enrollment.invitation_code is not None
        assert loaded_enrollment.invitation_code.id == code_id


async def test_invitation_code_hard_delete_nulls_enrollment_fk(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_scope,
) -> None:
    org_id = seeded_scope["org_id"]
    student_id = seeded_scope["student_id"]
    course_id = seeded_scope["course_id"]

    async with session_factory() as session:
        code = InvitationCode(
            course_id=course_id,
            organization_id=org_id,
            code=f"INVITE-{uuid.uuid4().hex[:8]}",
        )
        session.add(code)
        await session.flush()

        enrollment = Enrollment(
            course_id=course_id,
            student_id=student_id,
            source="invite_code",
            invitation_code_id=code.id,
        )
        session.add(enrollment)
        await session.commit()

        code_id = code.id
        enrollment_id = enrollment.id

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM course_invitation_codes WHERE id = :id"),
            {"id": code_id},
        )

    async with session_factory() as session:
        surviving_enrollment = (
            await session.execute(select(Enrollment).where(Enrollment.id == enrollment_id))
        ).scalar_one()
        assert surviving_enrollment.id == enrollment_id
        assert surviving_enrollment.invitation_code_id is None
