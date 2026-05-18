from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.enrollments.api.public import (
    EnrollmentDTO,
    get_course_enrollment,
    is_user_enrolled,
)


def test_module_exports() -> None:
    from abridgeai.features.enrollments.api import public

    assert {"get_course_enrollment", "is_user_enrolled"} <= set(public.__all__)


def test_dto_is_frozen() -> None:
    dto = EnrollmentDTO(
        id=uuid4(),
        course_id=uuid4(),
        student_id=uuid4(),
        status="active",
        source="self_enroll",
        enrolled_at=datetime.now(UTC),
        completed_at=None,
        dropped_at=None,
    )
    with pytest.raises(ValidationError):
        dto.status = "dropped"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_get_course_enrollment_missing_returns_none(
    test_engine: AsyncEngine,
) -> None:
    async with AsyncSession(test_engine) as session:
        result = await get_course_enrollment(
            session, student_id=uuid4(), course_id=uuid4()
        )
    assert result is None


async def _seed_user_course(
    session: AsyncSession,
    *,
    org_id: UUID,
    user_id: UUID,
    course_id: UUID,
) -> None:
    suffix = org_id.hex[:8]
    await session.execute(
        text(
            "INSERT INTO organizations (id, slug, name, status) "
            "VALUES (:id, :slug, :name, 'active')"
        ),
        {"id": str(org_id), "slug": f"o-{suffix}", "name": "O"},
    )
    await session.execute(
        text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
        {"id": str(user_id), "email": f"u-{suffix}@e.com"},
    )
    await session.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, "
            "title, status) "
            "VALUES (:id, :org, :owner, :slug, 'C', 'draft')"
        ),
        {
            "id": str(course_id),
            "org": str(org_id),
            "owner": str(user_id),
            "slug": f"c-{suffix}",
        },
    )


@pytest.mark.asyncio
async def test_get_and_is_enrolled_active(test_engine: AsyncEngine) -> None:
    org_id = uuid4()
    user_id = uuid4()
    course_id = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_user_course(
                session, org_id=org_id, user_id=user_id, course_id=course_id
            )
            await session.execute(
                text(
                    "INSERT INTO course_enrollments (id, course_id, "
                    "student_id, status, source, enrolled_at) "
                    "VALUES (:id, :c, :s, 'active', 'self_enroll', NOW())"
                ),
                {"id": str(uuid4()), "c": str(course_id), "s": str(user_id)},
            )
            await session.flush()

            dto = await get_course_enrollment(
                session, student_id=user_id, course_id=course_id
            )
            enrolled = await is_user_enrolled(
                session, student_id=user_id, course_id=course_id
            )
        finally:
            await trans.rollback()

    assert dto is not None
    assert isinstance(dto, EnrollmentDTO)
    assert dto.status == "active"
    assert dto.course_id == course_id
    assert dto.student_id == user_id
    assert enrolled is True


@pytest.mark.asyncio
async def test_is_user_enrolled_false_for_dropped(test_engine: AsyncEngine) -> None:
    org_id = uuid4()
    user_id = uuid4()
    course_id = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_user_course(
                session, org_id=org_id, user_id=user_id, course_id=course_id
            )
            await session.execute(
                text(
                    "INSERT INTO course_enrollments (id, course_id, "
                    "student_id, status, source, enrolled_at, dropped_at) "
                    "VALUES (:id, :c, :s, 'dropped', 'self_enroll', "
                    "NOW(), NOW())"
                ),
                {"id": str(uuid4()), "c": str(course_id), "s": str(user_id)},
            )
            await session.flush()

            dto = await get_course_enrollment(
                session, student_id=user_id, course_id=course_id
            )
            enrolled = await is_user_enrolled(
                session, student_id=user_id, course_id=course_id
            )
        finally:
            await trans.rollback()

    assert dto is not None
    assert dto.status == "dropped"
    assert enrolled is False
