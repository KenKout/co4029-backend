from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.progress.api.public import (
    AtRiskStudentDTO,
    LessonProgressDTO,
    get_at_risk_students,
    get_lesson_progress,
)


def test_module_exports() -> None:
    from abridgeai.features.progress.api import public

    assert {"get_lesson_progress", "get_at_risk_students"} <= set(public.__all__)


def test_dto_is_frozen() -> None:
    dto = LessonProgressDTO(
        id=uuid4(),
        user_id=uuid4(),
        lesson_id=uuid4(),
        status="not_started",
        completion_percent=Decimal("0"),
        last_activity_at=None,
        total_time_seconds=0,
    )
    with pytest.raises(ValidationError):
        dto.status = "completed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_get_lesson_progress_returns_none_when_missing(
    test_engine: AsyncEngine,
) -> None:
    async with AsyncSession(test_engine) as session:
        result = await get_lesson_progress(
            session, student_id=uuid4(), lesson_id=uuid4()
        )
    assert result is None


@pytest.mark.asyncio
async def test_get_lesson_progress_roundtrip(test_engine: AsyncEngine) -> None:
    user_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    lesson_id = uuid4()
    org_id = uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
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
                    "INSERT INTO courses (id, organization_id, owner_user_id, "
                    "slug, title, status) "
                    "VALUES (:id, :org, :owner, :slug, 'C', 'draft')"
                ),
                {
                    "id": str(course_id),
                    "org": str(org_id),
                    "owner": str(user_id),
                    "slug": f"c-{suffix}",
                },
            )
            await session.execute(
                text(
                    "INSERT INTO modules (id, course_id, position, title, status) "
                    "VALUES (:id, :course, 1, 'M', 'draft')"
                ),
                {"id": str(module_id), "course": str(course_id)},
            )
            await session.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title, status) "
                    "VALUES (:id, :module, :slug, 'L', 'draft')"
                ),
                {
                    "id": str(lesson_id),
                    "module": str(module_id),
                    "slug": f"l-{suffix}",
                },
            )
            await session.execute(
                text(
                    "INSERT INTO lesson_progress (id, user_id, lesson_id, "
                    "status, completion_percent, total_time_seconds, "
                    "last_activity_at) "
                    "VALUES (:id, :u, :l, 'in_progress', 42.5, 120, :at)"
                ),
                {
                    "id": str(uuid4()),
                    "u": str(user_id),
                    "l": str(lesson_id),
                    "at": datetime.now(UTC),
                },
            )
            await session.flush()

            result = await get_lesson_progress(
                session, student_id=user_id, lesson_id=lesson_id
            )
        finally:
            await trans.rollback()

    assert result is not None
    assert isinstance(result, LessonProgressDTO)
    assert result.user_id == user_id
    assert result.lesson_id == lesson_id
    assert result.status == "in_progress"
    assert result.completion_percent == Decimal("42.50")
    assert result.total_time_seconds == 120


@pytest.mark.asyncio
async def test_get_at_risk_students_empty_course(test_engine: AsyncEngine) -> None:
    async with AsyncSession(test_engine) as session:
        rows = await get_at_risk_students(session, uuid4())
    assert rows == []
    assert all(isinstance(r, AtRiskStudentDTO) for r in rows)
