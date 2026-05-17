from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select, text

from abridgeai.features.career_paths.models import CareerPath, CareerPathCourse

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_career_paths_for_org(
    db: AsyncSession,
    organization_id: UUID,
    *,
    include_archived: bool = False,
) -> list[CareerPath]:
    stmt = select(CareerPath).where(
        CareerPath.organization_id == organization_id,
        CareerPath.deleted_at.is_(None),
    )
    if not include_archived:
        stmt = stmt.where(CareerPath.status != "archived")
    return list((await db.execute(stmt.order_by(CareerPath.created_at.desc()))).scalars().all())


async def get_career_path_for_authoring(
    db: AsyncSession, career_path_id: UUID
) -> CareerPath | None:
    return await db.get(CareerPath, career_path_id)


_AUTHORING_PATH_COURSES_SQL = text(
    """
    SELECT cci.career_path_id, cci.course_id, cci.position, cci.is_required,
           c.slug AS course_slug, c.title AS course_title, c.status AS course_status
    FROM career_course_items cci
    JOIN courses c ON c.id = cci.course_id
    WHERE cci.career_path_id = :career_path_id
      AND c.deleted_at IS NULL
    ORDER BY cci.position
    """
)


async def list_authoring_career_path_courses(
    db: AsyncSession, career_path_id: UUID
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(_AUTHORING_PATH_COURSES_SQL, {"career_path_id": career_path_id})
    ).mappings()
    return [dict(row) for row in rows]


async def get_path_course_link(
    db: AsyncSession, career_path_id: UUID, course_id: UUID
) -> CareerPathCourse | None:
    stmt = select(CareerPathCourse).where(
        CareerPathCourse.career_path_id == career_path_id,
        CareerPathCourse.course_id == course_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_path_course_links(db: AsyncSession, career_path_id: UUID) -> list[CareerPathCourse]:
    stmt = (
        select(CareerPathCourse)
        .where(CareerPathCourse.career_path_id == career_path_id)
        .order_by(CareerPathCourse.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def next_path_course_position(db: AsyncSession, career_path_id: UUID) -> int:
    stmt = select(func.coalesce(func.max(CareerPathCourse.position), 0)).where(
        CareerPathCourse.career_path_id == career_path_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


_COURSE_BELONGS_TO_ORG_SQL = text(
    """
    SELECT 1
    FROM courses
    WHERE id = :course_id
      AND organization_id = :organization_id
      AND deleted_at IS NULL
    """
)


async def course_belongs_to_org(db: AsyncSession, course_id: UUID, organization_id: UUID) -> bool:
    row = (
        await db.execute(
            _COURSE_BELONGS_TO_ORG_SQL,
            {"course_id": course_id, "organization_id": organization_id},
        )
    ).one_or_none()
    return row is not None


__all__ = [
    "course_belongs_to_org",
    "get_career_path_for_authoring",
    "get_path_course_link",
    "list_authoring_career_path_courses",
    "list_career_paths_for_org",
    "list_path_course_links",
    "next_path_course_position",
]
