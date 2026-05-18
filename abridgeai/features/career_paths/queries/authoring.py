from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select

from abridgeai.features.career_paths.models import CareerPath, CareerPathCourse
from abridgeai.features.courses.api import public as courses_api

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


async def list_authoring_career_path_courses(
    db: AsyncSession, career_path_id: UUID
) -> list[dict[str, Any]]:
    link_stmt = (
        select(CareerPathCourse)
        .where(CareerPathCourse.career_path_id == career_path_id)
        .order_by(CareerPathCourse.position)
    )
    links = (await db.execute(link_stmt)).scalars().all()
    rows: list[dict[str, Any]] = []
    for link in links:
        course = await courses_api.get_course_by_id(db, link.course_id)
        if course is None:
            continue
        rows.append(
            {
                "career_path_id": link.career_path_id,
                "course_id": link.course_id,
                "position": link.position,
                "is_required": link.is_required,
                "course_slug": course.slug,
                "course_title": course.title,
                "course_status": course.status,
            }
        )
    return rows


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


async def course_belongs_to_org(db: AsyncSession, course_id: UUID, organization_id: UUID) -> bool:
    course = await courses_api.get_course_by_id(db, course_id)
    return course is not None and course.organization_id == organization_id


__all__ = [
    "course_belongs_to_org",
    "get_career_path_for_authoring",
    "get_path_course_link",
    "list_authoring_career_path_courses",
    "list_career_paths_for_org",
    "list_path_course_links",
    "next_path_course_position",
]
