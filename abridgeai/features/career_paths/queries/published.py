from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text

from abridgeai.features.career_paths.models import CareerPath

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_published_career_paths(
    db: AsyncSession,
    *,
    organization_id: UUID,
    limit: int = 20,
    offset: int = 0,
) -> list[CareerPath]:
    stmt = (
        select(CareerPath)
        .where(
            CareerPath.organization_id == organization_id,
            CareerPath.status == "published",
            CareerPath.deleted_at.is_(None),
        )
        .order_by(CareerPath.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_published_career_path_by_slug(
    db: AsyncSession, *, slug: str, organization_id: UUID
) -> CareerPath | None:
    stmt = select(CareerPath).where(
        CareerPath.slug == slug,
        CareerPath.organization_id == organization_id,
        CareerPath.status == "published",
        CareerPath.deleted_at.is_(None),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


_PUBLISHED_PATH_COURSES_SQL = text(
    """
    SELECT cci.course_id, c.slug AS course_slug, c.title AS course_title,
           cci.position, cci.is_required
    FROM career_course_items cci
    JOIN courses c ON c.id = cci.course_id
    WHERE cci.career_path_id = :career_path_id
      AND c.status = 'published'
      AND c.deleted_at IS NULL
    ORDER BY cci.position
    """
)


async def list_published_career_path_courses(
    db: AsyncSession, career_path_id: UUID
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(_PUBLISHED_PATH_COURSES_SQL, {"career_path_id": career_path_id})
    ).mappings()
    return [dict(row) for row in rows]


_USER_PRIMARY_ORG_SQL = text(
    """
    SELECT organization_id
    FROM user_role_assignments
    WHERE user_id = :user_id
      AND scope_kind IN ('organization', 'org_unit', 'course')
      AND organization_id IS NOT NULL
      AND (active_until IS NULL OR active_until > NOW())
    ORDER BY created_at DESC NULLS LAST
    LIMIT 1
    """
)


async def get_user_primary_organization_id(db: AsyncSession, user_id: UUID) -> UUID | None:
    result = await db.execute(_USER_PRIMARY_ORG_SQL, {"user_id": user_id})
    return result.scalar_one_or_none()


__all__ = [
    "get_published_career_path_by_slug",
    "get_user_primary_organization_id",
    "list_published_career_path_courses",
    "list_published_career_paths",
]
