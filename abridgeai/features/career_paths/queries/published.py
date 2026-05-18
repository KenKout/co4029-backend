from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text

from abridgeai.features.access_control.api import public as access_control_api
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


async def get_user_primary_organization_id(db: AsyncSession, user_id: UUID) -> UUID | None:
    """Resolve the requesting user's organization for catalog scoping.

    Delegates to :func:`access_control.api.public.get_user_primary_org`
    so the cross-feature read goes through the canonical typed surface
    (T33; deduplicates the parallel helper in
    ``features/courses/queries/published.py`` which T31 migrated).
    ``scope_kind='global'`` is intentionally excluded by the upstream
    surface -- platform admins do not implicitly belong to one org.
    Returns ``None`` for users with no scoped membership.
    """
    org = await access_control_api.get_user_primary_org(db, user_id)
    return org.id if org is not None else None


__all__ = [
    "get_published_career_path_by_slug",
    "get_user_primary_organization_id",
    "list_published_career_path_courses",
    "list_published_career_paths",
]
