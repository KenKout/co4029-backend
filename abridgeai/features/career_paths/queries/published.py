from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text, tuple_

from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.career_paths.models import CareerPath

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


async def list_published_career_paths(
    db: AsyncSession,
    *,
    organization_id: UUID,
    limit: int = 20,
    after_created_at: datetime | None = None,
    after_id: UUID | None = None,
) -> list[CareerPath]:
    stmt = (
        select(CareerPath)
        .where(
            CareerPath.organization_id == organization_id,
            CareerPath.status == "published",
            CareerPath.deleted_at.is_(None),
        )
        .order_by(CareerPath.created_at.desc(), CareerPath.id.desc())
        .limit(limit)
    )
    if after_created_at is not None and after_id is not None:
        stmt = stmt.where(
            tuple_(CareerPath.created_at, CareerPath.id) < (after_created_at, after_id)
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
           cci.position, cci.is_required, cci.stage_id
    FROM career_course_items cci
    JOIN career_path_stages s ON s.id = cci.stage_id
        AND s.deleted_at IS NULL
    JOIN courses c ON c.id = cci.course_id
    WHERE cci.career_path_id = :career_path_id
      AND c.status = 'published'
      AND c.deleted_at IS NULL
    ORDER BY s.position, cci.position
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
    """Resolve the requesting user's primary organization for catalog scoping.

    Delegates to :func:`access_control.api.public.get_user_primary_org`,
    which resolves via ``organization_memberships`` exclusively. Role
    assignments are not consulted -- belonging-to-org and
    permissions-in-org are independent concepts. Platform admins
    (``scope_kind='global'``) are NOT implicitly members of any one org.
    Returns ``None`` for users without an active membership.
    """
    org = await access_control_api.get_user_primary_org(db, user_id)
    return org.id if org is not None else None


__all__ = [
    "get_published_career_path_by_slug",
    "get_user_primary_organization_id",
    "list_published_career_path_courses",
    "list_published_career_paths",
]
