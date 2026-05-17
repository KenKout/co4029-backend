"""Learner-side course catalog reads.

Composes :mod:`features.courses.queries.published` accessors and serializes
the resulting ORM rows / dict trees into Pydantic public DTOs. Per plan
§4192 the public surface is intentionally narrow: list / detail / content.

§A11 — slug lookups REQUIRE ``organization_id`` (slug uniqueness is
per-organization). :func:`get_published_course_detail` accepts UUID OR
slug; when a slug is passed the caller must supply ``organization_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.courses.queries import (
    CursorPage,
    get_published_course_by_id,
    get_published_course_by_slug,
    get_published_course_content,
    list_published_courses,
)
from abridgeai.features.courses.schemas import (
    CourseContentPublic,
    CoursePublic,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _looks_like_uuid(value: str | UUID) -> bool:
    if isinstance(value, UUID):
        return True
    try:
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


async def list_published_courses_for_user(
    db: AsyncSession,
    user_id: UUID,
    *,
    organization_id: UUID,
    limit: int = 20,
    cursor: str | None = None,
) -> CursorPage[CoursePublic]:
    """Cursor-paginated learner view of published courses.

    Currently delegates to :func:`list_published_courses`. ``user_id``
    is accepted for API symmetry and future "enrolled OR published"
    composition once the enrollments feature lands in Phase 7 — until
    then it is unused, by design.
    """
    del user_id
    page = await list_published_courses(
        db, organization_id=organization_id, limit=limit, cursor=cursor
    )
    return CursorPage(
        items=[CoursePublic.model_validate(course) for course in page.items],
        next_cursor=page.next_cursor,
    )


async def get_published_course_detail(
    db: AsyncSession,
    course_id_or_slug: str | UUID,
    *,
    organization_id: UUID | None = None,
) -> CoursePublic | None:
    """Single course detail, accepting UUID OR slug.

    Slug lookups are organization-scoped per Reconciliation §A11 — the
    caller MUST supply ``organization_id`` when ``course_id_or_slug`` is
    a slug. UUID lookups ignore the org param.
    """
    if _looks_like_uuid(course_id_or_slug):
        course_id = (
            course_id_or_slug if isinstance(course_id_or_slug, UUID) else UUID(course_id_or_slug)
        )
        course = await get_published_course_by_id(db, course_id)
    else:
        if organization_id is None:
            raise ValueError(
                "organization_id is required when looking up a course by slug "
                "(see Reconciliation §A11)"
            )
        course = await get_published_course_by_slug(db, str(course_id_or_slug), organization_id)
    return None if course is None else CoursePublic.model_validate(course)


async def get_published_course_content_for_learner(
    db: AsyncSession, course_id: UUID
) -> CourseContentPublic | None:
    """Full published content tree (course + modules + items + lessons).

    Mirror of :func:`get_published_course_content` — returns ``None``
    when the course is missing / unpublished / soft-deleted.
    """
    tree = await get_published_course_content(db, course_id)
    if tree is None:
        return None
    return CourseContentPublic.model_validate(tree)


__all__ = [
    "get_published_course_content_for_learner",
    "get_published_course_detail",
    "list_published_courses_for_user",
]
