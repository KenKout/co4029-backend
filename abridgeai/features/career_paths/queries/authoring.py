from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select

from abridgeai.features.career_paths.models import (
    CareerPath,
    CareerPathCourse,
    CareerPathStage,
    StudentStageProgress,
)
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
        .order_by(CareerPathCourse.stage_id, CareerPathCourse.position)
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
                "stage_id": link.stage_id,
                "position": link.position,
                "is_required": link.is_required,
                "satisfied_by": link.satisfied_by,
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


async def list_stage_course_links(db: AsyncSession, stage_id: UUID) -> list[CareerPathCourse]:
    """Course items of ONE stage, in position order."""
    stmt = (
        select(CareerPathCourse)
        .where(CareerPathCourse.stage_id == stage_id)
        .order_by(CareerPathCourse.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_path_stages(
    db: AsyncSession, career_path_id: UUID, *, include_deleted: bool = False
) -> list[CareerPathStage]:
    """Stages of a path in position order (soft-deleted excluded by default)."""
    stmt = select(CareerPathStage).where(CareerPathStage.career_path_id == career_path_id)
    if not include_deleted:
        stmt = stmt.where(CareerPathStage.deleted_at.is_(None))
    return list((await db.execute(stmt.order_by(CareerPathStage.position))).scalars().all())


async def get_stage(db: AsyncSession, stage_id: UUID) -> CareerPathStage | None:
    """One stage by id; ``None`` when missing or soft-deleted."""
    stage = await db.get(CareerPathStage, stage_id)
    if stage is None or stage.deleted_at is not None:
        return None
    return stage


async def next_stage_position(db: AsyncSession, career_path_id: UUID) -> int:
    stmt = select(func.coalesce(func.max(CareerPathStage.position), 0)).where(
        CareerPathStage.career_path_id == career_path_id,
        CareerPathStage.deleted_at.is_(None),
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def next_stage_course_position(db: AsyncSession, stage_id: UUID) -> int:
    stmt = select(func.coalesce(func.max(CareerPathCourse.position), 0)).where(
        CareerPathCourse.stage_id == stage_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def count_stage_courses(db: AsyncSession, stage_id: UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(CareerPathCourse)
        .where(CareerPathCourse.stage_id == stage_id)
    )
    return int((await db.execute(stmt)).scalar_one())


async def next_path_course_position(db: AsyncSession, career_path_id: UUID) -> int:
    stmt = select(func.coalesce(func.max(CareerPathCourse.position), 0)).where(
        CareerPathCourse.career_path_id == career_path_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def has_latched_stage_progress(db: AsyncSession, stage_id: UUID) -> bool:
    """Any student has ever completed this stage (append-only latch rows exist).

    Guards stage deletion. A manager who moves every course out of a stage
    and then deletes it would otherwise orphan latch rows AND silently shift
    the progress denominator, moving a student's bar without the student
    doing anything.
    """
    stmt = select(StudentStageProgress.id).where(StudentStageProgress.stage_id == stage_id).limit(1)
    return (await db.execute(stmt)).scalar_one_or_none() is not None


async def course_belongs_to_org(db: AsyncSession, course_id: UUID, organization_id: UUID) -> bool:
    course = await courses_api.get_course_by_id(db, course_id)
    return course is not None and course.organization_id == organization_id


async def course_is_published_in_org(
    db: AsyncSession, course_id: UUID, organization_id: UUID
) -> bool:
    """The course exists, belongs to ``organization_id`` AND is published.

    Career paths are published surfaces: attaching a draft/archived course
    would put an invisible or never-visible item into a path students are
    shown. The UI picker already filters to the published catalogue; this is
    the backend guard so a direct API call cannot attach an unpublished
    course either.
    """
    course = await courses_api.get_course_by_id(db, course_id)
    return (
        course is not None
        and course.organization_id == organization_id
        and course.status == "published"
    )


__all__ = [
    "count_stage_courses",
    "course_belongs_to_org",
    "course_is_published_in_org",
    "get_career_path_for_authoring",
    "get_path_course_link",
    "get_stage",
    "has_latched_stage_progress",
    "list_authoring_career_path_courses",
    "list_career_paths_for_org",
    "list_path_course_links",
    "list_path_stages",
    "list_stage_course_links",
    "next_path_course_position",
    "next_stage_course_position",
    "next_stage_position",
]
