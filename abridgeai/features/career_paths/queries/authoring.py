from __future__ import annotations

from collections.abc import Sequence
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


async def get_path_impact(
    db: AsyncSession, career_path_id: UUID
) -> tuple[int, list[tuple[UUID, int, str | None, int, int]]]:
    """Blast radius of editing a published path (Gap 3 §2.1).

    Returns ``(active_enrollments, per_stage)`` where each tuple is
    ``(stage_id, position, title, students_in_stage, students_not_completed)``.

    * ``students_in_stage`` — active enrollments CURRENTLY on the stage: no
      latch for it AND no latch for any LATER stage (they have not moved past
      it).
    * ``students_not_completed`` — active enrollments with no latch for the
      stage at all; everyone who must still pass it is affected by an edit.

    ``status='active'`` only: dropped/completed enrollments are not walking
    the path. Stage rows soft-deleted are excluded. Raw SQL keeps the
    correlated NOT EXISTS latch checks readable (the ORM `exists()` form
    inside ``func.count().filter()`` does not correlate reliably).
    """
    from sqlalchemy import text as sa_text  # noqa: PLC0415

    active = int(
        (
            await db.execute(
                sa_text(
                    "SELECT count(*) FROM student_career_enrollments "
                    "WHERE career_path_id = CAST(:pid AS uuid) AND status = 'active'"
                ),
                {"pid": str(career_path_id)},
            )
        ).scalar_one()
    )

    rows = (
        await db.execute(
            sa_text(
                """
                SELECT s.id, s.position, s.title,
                       COUNT(e.id) FILTER (
                         WHERE e.status = 'active'
                           AND NOT EXISTS (
                             SELECT 1 FROM student_stage_progress lat
                             WHERE lat.enrollment_id = e.id AND lat.stage_id = s.id
                           )
                           -- every EARLIER stage is latched: s is the first
                           -- unlatch'd stage, i.e. the student is ON it.
                           AND NOT EXISTS (
                             SELECT 1 FROM career_path_stages s3
                             WHERE s3.career_path_id = s.career_path_id
                               AND s3.deleted_at IS NULL
                               AND s3.position < s.position
                               AND NOT EXISTS (
                                 SELECT 1 FROM student_stage_progress lat4
                                 WHERE lat4.enrollment_id = e.id
                                   AND lat4.stage_id = s3.id
                               )
                           )
                       ) AS students_in_stage,
                       COUNT(e.id) FILTER (
                         WHERE e.status = 'active'
                           AND NOT EXISTS (
                             SELECT 1 FROM student_stage_progress lat
                             WHERE lat.enrollment_id = e.id AND lat.stage_id = s.id
                           )
                       ) AS students_not_completed
                FROM career_path_stages s
                JOIN student_career_enrollments e
                  ON e.career_path_id = s.career_path_id
                WHERE s.career_path_id = CAST(:pid AS uuid)
                  AND s.deleted_at IS NULL
                GROUP BY s.id, s.position, s.title
                ORDER BY s.position
                """
            ),
            {"pid": str(career_path_id)},
        )
    ).all()

    return active, [
        (
            row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            int(row[1]),
            row[2],
            int(row[3]),
            int(row[4]),
        )
        for row in rows
    ]


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


async def list_path_stage_counts(
    db: AsyncSession, career_path_ids: Sequence[UUID]
) -> dict[UUID, int]:
    """Live (non-deleted) stage count per path, for the management list."""
    if not career_path_ids:
        return {}
    rows = await db.execute(
        select(CareerPathStage.career_path_id, func.count(CareerPathStage.id))
        .where(
            CareerPathStage.career_path_id.in_(career_path_ids),
            CareerPathStage.deleted_at.is_(None),
        )
        .group_by(CareerPathStage.career_path_id)
    )
    return {path_id: count for path_id, count in rows.all()}


async def list_path_course_counts(
    db: AsyncSession, career_path_ids: Sequence[UUID]
) -> dict[UUID, int]:
    """Attached-course count per path (career_course_items rows)."""
    if not career_path_ids:
        return {}
    rows = await db.execute(
        select(CareerPathCourse.career_path_id, func.count(CareerPathCourse.course_id))
        .where(CareerPathCourse.career_path_id.in_(career_path_ids))
        .group_by(CareerPathCourse.career_path_id)
    )
    return {path_id: count for path_id, count in rows.all()}


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
