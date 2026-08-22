from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select

from abridgeai.features.career_paths.models import (
    CareerPath,
    CareerPathCourse,
    CareerPathStage,
    CareerPathVersion,
    StudentStageProgress,
)
from abridgeai.features.courses.api import public as courses_api

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_versions(
    db: AsyncSession, career_path_id: UUID
) -> list[CareerPathVersion]:
    """All non-deleted versions of a path, newest first (Gap 3)."""
    stmt = (
        select(CareerPathVersion)
        .where(
            CareerPathVersion.career_path_id == career_path_id,
            CareerPathVersion.deleted_at.is_(None),
        )
        .order_by(CareerPathVersion.version_no.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_current_authoring_version(
    db: AsyncSession, career_path_id: UUID
) -> CareerPathVersion | None:
    """The version authoring reads/writes: the latest DRAFT if one exists,
    else the latest version (published). Post-fork there is exactly one
    draft; pre-fork the published version is the editable surface until the
    manager chooses to fork (Gap 3 D2a explicit)."""
    versions = await list_versions(db, career_path_id)
    for version in versions:
        if version.status == "draft":
            return version
    return versions[0] if versions else None


async def get_published_version(
    db: AsyncSession, career_path_id: UUID
) -> CareerPathVersion | None:
    """The latest PUBLISHED version of a path (what new enrollments pin to)."""
    stmt = (
        select(CareerPathVersion)
        .where(
            CareerPathVersion.career_path_id == career_path_id,
            CareerPathVersion.status == "published",
            CareerPathVersion.deleted_at.is_(None),
        )
        .order_by(CareerPathVersion.version_no.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_version(db: AsyncSession, version_id: UUID) -> CareerPathVersion | None:
    version = await db.get(CareerPathVersion, version_id)
    if version is None or version.deleted_at is not None:
        return None
    return version


async def next_version_no(db: AsyncSession, career_path_id: UUID) -> int:
    stmt = select(func.coalesce(func.max(CareerPathVersion.version_no), 0)).where(
        CareerPathVersion.career_path_id == career_path_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


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

    version = await get_current_authoring_version(db, career_path_id)
    if version is None:
        return active, []

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
                             WHERE s3.version_id = s.version_id
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
                  ON e.career_path_id = :pid
                WHERE s.version_id = CAST(:vid AS uuid)
                  AND s.deleted_at IS NULL
                GROUP BY s.id, s.position, s.title
                ORDER BY s.position
                """
            ),
            {"pid": str(career_path_id), "vid": str(version.id)},
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
    """Live (non-deleted) stage count per path's authoring version."""
    if not career_path_ids:
        return {}
    version_ids: dict[UUID, UUID] = {}
    for path_id in career_path_ids:
        version = await get_current_authoring_version(db, path_id)
        if version is not None:
            version_ids[path_id] = version.id
    if not version_ids:
        return {}
    rows = await db.execute(
        select(CareerPathStage.version_id, func.count(CareerPathStage.id))
        .where(
            CareerPathStage.version_id.in_(version_ids.values()),
            CareerPathStage.deleted_at.is_(None),
        )
        .group_by(CareerPathStage.version_id)
    )
    by_version = {version_id: count for version_id, count in rows.all()}
    return {
        path_id: by_version.get(version_id, 0)
        for path_id, version_id in version_ids.items()
    }


async def list_path_course_counts(
    db: AsyncSession, career_path_ids: Sequence[UUID]
) -> dict[UUID, int]:
    """Attached-course count per path (its authoring version's items)."""
    if not career_path_ids:
        return {}
    version_ids: dict[UUID, UUID] = {}
    for path_id in career_path_ids:
        version = await get_current_authoring_version(db, path_id)
        if version is not None:
            version_ids[path_id] = version.id
    if not version_ids:
        return {}
    rows = await db.execute(
        select(CareerPathCourse.version_id, func.count(CareerPathCourse.course_id))
        .where(CareerPathCourse.version_id.in_(version_ids.values()))
        .group_by(CareerPathCourse.version_id)
    )
    by_version = {version_id: count for version_id, count in rows.all()}
    return {
        path_id: by_version.get(version_id, 0)
        for path_id, version_id in version_ids.items()
    }


async def get_career_path_for_authoring(
    db: AsyncSession, career_path_id: UUID
) -> CareerPath | None:
    return await db.get(CareerPath, career_path_id)


async def list_authoring_career_path_courses(
    db: AsyncSession, career_path_id: UUID, *, version_id: UUID | None = None
) -> list[dict[str, Any]]:
    version = (
        await get_version(db, version_id)
        if version_id is not None
        else await get_current_authoring_version(db, career_path_id)
    )
    if version is None:
        return []
    link_stmt = (
        select(CareerPathCourse)
        .where(CareerPathCourse.version_id == version.id)
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
                "career_path_id": version.career_path_id,
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


async def get_version_course_link(
    db: AsyncSession, version_id: UUID, course_id: UUID
) -> CareerPathCourse | None:
    """Course item of ONE VERSION (learner pin resolution, Gap 3)."""
    stmt = select(CareerPathCourse).where(
        CareerPathCourse.version_id == version_id,
        CareerPathCourse.course_id == course_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_path_course_link(
    db: AsyncSession, career_path_id: UUID, course_id: UUID
) -> CareerPathCourse | None:
    version = await get_current_authoring_version(db, career_path_id)
    if version is None:
        return None
    stmt = select(CareerPathCourse).where(
        CareerPathCourse.version_id == version.id,
        CareerPathCourse.course_id == course_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_path_course_links(db: AsyncSession, career_path_id: UUID) -> list[CareerPathCourse]:
    version = await get_current_authoring_version(db, career_path_id)
    if version is None:
        return []
    stmt = (
        select(CareerPathCourse)
        .where(CareerPathCourse.version_id == version.id)
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


async def list_stages_for_version(
    db: AsyncSession, version_id: UUID, *, include_deleted: bool = False
) -> list[CareerPathStage]:
    """Stages of ONE version in position order."""
    stmt = select(CareerPathStage).where(CareerPathStage.version_id == version_id)
    if not include_deleted:
        stmt = stmt.where(CareerPathStage.deleted_at.is_(None))
    return list((await db.execute(stmt.order_by(CareerPathStage.position))).scalars().all())


async def list_items_for_version(
    db: AsyncSession, version_id: UUID
) -> list[CareerPathCourse]:
    """Course items of ONE version (all stages), for copy-on-write forks."""
    stmt = (
        select(CareerPathCourse)
        .where(CareerPathCourse.version_id == version_id)
        .order_by(CareerPathCourse.stage_id, CareerPathCourse.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_path_stages(
    db: AsyncSession,
    career_path_id: UUID,
    *,
    include_deleted: bool = False,
    version_id: UUID | None = None,
) -> list[CareerPathStage]:
    """Stages of a path's authoring version in position order."""
    version = (
        await get_version(db, version_id)
        if version_id is not None
        else await get_current_authoring_version(db, career_path_id)
    )
    if version is None:
        return []
    stmt = select(CareerPathStage).where(CareerPathStage.version_id == version.id)
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
    version = await get_current_authoring_version(db, career_path_id)
    if version is None:
        return 1
    stmt = select(func.coalesce(func.max(CareerPathStage.position), 0)).where(
        CareerPathStage.version_id == version.id,
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
    version = await get_current_authoring_version(db, career_path_id)
    if version is None:
        return 1
    stmt = select(func.coalesce(func.max(CareerPathCourse.position), 0)).where(
        CareerPathCourse.version_id == version.id
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
    "get_current_authoring_version",
    "get_path_course_link",
    "get_published_version",
    "get_stage",
    "get_version",
    "has_latched_stage_progress",
    "list_authoring_career_path_courses",
    "list_career_paths_for_org",
    "list_items_for_version",
    "list_path_course_links",
    "list_path_stages",
    "list_stage_course_links",
    "list_stages_for_version",
    "list_versions",
    "next_path_course_position",
    "next_stage_course_position",
    "next_stage_position",
    "next_version_no",
]
