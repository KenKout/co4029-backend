"""Processing-job queries (status polling + ops dashboards).

Plan §4765-4768. ``ProcessingJob`` is hard-deleted (no soft-delete
filter) — finished/failed jobs are audit history. The T0.7 listener
therefore does NOT auto-add a ``deleted_at IS NULL`` clause here.

Status state machine (Reconciliation §C10): ``pending → running →
{completed, failed, cancelled}``. Plan §4767 names ``queued`` as the
in-progress lead state but the baseline DDL ships ``pending`` (see
``_PROCESSING_JOB_STATUS_CHECK`` in ``features/materials/models.py``).
"In progress" therefore means ``status IN ('pending', 'running')``.

The ``course_id`` filter on :func:`list_jobs_in_progress` resolves
``ProcessingJob → LearningMaterialVersion → LearningMaterial →
lessons → modules.course_id`` via raw SQL on the lessons / modules
tables. Importing those ORM classes here would violate the
``Features are independent`` import-linter contract — raw SQL on the
foreign-feature tables is the documented escape hatch (T3.5 learnings).
``ProcessingJob.entity_id`` is polymorphic (``entity_type`` selects the
parent table); the join is therefore gated on
``entity_type='material_version'``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.models import ProcessingJob

_IN_PROGRESS_STATUSES: tuple[str, ...] = ("pending", "running")


async def get_latest_processing_job(
    db: AsyncSession, material_version_id: UUID
) -> ProcessingJob | None:
    """Most recent :class:`ProcessingJob` for a version, by ``created_at`` DESC.

    ``ProcessingJob.entity_id`` is polymorphic — match on
    ``entity_type='material_version'`` so jobs for lessons, quizzes,
    interview configs, and generation runs do not leak through.
    """
    stmt = (
        select(ProcessingJob)
        .where(
            ProcessingJob.entity_type == "material_version",
            ProcessingJob.entity_id == material_version_id,
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_jobs_in_progress(
    db: AsyncSession,
    *,
    course_id: UUID | None = None,
) -> list[ProcessingJob]:
    """Active jobs (``pending`` or ``running``) for the admin dashboard.

    When ``course_id`` is supplied the query joins to
    ``LearningMaterialVersion → LearningMaterial`` to use the denormalized
    ``course_id`` (Reconciliation §C11). Jobs whose ``entity_type`` is not
    ``material_version`` are excluded by the join — that is the intended
    semantics of "course-scoped processing dashboard".
    """
    if course_id is None:
        stmt = (
            select(ProcessingJob)
            .where(ProcessingJob.status.in_(_IN_PROGRESS_STATUSES))
            .order_by(ProcessingJob.created_at.desc())
        )
        return list((await db.execute(stmt)).scalars().all())

    course_version_ids_sql = text(
        """
        SELECT lmv.id
        FROM learning_material_versions lmv
        JOIN learning_materials lm ON lm.id = lmv.material_id
        JOIN lessons l ON l.id = lm.lesson_id
        JOIN modules m ON m.id = l.module_id
        WHERE m.course_id = :course_id
          AND lmv.deleted_at IS NULL
          AND lm.deleted_at IS NULL
          AND l.deleted_at IS NULL
          AND m.deleted_at IS NULL
        """
    )
    rows = (await db.execute(course_version_ids_sql, {"course_id": course_id})).all()
    version_ids = [row.id for row in rows]
    if not version_ids:
        return []
    stmt = (
        select(ProcessingJob)
        .where(
            ProcessingJob.entity_type == "material_version",
            ProcessingJob.status.in_(_IN_PROGRESS_STATUSES),
            ProcessingJob.entity_id.in_(version_ids),
        )
        .order_by(ProcessingJob.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_failed_jobs_recent(db: AsyncSession, since: datetime) -> list[ProcessingJob]:
    """``status='failed'`` AND ``created_at >= since`` — for ops alerts.

    Caller picks the cutoff (e.g. last 24 hours). Sorted newest-first so
    the most recent failures surface at the top of dashboards.
    """
    stmt = (
        select(ProcessingJob)
        .where(
            ProcessingJob.status == "failed",
            ProcessingJob.created_at >= since,
        )
        .order_by(ProcessingJob.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "get_latest_processing_job",
    "list_failed_jobs_recent",
    "list_jobs_in_progress",
]
