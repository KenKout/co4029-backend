"""IT Admin-side queries for the courses aggregate (T3.5 administration).

These joins span ``ai_model_calls`` + ``processing_jobs`` + ``generation_runs``
to produce a per-course processing audit. The cross-feature reach is
intentional and read-only: services consuming this module are themselves
the IT-Admin-scoped course administration layer.

The shared models (:class:`AIModelCall` / :class:`ProcessingJob` /
:class:`GenerationRun`) live in :mod:`abridgeai.ai.models`, outside the
``features/`` tree, so importing them here does not violate the
``Features are independent`` import-linter contract.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.models import AIModelCall, GenerationRun, ProcessingJob
from abridgeai.core.pagination import Page, paginate
from abridgeai.core.pagination.cursor import (
    CursorPage,
)
from abridgeai.features.courses.models import Course

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def _clamp(limit: int) -> int:
    return min(max(limit, 1), _MAX_LIMIT)


def _encode_admin_cursor(created_at: datetime, course_id: UUID) -> str:
    """Composite cursor for the admin list ``(created_at, id)`` order.

    Admin lists newest first so freshly-inserted rows surface on page 1
    regardless of UUID lexical order. The cursor encodes both keys so
    pagination is stable when many rows share a ``created_at`` value.
    """
    payload = json.dumps([created_at.isoformat(), str(course_id)], separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_admin_cursor(cursor: str) -> tuple[datetime, UUID]:
    padding = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode((cursor + padding).encode()).decode()
    created_iso, course_id = json.loads(raw)
    return datetime.fromisoformat(created_iso), UUID(course_id)


async def list_all_courses_admin(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
) -> CursorPage[Course]:
    """Cursor-paginated admin view of every course (newest first)."""
    capped = _clamp(limit)
    after = _decode_admin_cursor(cursor) if cursor else None
    stmt = select(Course).order_by(Course.created_at.desc(), Course.id.desc()).limit(capped)
    if after is not None:
        after_created_at, after_id = after
        stmt = stmt.where(
            or_(
                Course.created_at < after_created_at,
                and_(Course.created_at == after_created_at, Course.id < after_id),
            )
        )
    if include_deleted:
        stmt = stmt.execution_options(include_deleted=True)
    rows = list((await db.execute(stmt)).scalars().all())
    next_cursor = (
        _encode_admin_cursor(rows[-1].created_at, rows[-1].id) if len(rows) == capped else None
    )
    return CursorPage(items=rows, next_cursor=next_cursor)


async def search_all_courses_admin(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
    search: str | None = None,
    sort: str | None = None,
    sort_dir: str = "asc",
    page: int = 0,
    page_size: int = 25,
) -> Page[Course]:
    """Offset page of courses with server-side search (title / slug) +
    whitelisted sort. Backs the page-numbered admin table.

    Soft-delete: we opt out of the global ``do_orm_execute`` filter via
    ``execution_options(include_deleted=True)`` and re-add the
    ``deleted_at IS NULL`` predicate explicitly when ``include_deleted`` is
    false — baking it into ``stmt`` keeps the count subquery consistent with
    the returned rows (paginate carries the exec options to the count query).
    """
    stmt = select(Course).execution_options(include_deleted=True)
    if not include_deleted:
        stmt = stmt.where(Course.deleted_at.is_(None))
    if status is not None:
        stmt = stmt.where(Course.status == status)
    return await paginate(
        db,
        stmt,
        page=page,
        page_size=page_size,
        search=search,
        search_columns=[Course.title, Course.slug],
        sort=sort,
        sort_dir=sort_dir,
        sortable={
            "title": Course.title,
            "status": Course.status,
            "created_at": Course.created_at,
        },
        default_order=[Course.id],
    )


async def get_course_including_deleted(db: AsyncSession, course_id: UUID) -> Course | None:
    """Course-by-id lookup that ignores the soft-delete loader filter."""
    stmt = select(Course).where(Course.id == course_id).execution_options(include_deleted=True)
    return (await db.execute(stmt)).scalar_one_or_none()


async def restore_soft_deleted_course(db: AsyncSession, course_id: UUID) -> bool:
    """Clear ``deleted_at`` / ``deleted_by`` on the course row."""
    stmt = (
        select(Course)
        .where(Course.id == course_id, Course.deleted_at.is_not(None))
        .execution_options(include_deleted=True)
    )
    course = (await db.execute(stmt)).scalar_one_or_none()
    if course is None:
        return False
    course.deleted_at = None
    course.deleted_by = None
    await db.flush()
    return True


async def get_course_processing_audit(db: AsyncSession, course_id: UUID) -> dict[str, Any]:
    """Aggregate AI processing costs / counts scoped to ``course_id``.

    Joins ``ai_model_calls`` to the two parent FKs that may carry course
    context (``processing_jobs.course_id`` or ``generation_runs.course_id``).
    Returns a single-row aggregate; values are zero / None when no AI work
    has touched the course.
    """
    stmt = (
        select(
            func.coalesce(func.sum(AIModelCall.estimated_cost_usd), 0).label("total_cost_usd"),
            func.coalesce(func.sum(AIModelCall.input_tokens), 0).label("total_input_tokens"),
            func.coalesce(func.sum(AIModelCall.output_tokens), 0).label("total_output_tokens"),
            func.count(AIModelCall.id).label("total_calls"),
            func.count(func.distinct(AIModelCall.pipeline_run_id)).label("pipeline_runs"),
            func.count(func.distinct(ProcessingJob.id)).label("processing_jobs"),
            func.count(func.distinct(GenerationRun.id)).label("generation_runs"),
            func.min(AIModelCall.called_at).label("first_call_at"),
            func.max(AIModelCall.called_at).label("last_call_at"),
        )
        .select_from(AIModelCall)
        .outerjoin(ProcessingJob, ProcessingJob.id == AIModelCall.processing_job_id)
        .outerjoin(GenerationRun, GenerationRun.id == AIModelCall.generation_run_id)
        .where(GenerationRun.course_id == course_id)
    )
    row = (await db.execute(stmt)).one()
    return {
        "course_id": course_id,
        "total_cost_usd": float(row.total_cost_usd or 0),
        "total_input_tokens": int(row.total_input_tokens or 0),
        "total_output_tokens": int(row.total_output_tokens or 0),
        "total_calls": int(row.total_calls or 0),
        "pipeline_runs": int(row.pipeline_runs or 0),
        "processing_jobs": int(row.processing_jobs or 0),
        "generation_runs": int(row.generation_runs or 0),
        "first_call_at": row.first_call_at,
        "last_call_at": row.last_call_at,
    }


async def list_course_processing_jobs(
    db: AsyncSession, course_id: UUID, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Recent ``processing_jobs`` rows tied to ``course_id`` via ``generation_runs``."""
    stmt = (
        select(ProcessingJob)
        .join(AIModelCall, AIModelCall.processing_job_id == ProcessingJob.id)
        .join(GenerationRun, GenerationRun.id == AIModelCall.generation_run_id)
        .where(GenerationRun.course_id == course_id)
        .order_by(ProcessingJob.created_at.desc())
        .distinct()
        .limit(limit)
    )
    jobs = list((await db.execute(stmt)).scalars().all())
    return [
        {
            "id": job.id,
            "entity_type": job.entity_type,
            "entity_id": job.entity_id,
            "job_type": job.job_type,
            "status": job.status,
            "progress_percent": job.progress_percent,
            "error_message": job.error_message,
            "retry_count": job.retry_count,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        for job in jobs
    ]


async def get_course_stats(db: AsyncSession, *, top_draft_owners_limit: int = 10) -> dict[str, Any]:
    """Org-wide course aggregates for the admin dashboard."""
    by_status_stmt = (
        select(Course.status, func.count().label("status_count"))
        .group_by(Course.status)
        .order_by(Course.status)
    )
    by_status_rows = (await db.execute(by_status_stmt)).all()

    totals_stmt = (
        select(
            func.count().filter(Course.deleted_at.is_(None)).label("active_total"),
            func.count().filter(Course.deleted_at.is_not(None)).label("soft_deleted_total"),
        )
        .select_from(Course)
        .execution_options(include_deleted=True)
    )
    totals_row = (await db.execute(totals_stmt)).one()

    top_owners_stmt = (
        select(Course.owner_user_id, func.count().label("draft_count"))
        .where(Course.status == "draft")
        .group_by(Course.owner_user_id)
        .order_by(func.count().desc(), Course.owner_user_id)
        .limit(top_draft_owners_limit)
    )
    top_owner_rows = (await db.execute(top_owners_stmt)).all()

    return {
        "total_courses": int(totals_row.active_total or 0),
        "soft_deleted_courses": int(totals_row.soft_deleted_total or 0),
        "by_status": [
            {"status": row.status, "count": int(row.status_count)} for row in by_status_rows
        ],
        "top_draft_owners": [
            {"owner_user_id": row.owner_user_id, "draft_count": int(row.draft_count)}
            for row in top_owner_rows
        ],
    }


async def stamp_updated_by(db: AsyncSession, course_id: UUID, user_id: UUID) -> None:
    """Set ``updated_by`` on a course after restore (audit trail)."""
    course = await get_course_including_deleted(db, course_id)
    if course is not None:
        course.updated_by = user_id
        await db.flush()


__all__ = [
    "get_course_including_deleted",
    "get_course_processing_audit",
    "get_course_stats",
    "list_all_courses_admin",
    "list_course_processing_jobs",
    "restore_soft_deleted_course",
    "stamp_updated_by",
]
