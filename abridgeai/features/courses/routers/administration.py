"""Courses administration router -- IT Admin operational surface (T3.9).

Mounted at ``/api/v1/admin`` by T3.10. Returns ``CourseAuthoring``
schemas (drafts visible -- admin sees everything, including
soft-deleted rows).

**Permission perimeter (FIX-CRIT-4)**: every endpoint depends on
``require_permission("system.administer")`` -- the catalog does NOT
yet expose a finer-grained ``course.administer`` (T1.3) so the IT-Admin
capability code is the canonical gate. Manager / HOD / Teacher /
Student all 403 here.

**No hard-delete contract (plan §4526)**: the soft-delete invariant is
project-wide. This module deliberately exposes no ``DELETE`` route on
``/admin/courses/{id}``; the recovery path is the dedicated
``POST .../restore`` endpoint that lifts the T0.7 soft-delete loader
filter via :mod:`features.courses.services.administration`.

**Service-commit discipline**: the underlying queries (T3.5) flush
their ``UPDATE`` / ``SELECT`` statements but do NOT commit. This
router commits on the success path of each write so a service-level
exception leaves the transaction unchanged.

**T7 (orm-consolidation)**: the four intra-feature ``text()`` sites
(updated_by touch + three stats aggregations) were migrated to ORM
``select(...) / db.get(...)``. The remaining ``_LIST_PROCESSING_JOBS_SQL``
is a cross-feature read joining ``processing_jobs`` /
``ai_model_calls`` / ``generation_runs`` -- those tables are owned by
other features and Wave 5 will route the query through their public
APIs once they exist. Soft-delete filter is auto-applied to every
``courses`` SELECT via :mod:`abridgeai.core.db.soft_delete` so manual
``deleted_at IS NULL`` clauses are no longer needed.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_permission
from abridgeai.features.courses.models import Course
from abridgeai.features.courses.schemas import CourseAuthoring
from abridgeai.features.courses.services import administration as administration_service

router = APIRouter(prefix="/admin", tags=["courses-admin"])

_REQUIRE_ADMIN = require_permission("system.administer")


class AdminCoursePage(BaseModel):
    """Cursor-paginated admin view of every course (T3.5 ``CursorPage``)."""

    items: list[CourseAuthoring]
    next_cursor: str | None = None


class CourseProcessingAudit(BaseModel):
    """Aggregate AI-processing audit for a single course (T3.5 service shape)."""

    course_id: UUID
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    pipeline_runs: int
    processing_jobs: int
    generation_runs: int
    first_call_at: str | None = None
    last_call_at: str | None = None


class ProcessingJobRow(BaseModel):
    """Row returned by ``GET /admin/courses/{id}/processing``.

    ``processing_jobs`` has no direct ``course_id`` FK in the baseline
    schema -- jobs join to course context via ``ai_model_calls`` (and
    optionally ``generation_runs``). The forward-ref pattern (raw SQL
    here, ORM model when Phase 6 lands) mirrors T3.8's roster
    endpoint.
    """

    id: UUID
    entity_type: str
    entity_id: UUID
    job_type: str
    status: str
    progress_percent: int
    error_message: str | None = None
    retry_count: int
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str
    updated_at: str


class CourseStatusCount(BaseModel):
    status: str
    count: int


class TopOwnerRow(BaseModel):
    owner_user_id: UUID
    draft_count: int


class CourseStats(BaseModel):
    """Org-wide aggregate counts for the admin dashboard."""

    total_courses: int
    soft_deleted_courses: int
    by_status: list[CourseStatusCount]
    top_draft_owners: list[TopOwnerRow]


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


_LIST_PROCESSING_JOBS_SQL = text(
    """
    SELECT DISTINCT
        pj.id              AS id,
        pj.entity_type     AS entity_type,
        pj.entity_id       AS entity_id,
        pj.job_type        AS job_type,
        pj.status          AS status,
        pj.progress_percent AS progress_percent,
        pj.error_message   AS error_message,
        pj.retry_count     AS retry_count,
        pj.started_at      AS started_at,
        pj.finished_at     AS finished_at,
        pj.created_at      AS created_at,
        pj.updated_at      AS updated_at
    FROM processing_jobs pj
    LEFT JOIN ai_model_calls amc ON amc.processing_job_id = pj.id
    LEFT JOIN generation_runs gr ON gr.id = amc.generation_run_id
    WHERE gr.course_id = :course_id
    ORDER BY pj.created_at DESC
    LIMIT :limit
    """
)


def _isoformat(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


@router.get("/courses", response_model=AdminCoursePage)
async def list_all_courses(
    _admin: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    include_deleted: Annotated[bool, Query()] = True,
) -> AdminCoursePage:
    """Cursor-paginated list of every course in the system.

    ``include_deleted=True`` (the default for admin) lifts the T0.7
    soft-delete loader filter so tombstoned rows surface alongside
    live ones; pass ``include_deleted=false`` to scope to live rows
    only.
    """
    page = await administration_service.list_all_courses_admin(
        db, include_deleted=include_deleted, limit=limit, cursor=cursor
    )
    return AdminCoursePage(items=page.items, next_cursor=page.next_cursor)


@router.post("/courses/{course_id}/restore", response_model=CourseAuthoring)
async def restore_soft_deleted_course(
    course_id: UUID,
    admin: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseAuthoring:
    """Lift the soft-delete tombstone on a course.

    Clears ``deleted_at`` AND ``deleted_by`` and stamps ``updated_by``
    with the acting admin (audit trail per plan §4544). Children
    (modules, lessons) are NOT auto-restored -- T0.15's
    ``soft_delete_cascade`` left them tombstoned and admins must
    re-restore each subtree level explicitly via DB ops.

    Returns 404 when no soft-deleted row matches ``course_id``
    (already-active courses also return 404 -- there is nothing to
    restore).
    """
    try:
        restored = await administration_service.restore_soft_deleted_course(db, course_id, admin)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc

    course_orm = await db.get(Course, course_id)
    if course_orm is not None:
        course_orm.updated_by = admin.user_id

    await db.commit()
    return restored


@router.get("/courses/{course_id}/audit", response_model=CourseProcessingAudit)
async def get_course_audit(
    course_id: UUID,
    _admin: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CourseProcessingAudit:
    """Aggregate AI-processing audit for a single course.

    Joins ``ai_model_calls`` against the two parent FKs that may carry
    course context (``processing_jobs.course_id`` is absent in the
    baseline schema, so the join travels via
    ``generation_runs.course_id``). All zero / null when the course
    has not been touched by AI work.
    """
    audit = await administration_service.get_course_processing_audit(db, course_id)
    return CourseProcessingAudit(
        course_id=audit["course_id"],
        total_cost_usd=audit["total_cost_usd"],
        total_input_tokens=audit["total_input_tokens"],
        total_output_tokens=audit["total_output_tokens"],
        total_calls=audit["total_calls"],
        pipeline_runs=audit["pipeline_runs"],
        processing_jobs=audit["processing_jobs"],
        generation_runs=audit["generation_runs"],
        first_call_at=_isoformat(audit.get("first_call_at")),
        last_call_at=_isoformat(audit.get("last_call_at")),
    )


@router.get("/courses/{course_id}/processing", response_model=list[ProcessingJobRow])
async def list_course_processing_jobs(
    course_id: UUID,
    _admin: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ProcessingJobRow]:
    """Recent ``processing_jobs`` rows tied to ``course_id``.

    The baseline schema has no ``processing_jobs.course_id`` FK; the
    join walks ``ai_model_calls.processing_job_id`` and
    ``generation_runs.course_id`` to assemble the per-course view.
    Replace with the canonical ORM helper when Phase 6's
    ``ProcessingJob`` model lands.
    """
    rows = (
        await db.execute(_LIST_PROCESSING_JOBS_SQL, {"course_id": course_id, "limit": limit})
    ).mappings()
    return [
        ProcessingJobRow(
            id=row["id"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            job_type=row["job_type"],
            status=row["status"],
            progress_percent=row["progress_percent"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            started_at=_isoformat(row["started_at"]),
            finished_at=_isoformat(row["finished_at"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]


@router.get("/courses/_stats", response_model=CourseStats)
async def get_course_stats(
    _admin: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
    top_draft_owners_limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> CourseStats:
    """Org-wide course aggregates for the admin dashboard.

    The path uses the underscore-prefix ``/_stats`` so it does not
    collide with the ``/courses/{course_id}`` UUID-shaped path.

    Soft-delete semantics:
    * ``by_status`` and ``top_draft_owners`` only count live rows --
      the do_orm_execute listener auto-applies ``deleted_at IS NULL``
      to every ``Course`` SELECT.
    * ``totals`` distinguishes active vs tombstoned and therefore
      bypasses the loader filter via
      ``execution_options(include_deleted=True)``.
    """
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

    return CourseStats(
        total_courses=int(totals_row.active_total or 0),
        soft_deleted_courses=int(totals_row.soft_deleted_total or 0),
        by_status=[
            CourseStatusCount(status=row.status, count=int(row.status_count))
            for row in by_status_rows
        ],
        top_draft_owners=[
            TopOwnerRow(owner_user_id=row.owner_user_id, draft_count=int(row.draft_count))
            for row in top_owner_rows
        ],
    )


__all__ = ["router"]
