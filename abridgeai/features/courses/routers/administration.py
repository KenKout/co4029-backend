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
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.pagination import PageResponse
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_permission
from abridgeai.features.courses.schemas import (
    AdminCoursePage,
    CourseAuthoring,
    CourseProcessingAudit,
    CourseStats,
    CourseStatusCount,
    ProcessingJobRow,
    TopOwnerRow,
)
from abridgeai.features.courses.services import administration as administration_service

router = APIRouter(prefix="/admin", tags=["courses-admin"])

_REQUIRE_ADMIN = require_permission("system.administer")


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
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


@router.get("/courses/search", response_model=PageResponse[CourseAuthoring])
async def search_all_courses(
    _admin: Annotated[CurrentUser, Depends(_REQUIRE_ADMIN)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=200)] = None,
    course_status: Annotated[str | None, Query(alias="status")] = None,
    include_deleted: Annotated[bool, Query()] = True,
    sort: Annotated[str | None, Query()] = None,
    sort_dir: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> PageResponse[CourseAuthoring]:
    """Page-numbered admin course list with server-side search (title /
    slug) + whitelisted sort (``title`` / ``status`` / ``created_at``).
    Additive to the cursor ``GET /admin/courses`` above — this one backs
    the DataTable. ``include_deleted`` defaults to ``True`` (admin view)."""
    result = await administration_service.search_all_courses_admin(
        db,
        include_deleted=include_deleted,
        status=course_status,
        search=search,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )
    return PageResponse[CourseAuthoring](
        items=result.items,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


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
    """
    rows = await administration_service.list_course_processing_jobs(db, course_id, limit=limit)
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
    stats = await administration_service.get_course_stats(
        db, top_draft_owners_limit=top_draft_owners_limit
    )
    return CourseStats(
        total_courses=stats["total_courses"],
        soft_deleted_courses=stats["soft_deleted_courses"],
        by_status=[
            CourseStatusCount(status=row["status"], count=row["count"])
            for row in stats["by_status"]
        ],
        top_draft_owners=[
            TopOwnerRow(owner_user_id=row["owner_user_id"], draft_count=row["draft_count"])
            for row in stats["top_draft_owners"]
        ],
    )


__all__ = ["router"]
