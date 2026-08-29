"""Processing router -- ``/admin/processing`` (T7.5).

* Reads (queue depth, list, get) require ``ai.processing.read`` or
  ``system.administer``.
* Retry (write) requires ``system.administer`` -- restoring a failed job
  and re-enqueueing it is an IT Admin operation.

The retry endpoint is best-effort against ARQ (see service docstring).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_permission,
)
from abridgeai.features.admin.services import processing as processing_service

router = APIRouter(prefix="/admin/processing", tags=["admin", "processing"])

_REQUIRE_READ = require_any_permission("ai.processing.read", "system.administer")
_REQUIRE_RETRY = require_permission("system.administer")


class QueueDepthOut(BaseModel):
    pending: int
    running: int
    failed: int
    completed: int
    cancelled: int
    total: int


class ProcessingJobOut(BaseModel):
    id: UUID
    entity_type: str
    entity_id: UUID
    job_type: str
    status: str
    progress_percent: int
    error_message: str | None = None
    retry_count: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    #: The HTTP request that enqueued this job, when one did. NULL for work no
    #: request started, and for every row predating migration 0090.
    request_id: UUID | None = None


async def get_arq_pool() -> object | None:
    """ARQ Redis pool dependency (overridable in tests).

    Mirrors the materials feature -- returns ``None`` until the app factory
    wires a real ``ArqRedis`` pool via ``app.dependency_overrides``.
    """
    return None


class JobOwnerOut(BaseModel):
    """Who a job belongs to, resolved at read time from its polymorphic parent.

    Every field is nullable and that is not laziness: a generation run scoped
    to a lesson has no course, a deleted course leaves a job orphaned, and the
    correct answer in both cases is "unknown", not a fabricated owner.
    """

    course_id: UUID | None = None
    course_title: str | None = None
    course_slug: str | None = None
    organization_id: UUID | None = None
    organization_name: str | None = None


class JobTimingOut(BaseModel):
    """Derived timings.

    ``queue_wait_seconds`` is the number that distinguishes a slow job from a
    job that merely waited behind other work -- duration alone cannot tell
    those apart, which is why both are here. Both are ``None`` before the job
    starts: "has not run" is not "ran for zero seconds".
    """

    queue_wait_seconds: int | None = None
    duration_seconds: int | None = None
    is_running: bool = False


class JobStageOut(BaseModel):
    """One pipeline stage's AI usage."""

    stage: str
    call_count: int
    failed_count: int
    spend_usd: float
    tokens: int
    max_latency_ms: int | None = None


class JobAiCallOut(BaseModel):
    """One AI call the job made.

    Prompt and completion payloads are deliberately absent -- they hold course
    and student content, and triage needs the error and the model, not the
    material.
    """

    id: UUID
    stage_name: str | None = None
    role: str | None = None
    model_name: str
    operation: str
    status: str
    error_message: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    request_id: UUID | None = None
    called_at: datetime


class JobInvestigationOut(BaseModel):
    """Everything about one job, in one response (PRD ADM-013/014).

    The point of assembling it server-side is that an operator stops copying
    UUIDs between screens: the owner, the timings, the stage breakdown and the
    failing call all arrive together, and ``correlation_id`` links onward to
    the request that started it and to that request's audit entry.
    """

    job: ProcessingJobOut
    owner: JobOwnerOut
    timing: JobTimingOut
    stages: list[JobStageOut]
    ai_calls: list[JobAiCallOut]
    #: The HTTP request that enqueued this job. ``None`` for work that no
    #: request started -- a worker tick, a reaper sweep, a CLI backfill.
    correlation_id: UUID | None = None
    as_of: datetime


@router.get("/queue", response_model=QueueDepthOut)
async def get_queue_depth(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QueueDepthOut:
    return QueueDepthOut(**await processing_service.queue_depth(db))


@router.get("/summary", response_model=QueueDepthOut)
async def get_processing_summary(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[datetime, Query(description="Lower bound on updated_at (required).")],
    until: Annotated[
        datetime | None,
        Query(description="Optional upper bound on updated_at (custom range 'to' date)."),
    ] = None,
) -> QueueDepthOut:
    """Per-status job counts over the same ``since`` window as ``GET /jobs``.

    The admin processing page's status-tab badges read from here. Deriving
    them client-side from the status-filtered jobs list made every other
    tab's count collapse to zero the moment one status was selected; a
    client-side derivation from an unfiltered list would additionally be
    wrong once a window holds more rows than the list endpoint's cap.
    """
    return QueueDepthOut(
        **(
            await processing_service.status_counts_since(
                db, since=since, until=until
            )
        )
    )


@router.get("/jobs", response_model=list[ProcessingJobOut])
async def list_jobs(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[datetime, Query(description="Lower bound on updated_at (required).")],
    until: Annotated[
        datetime | None,
        Query(description="Optional upper bound on updated_at (custom range 'to' date)."),
    ] = None,
    job_status: Annotated[
        str | None,
        Query(
            alias="status",
            description="Filter by status; one of pending|running|completed|failed|cancelled.",
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ProcessingJobOut]:
    rows = await processing_service.list_jobs(
        db, status=job_status, since=since, until=until, limit=limit
    )
    return [ProcessingJobOut.model_validate(r) for r in rows]


@router.get("/jobs/{job_id}", response_model=ProcessingJobOut)
async def get_job_detail(
    job_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProcessingJobOut:
    try:
        row = await processing_service.get_job(db, job_id=job_id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "resource": "processing_job", "id": str(job_id)},
        ) from exc
    return ProcessingJobOut.model_validate(row)


@router.get(
    "/jobs/{job_id}/investigation", response_model=JobInvestigationOut
)
async def get_job_investigation(
    job_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    ai_call_limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> JobInvestigationOut:
    """One job with its owner, timings, stage breakdown and AI calls.

    Additive to ``GET /jobs/{job_id}``: the plain detail stays the cheap read
    for the jobs list, while this one pays for the joins only when somebody is
    actually investigating.
    """
    try:
        data = await processing_service.job_investigation(
            db, job_id=job_id, ai_call_limit=ai_call_limit
        )
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "resource": "processing_job",
                "id": str(job_id),
            },
        ) from exc

    context = data["context"]
    job = data["job"]
    return JobInvestigationOut(
        job=ProcessingJobOut.model_validate(job),
        owner=JobOwnerOut(
            course_id=context.get("course_id"),
            course_title=context.get("course_title"),
            course_slug=context.get("course_slug"),
            organization_id=context.get("organization_id"),
            organization_name=context.get("organization_name"),
        ),
        timing=JobTimingOut(
            queue_wait_seconds=context.get("queue_wait_seconds"),
            duration_seconds=context.get("duration_seconds"),
            is_running=bool(context.get("is_running", False)),
        ),
        stages=[JobStageOut(**row) for row in data["stages"]],
        ai_calls=[JobAiCallOut(**row) for row in data["ai_calls"]],
        correlation_id=job.get("request_id"),
        as_of=data["as_of"],
    )


@router.post("/jobs/{job_id}/retry", response_model=ProcessingJobOut)
async def retry_job(
    job_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_RETRY)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> ProcessingJobOut:
    try:
        row = await processing_service.retry_failed_job(db, job_id=job_id, arq_pool=arq_pool)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_retriable",
                "message": str(exc),
                "id": str(job_id),
            },
        ) from exc
    return ProcessingJobOut.model_validate(row)


__all__ = ["get_arq_pool", "router"]
