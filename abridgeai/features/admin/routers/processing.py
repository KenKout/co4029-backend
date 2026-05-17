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


async def get_arq_pool() -> object | None:
    """ARQ Redis pool dependency (overridable in tests).

    Mirrors the materials feature -- returns ``None`` until the app factory
    wires a real ``ArqRedis`` pool via ``app.dependency_overrides``.
    """
    return None


@router.get("/queue", response_model=QueueDepthOut)
async def get_queue_depth(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QueueDepthOut:
    return QueueDepthOut(**await processing_service.queue_depth(db))


@router.get("/jobs", response_model=list[ProcessingJobOut])
async def list_jobs(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    since: Annotated[datetime, Query(description="Lower bound on updated_at (required).")],
    job_status: Annotated[
        str | None,
        Query(
            alias="status",
            description="Filter by status; one of pending|running|completed|failed|cancelled.",
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ProcessingJobOut]:
    rows = await processing_service.list_jobs(db, status=job_status, since=since, limit=limit)
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
