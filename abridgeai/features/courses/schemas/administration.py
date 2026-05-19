"""IT Admin response DTOs for the courses aggregate.

These schemas back ``routers/administration.py`` (T3.9) and represent
the admin-only projections: processing audit, job rows, and org-wide
stats. They are NOT extensions of the public schemas — admin surfaces
expose cross-feature aggregates that have no student-facing equivalent.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from abridgeai.features.courses.schemas.authoring import CourseAuthoring


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


__all__ = [
    "AdminCoursePage",
    "CourseProcessingAudit",
    "CourseStats",
    "CourseStatusCount",
    "ProcessingJobRow",
    "TopOwnerRow",
]
