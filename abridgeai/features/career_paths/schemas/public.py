from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CareerPathCoursePublic(BaseModel):
    course_id: UUID
    slug: str
    title: str
    position: int
    is_required: bool
    stage_id: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class CareerPathPublic(BaseModel):
    id: UUID
    slug: str
    name: str
    description: str | None = None
    status: Literal["published"]
    courses: list[CareerPathCoursePublic] = []

    model_config = ConfigDict(from_attributes=True)


class CareerPathListPage(BaseModel):
    """Cursor-paginated published career paths.

    ``next_cursor`` is opaque and round-trips through subsequent calls.
    Set when the page filled to ``limit`` (more rows may exist); ``None``
    otherwise. Reconciliation §A10/§D2: cursor pagination, not offset.
    """

    items: list[CareerPathPublic]
    next_cursor: str | None = None


class MyCareerEnrollmentRead(BaseModel):
    career_path_id: UUID
    slug: str
    name: str
    status: Literal["active", "completed", "dropped"]
    started_at: datetime
    completed_at: datetime | None = None
    overall_percent: float = 0.0
    is_prepared: bool = False

    model_config = ConfigDict(from_attributes=True)


class CourseProgressSummary(BaseModel):
    course_id: UUID
    slug: str
    title: str
    status: str
    completion_percent: float
    stage_id: UUID | None = None
    is_required: bool = True
    satisfied: bool = False
    """``course_enrollments.status = 'completed'`` — NOT ``completion_percent
    >= 100``. The two can disagree: the writer is synchronous but a course
    the student never enrolled in has no status row at all."""
    is_enrolled: bool = False
    """Whether a lazy course enrollment exists yet (Pattern B)."""


class StageProgressRead(BaseModel):
    """One stage of a path, as the student sees it."""

    stage_id: UUID
    position: int
    title: str | None = None
    description: str | None = None
    min_optional_to_complete: int
    unlock_policy: str
    enforcement: str
    unlocked: bool
    """Stage 1 is ALWAYS unlocked regardless of its stored policy — a path
    whose first stage is locked could never be started."""
    complete: bool
    """True when latched in ``student_stage_progress`` OR the live rule
    holds. Latched wins: a stage that ever completed stays complete."""
    latched: bool
    required_count: int
    satisfied_required: int
    optional_count: int
    satisfied_optional: int
    stage_total: int
    stage_done: int
    courses: list[CourseProgressSummary] = []


class CareerPathProgressRead(BaseModel):
    career_path_id: UUID
    overall_percent: float
    course_count: int
    completed_courses: int
    in_progress_courses: int
    courses: list[CourseProgressSummary]
    stages: list[StageProgressRead] = []
    formula_version: int = 1
    max_concurrent: int | None = None
    active_in_path: int = 0
    over_concurrency_cap: bool = False
    """Advisory ONLY. The cap never blocks — not even under ``hard``
    enforcement, which governs stage lock exclusively."""


class CareerReadinessSnapshotRead(BaseModel):
    """One historical readiness point for the calling student (FR-6.8)."""

    model_config = ConfigDict(from_attributes=True)

    readiness_score: float
    captured_at: datetime
    formula_version: int = 1
    """Which formula produced this point. The chart MUST segment or annotate
    where this changes — an unsegmented line across two formulas misleads
    even though every point on it is honest."""


class StartCourseResult(BaseModel):
    """Result of the student-initiated Start (Pattern B lazy enrollment)."""

    course_id: UUID
    stage_id: UUID
    created: bool
    """False when an enrollment already existed — Start is idempotent."""
    over_concurrency_cap: bool = False
    """Advisory warning; the cap never blocks the Start."""


__all__ = [
    "CareerPathCoursePublic",
    "CareerPathListPage",
    "CareerPathProgressRead",
    "CareerPathPublic",
    "CareerReadinessSnapshotRead",
    "CourseProgressSummary",
    "MyCareerEnrollmentRead",
    "StageProgressRead",
    "StartCourseResult",
]
