from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CareerPathAuthoring(BaseModel):
    id: UUID
    organization_id: UUID
    org_unit_id: UUID | None = None
    slug: str
    name: str
    description: str | None = None
    status: str
    max_concurrent: int | None = None
    # List-surface enrichment (authoring service): how many stages the path
    # has and how many courses are attached across them, so the management
    # table can show depth without N+1 detail fetches. 0 for a fresh path.
    stage_count: int = 0
    course_count: int = 0
    created_at: datetime
    updated_at: datetime
    created_by: UUID | None = None
    updated_by: UUID | None = None
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class CareerPathCourseAuthoring(BaseModel):
    career_path_id: UUID
    course_id: UUID
    stage_id: UUID
    position: int
    is_required: bool
    satisfied_by: str
    course_slug: str
    course_title: str
    course_status: str

    model_config = ConfigDict(from_attributes=True)


class CareerPathCourseCandidate(BaseModel):
    """A course the manager may attach to a path (picker row).

    The full org catalogue, ANY status: a draft path may hold draft/archived
    courses (the publish gate re-checks every link), so the picker must not
    hide them the way the learner ``/courses`` catalogue does.
    """

    id: UUID
    title: str
    slug: str
    status: str


UnlockPolicy = Literal["always", "after_previous", "after_previous_required"]
StageEnforcement = Literal["hard", "soft", "advisory"]


class CareerPathStageAuthoring(BaseModel):
    id: UUID
    career_path_id: UUID
    position: int
    title: str | None = None
    description: str | None = None
    min_optional_to_complete: int
    unlock_policy: str
    enforcement: str
    course_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class CareerPathStageCreate(BaseModel):
    """Create a stage. Every field is optional but ``position``-less creates
    append to the end.

    Note there is intentionally NO "stage must contain >= 1 course" check
    here: the normal authoring flow is *create empty stage, then add
    courses*, so enforcing it on the mutation path would make adding a
    second stage to a published path impossible. That invariant is a
    **publish gate** instead.
    """

    title: str | None = Field(default=None, max_length=200)
    description: str | None = None
    position: int | None = Field(default=None, gt=0)
    min_optional_to_complete: int = Field(default=0, ge=0)
    unlock_policy: UnlockPolicy = "after_previous"
    enforcement: StageEnforcement = "soft"

    model_config = ConfigDict(extra="forbid")


class CareerPathStageUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    description: str | None = None
    min_optional_to_complete: int | None = Field(default=None, ge=0)
    unlock_policy: UnlockPolicy | None = None
    enforcement: StageEnforcement | None = None

    model_config = ConfigDict(extra="forbid")


class CareerPathStageReorder(BaseModel):
    stage_ids: list[UUID]

    model_config = ConfigDict(extra="forbid")


class StageReorderWarning(BaseModel):
    """A reorder that would silently change effective unlock for students.

    Reorder deliberately does NOT rewrite ``unlock_policy`` — that is
    destructive and silently edits manager intent. It warns instead.
    """

    stage_id: UUID
    code: str
    message: str


class CareerPathStageReorderResult(BaseModel):
    stages: list[CareerPathStageAuthoring]
    warnings: list[StageReorderWarning] = []


class CareerPathCourseMove(BaseModel):
    """Move an item to ``stage_id`` (possibly a different stage) at ``position``."""

    stage_id: UUID
    position: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class CareerPathCreate(BaseModel):
    org_unit_id: UUID | None = None
    slug: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class CareerPathUpdate(BaseModel):
    """Editable metadata for an existing path.

    ``org_unit_id`` is deliberately ABSENT: a path's organization is fixed at
    creation (server-derived from the actor's primary org), the column is not
    consumed by any backend read path, and a stray unit write on a locked-org
    path would silently re-scope metadata nothing reads. The create schema
    keeps the field for completeness; updates never touch it.
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    max_concurrent: int | None = Field(default=None, gt=0)
    """Attention cap; ``None`` in a PATCH means "unchanged", not "clear"."""

    model_config = ConfigDict(extra="forbid")


class CareerPathCourseAdd(BaseModel):
    stage_id: UUID
    course_id: UUID
    position: int | None = Field(default=None, gt=0)
    is_required: bool = True
    satisfied_by: Literal["completion"] = "completion"

    model_config = ConfigDict(extra="forbid")


class CareerPathCoursePatch(BaseModel):
    """Partial update of an existing course-in-stage link."""

    is_required: bool | None = None
    satisfied_by: Literal["completion"] | None = None

    model_config = ConfigDict(extra="forbid")


class CareerPathCourseReorder(BaseModel):
    course_ids: list[UUID]

    model_config = ConfigDict(extra="forbid")


class CareerPathStudentEnroll(BaseModel):
    student_id: UUID

    model_config = ConfigDict(extra="forbid")


class StudentCareerEnrollmentAuthoring(BaseModel):
    id: UUID
    career_path_id: UUID
    student_id: UUID
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    created_by: UUID | None = None
    updated_by: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class StudentPathProgressAuthoring(BaseModel):
    student_id: UUID
    student_email: str
    overall_percent: float
    completed_courses: int
    course_count: int


class StudentReadinessRead(BaseModel):
    """Per-student latest readiness snapshot — manager drill-down row.

    Carries score + identity only; per-interview rubric details are
    intentionally excluded (FR-6.8 / FR-5.7).
    """

    model_config = ConfigDict(from_attributes=True)

    student_id: UUID
    student_email: str
    readiness_score: float
    captured_at: datetime


class PathReadinessOverview(BaseModel):
    """Org-scoped readiness aggregate for one career path (FR-6.8)."""

    career_path_id: UUID
    average_score: float | None = None
    student_count: int
    students: list[StudentReadinessRead]


__all__ = [
    "CareerPathAuthoring",
    "CareerPathCourseAdd",
    "CareerPathCourseAuthoring",
    "CareerPathCourseMove",
    "CareerPathCoursePatch",
    "CareerPathCourseReorder",
    "CareerPathCreate",
    "CareerPathStageAuthoring",
    "CareerPathStageCreate",
    "CareerPathStageReorder",
    "CareerPathStageReorderResult",
    "CareerPathStageUpdate",
    "CareerPathStudentEnroll",
    "CareerPathUpdate",
    "PathReadinessOverview",
    "StageEnforcement",
    "StageReorderWarning",
    "StudentCareerEnrollmentAuthoring",
    "StudentPathProgressAuthoring",
    "StudentReadinessRead",
    "UnlockPolicy",
]
