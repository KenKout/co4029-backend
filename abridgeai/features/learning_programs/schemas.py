from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

PathChangeRejectionReasonCode = Literal[
    "insufficient_justification",
    "progress_loss_too_high",
    "target_path_not_suitable",
    "preserve_remaining_switch",
    "advising_required",
    "documentation_missing",
    "other",
]
"""Reason a Faculty Dean rejects a path-change request.

Duplicated as a ``Literal`` (rather than derived from
``models.PATH_CHANGE_REJECTION_REASON_CODES``) so the OpenAPI schema carries the
enum and the frontend's dialog is generated against it; the DB CHECK constraint
in migration 0097 is the third copy and the authority. Adding a code means
touching all three.
"""


class ProgramCreate(BaseModel):
    faculty_id: UUID
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    max_path_switches: int = Field(default=3, ge=0, le=100)
    career_path_ids: list[UUID] = Field(default_factory=list)


class ProgramUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    slug: str | None = Field(
        default=None, min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
    )
    description: str | None = None
    max_path_switches: int | None = Field(default=None, ge=0, le=100)
    career_path_ids: list[UUID] | None = None


class ProgramVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version_no: int
    status: str
    max_path_switches: int
    published_at: datetime | None
    published_by: UUID | None = None
    published_by_name: str | None = None


class ProgramOptionRead(BaseModel):
    id: UUID
    name: str
    slug: str | None = None
    description: str | None = None


class CareerPathOptionRead(ProgramOptionRead):
    """Career-path entry in the authoring-options picker payload.

    ``selectable`` is False for paths that exist but cannot be attached to a
    program version yet (draft/archived, or no published version to pin).
    The backend PATCH/POST gate stays authoritative — this flag only keeps
    the UI picker from offering a choice that would be rejected with
    ``all_paths_must_be_published_and_not_archived``.
    """

    selectable: bool = True
    not_selectable_reason: str | None = None


class ProgramAuthoringOptions(BaseModel):
    faculties: list[ProgramOptionRead] = Field(default_factory=list)
    career_paths: list[CareerPathOptionRead] = Field(default_factory=list)
    default_faculty_id: UUID | None = None


class ProgramPathRead(BaseModel):
    career_path_id: UUID
    career_path_version_id: UUID
    career_path_version_no: int
    name: str
    slug: str
    description: str | None
    status: str
    position: int


class ProgramRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    faculty_id: UUID
    owner_faculty_dean_id: UUID | None
    slug: str
    name: str
    description: str | None
    status: str
    current_version: ProgramVersionRead
    paths: list[ProgramPathRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    # Management-list card statistics (user decision 2026-08-31). Filled by
    # ``list_programs`` in one batched pass; detail endpoints leave the
    # defaults. ``has_draft_version`` is what the eye-catching "draft exists"
    # badge on the card reads — a program with a published version plus an
    # open draft is mid-revise.
    student_count: int = 0
    path_change_request_count: int = 0
    has_draft_version: bool = False


class ProgramEnrollRequest(BaseModel):
    student_ids: list[UUID] = Field(min_length=1, max_length=500)


class ProgramWithdrawRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class PathAttemptRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    career_path_id: UUID
    career_path_version_id: UUID
    previous_attempt_id: UUID | None
    status: str
    selected_at: datetime
    ended_at: datetime | None
    exit_snapshot: dict[str, object] | None


class ProgramCsvImportRow(BaseModel):
    """One roster line. Only ``email`` is required."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    given_name: str | None = Field(default=None, max_length=100)
    family_name: str | None = Field(default=None, max_length=100)
    display_name: str | None = Field(default=None, max_length=200)


class ProgramCsvImportFailure(BaseModel):
    """Why one row did not import, keyed to its position in the file."""

    row_number: int
    identifier: str | None = None
    reason: str


class ProgramCsvImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, str]] = Field(min_length=1, max_length=2000)


class ProgramCsvImportResult(BaseModel):
    """Per-row outcome of a roster import.

    ``enrolled`` and ``created_users`` are disjoint counts of the same run:
    a row can enrol an existing account (enrolled, not created) or a brand
    new one (both). ``failures`` carries the rows that did not import, with
    the reason — a bad row must not abort the batch, because a roster file
    with one typo in it is the normal case, not the exception.
    """

    enrolled: list[UUID] = Field(default_factory=list)
    created_users: list[UUID] = Field(default_factory=list)
    already_enrolled: list[UUID] = Field(default_factory=list)
    failures: list[ProgramCsvImportFailure] = Field(default_factory=list)


class ProgramEnrollmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    learning_program_id: UUID
    program_version_id: UUID
    student_id: UUID
    status: str
    enrolled_at: datetime
    completed_at: datetime | None
    withdrawn_at: datetime | None
    program_name: str
    program_version_no: int
    max_path_switches: int
    approved_switch_count: int = 0
    current_progress_percent: float = 0
    current_completed_courses: int = 0
    current_total_courses: int = 0
    paths: list[ProgramPathRead] = Field(default_factory=list)
    attempts: list[PathAttemptRead] = Field(default_factory=list)
    # The student's OPEN request — ``pending`` *or* ``in_progress``. The name
    # predates the ``in_progress`` status and is kept because it is the field
    # the student UI gates "you already have a request in flight" on, which is
    # true of both statuses. ``status`` inside tells them which.
    pending_change_request: dict[str, object] | None = None
    # Every request this enrolment ever filed, newest first — including
    # rejections with their reason. This is the student-side rejection history;
    # without it a rejected request vanished from their view entirely and the
    # only record was in the dean's queue.
    change_request_history: list[PathChangeRequestRead] = Field(default_factory=list)


class SelectPathRequest(BaseModel):
    career_path_id: UUID


class ChangePathRequestCreate(BaseModel):
    target_career_path_id: UUID
    reason: str = Field(min_length=1, max_length=4000)


class ChangeRequestDecision(BaseModel):
    """Approval payload (and the legacy reject shape).

    Free-text only. Rejections go through :class:`ChangeRequestRejection`,
    which additionally demands a reason CODE — an approval needs no
    justification, a rejection does.
    """

    reason: str | None = Field(default=None, max_length=4000)


class ChangeRequestRejection(BaseModel):
    """Rejection payload: a code from the fixed list, plus optional detail.

    ``reason_code='other'`` REQUIRES ``reason``: the whole point of ``other``
    is that the dean types what actually happened, and a bare "other" in the
    student's notification and in the rejection history would be worse than the
    canned codes it escapes.
    """

    model_config = ConfigDict(extra="forbid")

    reason_code: PathChangeRejectionReasonCode
    reason: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _require_detail_for_other(self) -> ChangeRequestRejection:
        if self.reason_code == "other" and not (self.reason or "").strip():
            raise ValueError("reason_is_required_when_reason_code_is_other")
        return self


class PathChangeRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    program_enrollment_id: UUID
    from_attempt_id: UUID
    target_career_path_id: UUID
    target_career_path_version_id: UUID
    reason: str
    status: str
    in_progress_at: datetime | None = None
    in_progress_by: UUID | None = None
    reviewed_by: UUID | None
    reviewed_at: datetime | None
    decision_reason_code: str | None = None
    decision_reason: str | None
    new_attempt_id: UUID | None
    created_at: datetime


__all__ = [
    "CareerPathOptionRead",
    "ChangePathRequestCreate",
    "ChangeRequestDecision",
    "ChangeRequestRejection",
    "PathChangeRejectionReasonCode",
    "PathAttemptRead",
    "PathChangeRequestRead",
    "ProgramCreate",
    "ProgramAuthoringOptions",
    "ProgramOptionRead",
    "ProgramEnrollmentRead",
    "ProgramEnrollRequest",
    "ProgramPathRead",
    "ProgramRead",
    "ProgramUpdate",
    "ProgramVersionRead",
    "ProgramWithdrawRequest",
    "SelectPathRequest",
]
