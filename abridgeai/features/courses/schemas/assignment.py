"""HOD/Manager teacher-assignment DTOs for the courses aggregate.

These schemas back ``routers/assignment.py`` (T3.8) and represent the
staffing-surface projections: teacher assignments, roster entries, and
the assign-teacher request body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TeacherAssignmentRead(BaseModel):
    """Authoring DTO for a teacher-on-course assignment.

    ``primary_email`` / ``display_name`` are joined from ``users`` +
    ``user_profiles``; ``active_until`` is non-null for soft-revoked
    rows (audit trail). ``course_role`` is the course-scoped title
    (``course_instructor`` | ``teacher_assistant`` — "no catalog logic for
    titles", user decision 2026-08-18).
    """

    user_id: UUID
    display_name: str
    primary_email: str
    assignment_id: UUID | None = None
    course_role: str | None = None
    active_from: datetime | None = None
    active_until: datetime | None = None
    # Short-TTL presigned GET URL for the teacher's uploaded avatar, minted by
    # the service layer. ``None`` when no avatar is set (SPA falls back to
    # initials). Not persisted — a projection.
    avatar_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class CoursePathPlacement(BaseModel):
    """Where a course sits on one career path."""

    career_path_id: UUID
    career_path_name: str
    career_path_status: str
    stage_id: UUID
    stage_title: str | None = None
    stage_position: int
    is_required: bool

    model_config = ConfigDict(from_attributes=True)


class CourseReadiness(BaseModel):
    """Whether a course is actually deliverable, before publish rather than after.

    The manager's checklist. `can_publish` mirrors the publish gate's condition
    exactly (at least one gradeable unit, at least one learning outcome, not
    archived), so the checklist and the 409 cannot disagree.
    """

    course_id: UUID
    status: str
    teacher_count: int
    course_instructor_count: int = 0
    min_teachers_per_course: int = 0
    max_teachers_per_course: int = 0
    staffing_ok: bool = True
    gradeable_unit_count: int
    learning_outcome_count: int = 0
    career_paths: list[CoursePathPlacement] = []
    blocks_required_stage: bool = False
    """True when this course has no gradeable unit AND is required on a path —
    it is locking that stage and every stage behind it for every student."""
    can_publish: bool

    model_config = ConfigDict(from_attributes=True)


class AssignableTeacher(BaseModel):
    """A teacher the manager may pick for this course.

    The list is org-scoped server-side from the course, so every entry here is
    already a legal choice — the client does not filter, it renders.
    """

    user_id: UUID
    primary_email: str
    display_name: str | None = None
    already_assigned: bool = False
    """True when this user already teaches the course: show as chosen rather
    than offering an assignment that would be a no-op."""

    model_config = ConfigDict(from_attributes=True)


class AssignTeacherRequest(BaseModel):
    user_id: UUID
    # Course-scoped title. Optional: the service derives it (first teacher is
    # always the Course Instructor; everyone else defaults to Teacher
    # Assistant). Must be ``course_instructor`` when no instructor exists yet.
    course_role: Literal["course_instructor", "teacher_assistant"] | None = None

    model_config = ConfigDict(extra="forbid")


class CourseTeacherRoleRequest(BaseModel):
    """Switch an assigned teacher's course-scoped title.

    Exactly one Course Instructor is enforced; see the service rules in
    ``courses.services.assignment.set_course_role``.
    """

    course_role: Literal["course_instructor", "teacher_assistant"]

    model_config = ConfigDict(extra="forbid")


class TeacherAssignmentCreated(BaseModel):
    """Response payload for POST -- the new (or pre-existing) assignment row."""

    id: UUID
    course_id: UUID
    user_id: UUID
    role_code: str
    scope_kind: str
    organization_id: UUID
    granted_by: UUID
    course_role: str | None = None

    model_config = ConfigDict(from_attributes=True)


class RosterEntry(BaseModel):
    """Single row in the roster response (shared by authoring + assignment routers).

    Phase 7 will replace this with the canonical ``EnrollmentRead``
    schema once the enrollments aggregate lands; the raw-SQL read here
    is the documented bridge per plan §4461.
    """

    enrollment_id: UUID
    student_id: UUID
    display_name: str | None = None
    primary_email: str
    status: str
    enrolled_at: datetime
    completed_at: datetime | None = None
    dropped_at: datetime | None = None
    # Short-TTL presigned GET URL for the student's uploaded avatar, minted by
    # the service layer. ``None`` when no avatar is set (SPA falls back to
    # initials). Not persisted — a projection.
    avatar_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class RosterStudentRead(BaseModel):
    """One student row for the teacher "Students" page.

    Wider than :class:`RosterEntry` — adds ``progress_percent`` /
    ``at_risk_level`` / ``last_activity_at`` computed from
    ``lesson_progress`` (same aggregation the Progress page uses, see
    ``queries/sql/roster_with_progress.sql``), and renames ``status`` to
    ``enrollment_status`` to match the SPA's ``RosterStudent`` type.
    """

    enrollment_id: UUID
    student_id: UUID
    display_name: str | None = None
    primary_email: str
    enrollment_status: str
    enrolled_at: datetime
    completed_at: datetime | None = None
    dropped_at: datetime | None = None
    progress_percent: float
    at_risk_level: str
    last_activity_at: datetime | None = None
    final_grade: str | None = None
    # Short-TTL presigned GET URL for the student's uploaded avatar, minted by
    # the service layer. ``None`` when no avatar is set (SPA falls back to
    # initials). Not persisted — a projection.
    avatar_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class CourseRosterRead(BaseModel):
    """Envelope for the teacher "Students" page roster response."""

    course_id: UUID
    students: list[RosterStudentRead]

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "AssignTeacherRequest",
    "AssignableTeacher",
    "CoursePathPlacement",
    "CourseReadiness",
    "CourseRosterRead",
    "CourseTeacherRoleRequest",
    "RosterEntry",
    "RosterStudentRead",
    "TeacherAssignmentCreated",
    "TeacherAssignmentRead",
]
