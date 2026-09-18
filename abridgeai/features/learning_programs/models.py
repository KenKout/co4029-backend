from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

PATH_CHANGE_OPEN_STATUSES = ("pending", "in_progress")
"""Statuses that occupy the one-open-request-per-enrolment slot.

Mirrors the partial unique index ``uq_path_change_requests_one_open``
(migration 0097). ``in_progress`` is an acknowledgement, not a decision, so it
must keep blocking a second request the same way ``pending`` does.
"""

PATH_REQUEST_KINDS = ("change", "drop")
"""What a student is asking for on their own enrolment.

``change`` moves one attempt to a different Career Path; ``drop`` ends one
without a replacement. They share this table, the review queue and the switch
budget, and differ only in whether a destination exists — see
``ck_path_change_requests_kind_target`` (migration 0122).
"""

PATH_CHANGE_REJECTION_REASON_CODES = (
    "insufficient_justification",
    "progress_loss_too_high",
    "target_path_not_suitable",
    "preserve_remaining_switch",
    "advising_required",
    "documentation_missing",
    "other",
)
"""Fixed vocabulary a Faculty Dean picks from when rejecting a request.

Kept in sync with ``ck_path_change_requests_decision_reason_code`` and with the
frontend's reject dialog. ``other`` exists so a dean is never forced into a
wrong bucket, and it REQUIRES the free-text ``decision_reason``. An optional
``decision_note`` remains separate for every category.
"""


class LearningProgram(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "learning_programs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft','published','archived')",
            name="ck_learning_programs_status",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="NO ACTION"), index=True
    )
    faculty_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("org_units.id", ondelete="NO ACTION"), index=True
    )
    slug: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'draft'"))


#: Upper bound for a program version's own career-path limit, when it sets one.
#:
#: The limit itself is NULLABLE, and null is the interesting value: it means
#: the program imposes no cap of its own and the student is bounded by the
#: organization's ``learning_program.max_concurrent_paths_per_student`` --
#: and, implicitly, by how many paths the program actually offers.
#:
#: An organization-level ceiling on this number used to exist as a runtime
#: setting. Its default equalled this bound, so out of the box it constrained
#: nothing while adding a second place the same number could be refused.
#:
#: Exported so the authoring API can hand the manager's picker the same bound
#: the CHECK constraint enforces, instead of the SPA hardcoding a 10.
MAX_CAREER_PATHS_PER_ENROLLMENT = 10


class LearningProgramVersion(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    __tablename__ = "learning_program_versions"
    __table_args__ = (
        UniqueConstraint(
            "learning_program_id", "version_no", name="uq_learning_program_versions_no"
        ),
        CheckConstraint(
            "status IN ('draft','published')", name="ck_learning_program_versions_status"
        ),
        CheckConstraint("version_no > 0", name="ck_learning_program_versions_no"),
        CheckConstraint("max_path_switches >= 0", name="ck_learning_program_versions_switches"),
        CheckConstraint(
            f"max_career_paths_per_enrollment BETWEEN 1 AND {MAX_CAREER_PATHS_PER_ENROLLMENT}",
            name="ck_learning_program_versions_career_path_limit",
        ),
    )

    learning_program_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("learning_programs.id", ondelete="NO ACTION"), index=True
    )
    version_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'draft'"))
    max_path_switches: Mapped[int] = mapped_column(Integer, server_default=text("3"))
    #: Paths a student may hold in THIS program, or NULL for "no cap of its
    #: own". Nullable rather than a sentinel like 0 or the bound itself,
    #: because "the manager did not cap this" and "the manager capped it at
    #: the maximum" are different intentions -- and the second silently stops
    #: meaning what it meant if the bound ever moves.
    #:
    #: No server default: a row that does not name a limit does not have one.
    max_career_paths_per_enrollment: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LearningProgramVersionPath(CreatedAtMixin, Base):
    __tablename__ = "learning_program_version_paths"
    __table_args__ = (
        UniqueConstraint(
            "program_version_id", "position", name="uq_learning_program_version_paths_position"
        ),
        CheckConstraint("position > 0", name="ck_learning_program_version_paths_position"),
    )

    program_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_program_versions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="NO ACTION"),
        primary_key=True,
    )
    career_path_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_path_versions.id", ondelete="NO ACTION")
    )
    position: Mapped[int] = mapped_column(Integer)
    # Nullable-by-version semantics are intentional: versions published before
    # the default-path feature keep every mapping False.  Every newly published
    # version is validated by the service to have exactly one True row.
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))


class ProgramEnrollment(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, Base):
    __tablename__ = "program_enrollments"
    __table_args__ = (
        UniqueConstraint(
            "learning_program_id", "student_id", name="uq_program_enrollments_program_student"
        ),
        CheckConstraint(
            "status IN ('awaiting_path','active','completed','withdrawn','cancelled')",
            name="ck_program_enrollments_status",
        ),
    )

    learning_program_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("learning_programs.id", ondelete="NO ACTION"), index=True
    )
    program_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("learning_program_versions.id", ondelete="NO ACTION")
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), server_default=text("'awaiting_path'"))
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    withdrawal_reason: Mapped[str | None] = mapped_column(Text)


class ProgramPathAttempt(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, Base):
    __tablename__ = "program_path_attempts"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active','completed','switched_out','cancelled')",
            name="ck_program_path_attempts_status",
        ),
        CheckConstraint(
            "selection_source IN ('student','program_default','path_change')",
            name="ck_program_path_attempts_selection_source",
        ),
    )

    program_enrollment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_enrollments.id", ondelete="NO ACTION"), index=True
    )
    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_paths.id", ondelete="NO ACTION")
    )
    career_path_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_path_versions.id", ondelete="NO ACTION")
    )
    previous_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_path_attempts.id", ondelete="NO ACTION")
    )
    status: Mapped[str] = mapped_column(String(20), server_default=text("'active'"))
    selection_source: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'student'")
    )
    selected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class PathChangeRequest(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, Base):
    __tablename__ = "path_change_requests"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','in_progress','approved','rejected','cancelled','invalidated')",
            name="ck_path_change_requests_status",
        ),
        CheckConstraint(
            "kind IN ('change','drop')",
            name="ck_path_change_requests_kind",
        ),
        CheckConstraint(
            "(kind = 'change' AND target_career_path_id IS NOT NULL "
            "AND target_career_path_version_id IS NOT NULL) "
            "OR (kind = 'drop' AND target_career_path_id IS NULL "
            "AND target_career_path_version_id IS NULL)",
            name="ck_path_change_requests_kind_target",
        ),
        CheckConstraint(
            "decision_reason_code IS NULL OR decision_reason_code IN ("
            "'insufficient_justification','progress_loss_too_high',"
            "'target_path_not_suitable','preserve_remaining_switch',"
            "'advising_required','documentation_missing','other')",
            name="ck_path_change_requests_decision_reason_code",
        ),
    )

    program_enrollment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_enrollments.id", ondelete="NO ACTION")
    )
    from_attempt_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_path_attempts.id", ondelete="NO ACTION")
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'change'"))
    # NULL for a drop, which has no destination. The kind/target CHECK above is
    # what keeps that from degrading into "a switch with a missing target".
    target_career_path_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_paths.id", ondelete="NO ACTION")
    )
    target_career_path_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_path_versions.id", ondelete="NO ACTION")
    )
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'pending'"))
    # Acknowledgement (``in_progress``) is distinct from the decision: the dean
    # has opened the request and is verifying data, but nothing about the
    # student's path has changed yet. Kept in its own columns so a later
    # approve/reject does not overwrite who first picked the request up.
    in_progress_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    in_progress_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="NO ACTION")
    )
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="NO ACTION")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_reason_code: Mapped[str | None] = mapped_column(String(40))
    decision_reason: Mapped[str | None] = mapped_column(Text)
    decision_note: Mapped[str | None] = mapped_column(Text)
    new_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_path_attempts.id", ondelete="NO ACTION")
    )


class CourseCompletionAward(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "course_completion_awards"
    __table_args__ = (
        UniqueConstraint(
            "student_id", "course_id", name="uq_course_completion_awards_student_course"
        ),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("courses.id", ondelete="NO ACTION")
    )
    awarded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    source_enrollment_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("course_enrollments.id", ondelete="SET NULL")
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="NO ACTION")
    )
    revocation_reason: Mapped[str | None] = mapped_column(Text)


class CourseEnrollmentEntitlement(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "course_enrollment_entitlements"
    __table_args__ = (
        UniqueConstraint(
            "course_enrollment_id",
            "source_type",
            "source_id",
            name="uq_course_enrollment_entitlements_source",
        ),
        CheckConstraint(
            "source_type IN ('path_attempt','manual','invitation','legacy')",
            name="ck_course_enrollment_entitlements_source_type",
        ),
    )

    course_enrollment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("course_enrollments.id", ondelete="CASCADE")
    )
    source_type: Mapped[str] = mapped_column(String(30))
    source_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True))
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )


__all__ = [
    "MAX_CAREER_PATHS_PER_ENROLLMENT",
    "PATH_CHANGE_OPEN_STATUSES",
    "PATH_CHANGE_REJECTION_REASON_CODES",
    "PATH_REQUEST_KINDS",
    "CourseCompletionAward",
    "CourseEnrollmentEntitlement",
    "LearningProgram",
    "LearningProgramVersion",
    "LearningProgramVersionPath",
    "PathChangeRequest",
    "ProgramEnrollment",
    "ProgramPathAttempt",
]
