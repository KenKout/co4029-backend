from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
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
    owner_faculty_dean_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="NO ACTION")
    )
    slug: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'draft'"))


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
    )

    learning_program_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("learning_programs.id", ondelete="NO ACTION"), index=True
    )
    version_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'draft'"))
    max_path_switches: Mapped[int] = mapped_column(Integer, server_default=text("3"))
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
    selected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class PathChangeRequest(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, Base):
    __tablename__ = "path_change_requests"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','approved','rejected','cancelled','invalidated')",
            name="ck_path_change_requests_status",
        ),
    )

    program_enrollment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_enrollments.id", ondelete="NO ACTION")
    )
    from_attempt_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("program_path_attempts.id", ondelete="NO ACTION")
    )
    target_career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_paths.id", ondelete="NO ACTION")
    )
    target_career_path_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("career_path_versions.id", ondelete="NO ACTION")
    )
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'pending'"))
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="NO ACTION")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_reason: Mapped[str | None] = mapped_column(Text)
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
    "CourseCompletionAward",
    "CourseEnrollmentEntitlement",
    "LearningProgram",
    "LearningProgramVersion",
    "LearningProgramVersionPath",
    "PathChangeRequest",
    "ProgramEnrollment",
    "ProgramPathAttempt",
]
