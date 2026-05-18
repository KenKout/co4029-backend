"""Enrollments aggregate ORM models (T7.1).

Ports the legacy ``CourseEnrollment`` + ``CourseInvitationCode`` models
to the feature-first layout. Tables are baselined in
``migrations/0001_baseline_schema.py`` (lines 404-432); migration 0008
adds the audit / soft-delete / multi-tenant columns the mixins demand.

Reconciliation directives applied:

* §A13 — baseline > spec. ``course_enrollments.status`` keeps the
  baseline domain ``{active, completed, dropped, waitlisted}`` rather
  than the plan's suggested ``{active, dropped, completed, suspended}``.
  The plan's lifecycle invariants (``status='dropped'`` for unenroll,
  ``status='active'`` for fresh enroll) operate on this domain
  unchanged.
* :class:`Enrollment` carries ``UUIDPrimaryKey + Timestamp + AuditedBy``
  (NO :class:`SoftDeleteMixin` — uses ``status`` lifecycle field).
* :class:`InvitationCode` carries the full
  ``UUIDPrimaryKey + Timestamp + AuditedBy + SoftDelete`` set so
  Manager-issued codes can be revoked while preserving audit history.
* The legacy ``invitation_code_id`` FK from ``course_enrollments``
  (set when a student enrolled via a code in the legacy self-enroll
  flow) is preserved as a nullable column. T7.1 does NOT issue redeem
  paths, but the column stays for read-only audit per
  ``backend/app/routes/courses/router.py`` parity.
* ``course_invitation_codes`` retains the legacy ``is_active`` flag
  alongside the soft-delete pair. Soft-delete is the canonical
  expire-and-archive signal post-T7.1; ``is_active=FALSE`` is kept for
  legacy reads.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class Enrollment(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, Base):
    """Single enrollment row per (course, student) pair.

    Lifecycle is tracked via the ``status`` column rather than soft
    delete: the unique constraint ``(course_id, student_id)`` prevents
    re-enrollment, and an "unenroll" sets ``status='dropped'`` while
    keeping the row intact for audit.
    """

    __tablename__ = "course_enrollments"
    __table_args__ = (
        UniqueConstraint(
            "course_id",
            "student_id",
            name="course_enrollments_course_id_student_id_key",
        ),
        CheckConstraint(
            "status IN ('active', 'completed', 'dropped', 'waitlisted')",
            name="course_enrollments_status_check",
        ),
        CheckConstraint(
            "source IN ('self_enroll', 'invite_code', 'admin_bulk', 'manager_bulk', 'manual')",
            name="course_enrollments_source_check",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'active'"))
    waitlist_position: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'self_enroll'")
    )
    invitation_code_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("course_invitation_codes.id", ondelete="SET NULL"),
    )
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dropped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    invitation_code: Mapped[InvitationCode | None] = relationship(
        back_populates="enrollments",
        foreign_keys=[invitation_code_id],
    )


class InvitationCode(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Manager-issued invitation code. NOT student-redeemable in T7.1.

    Per the locked T7.1 decision, codes are tracking artefacts that
    Managers hand out via email; there is no student-facing redemption
    endpoint. ``current_uses`` increments only when a Manager-side bulk
    flow attributes an enrollment to a code (future hook); soft-delete
    is the canonical revoke signal.
    """

    __tablename__ = "course_invitation_codes"
    __table_args__ = (
        CheckConstraint(
            "max_uses IS NULL OR max_uses > 0",
            name="course_invitation_codes_max_uses_check",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    max_uses: Mapped[int | None] = mapped_column(Integer)
    current_uses: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))

    enrollments: Mapped[list[Enrollment]] = relationship(
        back_populates="invitation_code",
        foreign_keys="[Enrollment.invitation_code_id]",
        cascade="save-update, merge, refresh-expire, expunge",
    )


__all__ = ["Enrollment", "InvitationCode"]
