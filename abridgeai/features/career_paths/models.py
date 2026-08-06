from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from abridgeai.features.access_control.models import (
    CareerPath,
    StudentCareerEnrollment,
)


class CareerPathStage(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """An ordered group of courses inside a career path (migration 0070).

    A stage carries its own gating policy, so a path is a sequence of gates
    rather than a flat list:

    * ``unlock_policy`` — when the stage opens for a student.
      ``always`` (never gated) / ``after_previous`` (previous stage
      *complete*, i.e. required **and** the min-optional quota) /
      ``after_previous_required`` (previous stage's required courses only —
      lets a student move on without finishing electives).
    * ``enforcement`` — what a locked stage *does*: ``hard`` blocks the
      course read, ``soft`` warns, ``advisory`` only decorates the UI.
      It governs **stage lock only**; it has no bearing on
      ``career_paths.max_concurrent``, which never blocks.
    * ``min_optional_to_complete`` — how many of the stage's optional
      courses count toward completing it (0 ⇒ optional courses are purely
      extra).

    ``title`` is NULLABLE by design: NULL means "unnamed", and the client
    renders ``Stage {position}`` in the user's own locale. Storing a
    server-side English default would leak untranslated text into a
    Vietnamese UI.

    **Position 1 is an implicit unlock override.** ``unlocked(stage_1)`` is
    unconditionally true regardless of this column, because a path whose
    first stage is locked can never be started by anyone. The stored policy
    on stage 1 is inert but preserved rather than normalised: reordering a
    stage away from position 1 must restore the manager's original intent,
    which is impossible if reorder rewrites the column. The editor hides the
    control on the first stage, and reorder *warns* instead of rewriting.
    """

    __tablename__ = "career_path_stages"
    __table_args__ = (
        CheckConstraint("position > 0", name="career_path_stages_position_check"),
        CheckConstraint(
            "min_optional_to_complete >= 0",
            name="career_path_stages_min_optional_to_complete_check",
        ),
        CheckConstraint(
            "unlock_policy IN ('always','after_previous','after_previous_required')",
            name="career_path_stages_unlock_policy_check",
        ),
        CheckConstraint(
            "enforcement IN ('hard','soft','advisory')",
            name="career_path_stages_enforcement_check",
        ),
    )

    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    min_optional_to_complete: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    unlock_policy: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'after_previous'")
    )
    enforcement: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'soft'")
    )

    career_path: Mapped[CareerPath] = relationship()


class CareerPathCourse(CreatedAtMixin, Base):
    __tablename__ = "career_course_items"
    __table_args__ = (
        # (stage_id, position) is a UNIQUE INDEX created in migration 0070,
        # NOT a table constraint, and NOT partial — this table has no
        # deleted_at column (0002 skip-list), so a partial filter on
        # deleted_at would error out at index-creation time.
        UniqueConstraint(
            "stage_id",
            "position",
            name="career_course_items_stage_position_key",
        ),
        CheckConstraint("position > 0", name="career_course_items_position_check"),
        CheckConstraint(
            "satisfied_by IN ('completion','pass')",
            name="career_course_items_satisfied_by_check",
        ),
    )

    # PRIMARY KEY (career_path_id, course_id) is KEPT deliberately: it is
    # what makes "the same course in two stages of one path" structurally
    # impossible. Re-keying on (stage_id, course_id) would permit it.
    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="CASCADE"),
        primary_key=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        primary_key=True,
    )
    stage_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_path_stages.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    is_required: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))
    satisfied_by: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'completion'")
    )
    """How this item becomes ``satisfied``.

    ``completion`` (the only behaviour implemented) ⟺
    ``course_enrollments.status = 'completed'``. ``pass`` is accepted by the
    CHECK so the column can carry a graded variant later without a
    migration; until that evaluator exists it is treated as ``completion``.
    """

    career_path: Mapped[CareerPath] = relationship()
    stage: Mapped[CareerPathStage] = relationship()


class StudentStageProgress(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Append-only latch: this enrollment completed this stage.

    Append-only latch. Deliberately omits ``TimestampMixin`` /
    ``AuditedByMixin`` / ``SoftDeleteMixin``. Adding ``SoftDeleteMixin``
    would let a soft-deleted latch row un-complete a student — the exact
    failure this table exists to prevent. Rows are never updated or deleted.

    Why a latch at all: stage completion is derived from course completion,
    and course completion can move **backward** (``progress.services.
    tracking.unmark_lesson_complete`` recomputes status from engagement, so a
    student can un-tick a lesson). A manager editing the path — removing an
    elective, raising ``min_optional_to_complete`` — can also change the
    derived answer under a student's feet. Without a latch either event would
    silently re-lock a stage the student had already passed.

    The asymmetry is intentional and load-bearing: un-marking a lesson
    demotes the *course* (``'completed' → 'active'``, so ``satisfied``
    stays honest) but does **not** un-latch the *stage*. A stage that ever
    completed stays complete.

    Mixin choice follows the two in-repo append-only precedents,
    :class:`CareerReadinessSnapshot` and ``CardReview``: both are
    ``(UUIDPrimaryKeyMixin, CreatedAtMixin, Base)``.
    """

    __tablename__ = "student_stage_progress"
    __table_args__ = (
        UniqueConstraint(
            "enrollment_id",
            "stage_id",
            name="student_stage_progress_enrollment_id_stage_id_key",
        ),
    )

    enrollment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("student_career_enrollments.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    stage_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_path_stages.id", ondelete="NO ACTION"),
        nullable=False,
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


class CareerReadinessSnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Point-in-time readiness score for a (student, career path) pair.

    FR-6.8: written nightly by the readiness cron (and on demand) so
    managers can aggregate pathway-level outcomes over time. Maps the
    baseline-DDL ``career_readiness_snapshots`` table (migration 0001);
    ``readiness_score`` is the 0–100 pathway completion aggregate from
    :func:`features.career_paths.services.enrollment.get_my_path_progress`.
    """

    __tablename__ = "career_readiness_snapshots"
    __table_args__ = (
        CheckConstraint(
            "readiness_score BETWEEN 0 AND 100",
            name="career_readiness_snapshots_readiness_score_check",
        ),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="CASCADE"),
        nullable=False,
    )
    readiness_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    formula_version: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("1")
    )
    """Which progress formula produced ``readiness_score``.

    Server default is **1**, matching the default of the
    ``careerpath.progress_formula_version`` runtime setting — a default of 2
    while the formula was still gated to 1 would mislabel every snapshot
    written before the cutover, defeating the column's only purpose.
    :mod:`services.readiness` writes the value it actually used explicitly
    rather than relying on this default, and the readiness chart must
    segment (or annotate) on a version change: honest data on an
    unsegmented line still misleads.
    """


__all__ = [
    "CareerPath",
    "CareerPathCourse",
    "CareerPathStage",
    "CareerReadinessSnapshot",
    "StudentCareerEnrollment",
    "StudentStageProgress",
]
