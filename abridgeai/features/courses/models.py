"""Courses aggregate ORM models.

Ports the legacy ``backend/app/models/course.py`` aggregate to the
feature-first layout, with the schema source of truth being the
baseline Alembic migration plus migration 0006 (which backfills the
unlock-config columns / lesson_prerequisites table that T3.1 demands).

Reconciliation directives applied (see plan §280-336):

* §A1 — 8-model aggregate. ``Tag``, ``CourseTag``,
  ``CourseLearningOutcome`` live HERE (not in identity / access_control
  / career_paths). ``CareerCourseItem`` is intentionally absent — it
  ports to ``features/career_paths/models.py`` in Phase 7 / T7.3.
  ``Enrollment`` and ``InvitationCode`` are also absent — they port to
  ``features/enrollments/models.py`` in Phase 7.
* §A2 — ``Course.published_at`` and ``Course.instructor_summary`` are
  intentionally NOT declared. ``Lesson.position`` likewise omitted;
  ordering lives on the parent ``ModuleItem.position`` link.
* §A3 — ``ModuleItem`` polymorphism keeps the canonical column names
  ``lesson_id`` / ``quiz_id`` / ``interview_config_id`` (NOT the
  hypothetical ``target_*`` variants).
* §A4 — ``Lesson.unlock_rule_json`` JSONB is preserved as the
  unstructured extension carrier; the structured columns
  ``ef_min_unlock`` / ``tau_unlock`` / ``requires_interview_pass``
  supplement it.
* §A11 — ``Course.slug`` uniqueness is the partial unique index
  ``uq_courses_org_slug`` from migration 0002 (T0.16). The mapped
  column does NOT carry ``unique=True``.
* §A12 — ``CourseLearningOutcome`` ports here with its own
  ``(course_id, position)`` unique constraint.
* §A13 — audit columns (``created_by`` / ``updated_by`` /
  ``deleted_at`` / ``deleted_by``) are NET-NEW and supplied by the
  mixins. They map onto the columns added by baseline migration
  0001's ``_add_audit_columns`` loop for every soft-deletable table
  in the courses aggregate (``courses``, ``modules``, ``lessons``,
  ``module_items``, ``lesson_resources``, ``course_learning_outcomes``).
  ``tags`` and ``course_tags`` are NOT in ``SOFT_DELETE_TABLES`` so
  they keep the minimal mixin set (``UUIDPrimaryKeyMixin`` +
  ``TimestampMixin`` for ``Tag``; ``CreatedAtMixin`` only for
  ``CourseTag``) — matching baseline DDL exactly.

Pydantic validators
-------------------
The ORM mappers carry DB-level CHECK constraints for the unlock-config
ranges (added by migration 0006). For Python-side validation of the
same ranges (e.g. before INSERT, or in API request handlers from
T3.2), :class:`LessonUnlockConfig` is exported as a Pydantic v2 model
sibling. It re-asserts ``1.3 <= ef_min_unlock <= 2.5`` and
``0.0 < tau_unlock <= 1.0`` so the test surface can exercise the
range gates without touching the database.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
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

_MODULE_ITEM_TYPE_CHECK = (
    "(item_type = 'lesson' AND lesson_id IS NOT NULL "
    "AND quiz_id IS NULL AND interview_config_id IS NULL) OR "
    "(item_type = 'quiz' AND lesson_id IS NULL "
    "AND quiz_id IS NOT NULL AND interview_config_id IS NULL) OR "
    "(item_type = 'interview' AND lesson_id IS NULL "
    "AND quiz_id IS NULL AND interview_config_id IS NOT NULL)"
)


class Course(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Top-level course owned by a teacher within an organization.

    ``owner_user_id`` is business ownership (the teacher whose course
    this is). ``created_by`` from :class:`AuditedByMixin` is record
    audit (whoever inserted the row, possibly an admin acting on
    behalf). The two are intentionally distinct columns per the draft
    "Naming convention" decision.
    """

    __tablename__ = "courses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_courses_status",
        ),
        CheckConstraint(
            "level IS NULL OR level IN ('beginner', 'intermediate', 'advanced')",
            name="ck_courses_level",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    org_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id", ondelete="SET NULL"),
    )
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="NO ACTION"),
        nullable=False,
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    level: Mapped[str | None] = mapped_column(String(20))
    thumbnail_object_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="SET NULL"),
    )
    estimated_minutes: Mapped[int | None] = mapped_column(Integer)
    expected_completion_days: Mapped[int | None] = mapped_column(Integer)
    enrollment_cap: Mapped[int | None] = mapped_column(Integer)

    modules: Mapped[list[Module]] = relationship(
        back_populates="course",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class Tag(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per Reconciliation §A1, ``Tag`` is a course-feature concern.

    Baseline DDL gives this table only ``slug`` + ``name`` columns and
    does NOT include it in ``SOFT_DELETE_TABLES`` — so no
    ``AuditedByMixin`` / ``SoftDeleteMixin`` here. The prompt's
    suggested ``organization_id`` / ``display_name`` shape is rejected
    in favor of baseline DDL per §A13 ("baseline > spec"). Future
    multi-tenant extensions can add columns via a separate Alembic
    migration.
    """

    __tablename__ = "tags"

    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)


class CourseTag(CreatedAtMixin, Base):
    """Append-only link table between :class:`Course` and :class:`Tag`.

    Baseline DDL gives this table only ``created_at`` (no
    ``created_by``, no audit columns). The prompt's
    ``created_by: UUID | None`` is intentionally NOT added — that
    would require an Alembic migration outside T3.1's scope. Same
    "baseline > spec" rule (§A13).
    """

    __tablename__ = "course_tags"

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        primary_key=True,
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tags.id", ondelete="NO ACTION"),
        primary_key=True,
    )


class CourseLearningOutcome(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    __tablename__ = "course_learning_outcomes"
    # Position uniqueness is enforced at the DB level by a partial expression
    # index over COALESCE(parent_id, course_id) (see migration 0059) so it is
    # per-parent (siblings) rather than course-wide. Not expressible as a
    # simple __table_args__ UniqueConstraint, hence no constraint here.

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        nullable=False,
    )
    # Self-referential parent for arbitrary-depth hierarchy (L.O.1 → L.O.1.1 →
    # L.O.1.1.1 …). NULL = top-level. ``position`` is now the sibling order
    # WITHIN a parent (top-level rows share parent_id IS NULL), so the display
    # code is the dotted path of positions walking root→leaf, derived at read
    # time and never stored. ON DELETE NO ACTION: we soft-delete and re-parent
    # in the service layer, never hard-delete.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("course_learning_outcomes.id", ondelete="NO ACTION"),
        nullable=True,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome_text: Mapped[str] = mapped_column(Text, nullable=False)


class Module(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Group of related lessons / quizzes / interviews inside a course.

    ``requires_all_lessons_unlocked`` defaults to FALSE per the locked
    user decision (§3900) — loose semantics. When TRUE, the entire
    module is gated behind every lesson within it being unlocked.
    """

    __tablename__ = "modules"
    __table_args__ = (
        UniqueConstraint("course_id", "position", name="uq_modules_course_position"),
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_modules_status",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    estimated_minutes: Mapped[int | None] = mapped_column(Integer)
    requires_all_lessons_unlocked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )

    course: Mapped[Course] = relationship(back_populates="modules")
    lessons: Mapped[list[Lesson]] = relationship(
        back_populates="module",
        cascade="save-update, merge, refresh-expire, expunge",
    )
    items: Mapped[list[ModuleItem]] = relationship(
        back_populates="module",
        cascade="save-update, merge, refresh-expire, expunge",
        foreign_keys="[ModuleItem.module_id]",
    )


class ModulePrerequisite(CreatedAtMixin, Base):
    __tablename__ = "module_prerequisites"
    __table_args__ = (
        CheckConstraint(
            "module_id <> prerequisite_module_id",
            name="ck_module_prerequisites_not_self",
        ),
    )

    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
        primary_key=True,
    )
    prerequisite_module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
        primary_key=True,
    )


class Lesson(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Individual lesson within a :class:`Module`.

    Unlock-config columns (added by migration 0006 per §A4):

    * ``ef_min_unlock`` — easiness threshold for SR cards.
    * ``tau_unlock`` — proportion of cards above ``ef_min`` required.
    * ``requires_interview_pass`` — gates lesson behind interview pass.
    * ``unlock_rule_json`` — unstructured extension carrier; KEPT per
      §A4 alongside the structured columns.

    Note: ``Lesson.position`` is intentionally absent (§A2) — ordering
    lives on the parent ``ModuleItem.position`` link.
    """

    __tablename__ = "lessons"
    __table_args__ = (
        UniqueConstraint("module_id", "slug", name="uq_lessons_module_slug"),
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_lessons_status",
        ),
        CheckConstraint(
            "lesson_type IN ('video', 'reading')",
            name="ck_lessons_lesson_type",
        ),
        CheckConstraint(
            "ef_min_unlock >= 1.3 AND ef_min_unlock <= 2.5",
            name="ck_lessons_ef_min_unlock_range",
        ),
        CheckConstraint(
            "tau_unlock > 0.0 AND tau_unlock <= 1.0",
            name="ck_lessons_tau_unlock_range",
        ),
    )

    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    notes_markdown: Mapped[str | None] = mapped_column(Text)
    primary_material_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_materials.id", ondelete="SET NULL"),
    )
    lesson_type: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'video'")
    )
    difficulty: Mapped[str | None] = mapped_column(String(20))
    estimated_minutes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    ef_min_unlock: Mapped[Decimal] = mapped_column(
        Numeric, nullable=False, server_default=text("2.0")
    )
    tau_unlock: Mapped[Decimal] = mapped_column(Numeric, nullable=False, server_default=text("0.8"))
    requires_interview_pass: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    unlock_rule_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    module: Mapped[Module] = relationship(back_populates="lessons")
    resources: Mapped[list[LessonResource]] = relationship(
        back_populates="lesson",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class LessonPrerequisite(CreatedAtMixin, Base):
    """Append-only self-edge between :class:`Lesson` rows.

    Mirrors :class:`ModulePrerequisite`. The DB-level
    ``ck_lesson_prerequisites_not_self`` CHECK rejects self-prereq
    cycles. ON DELETE NO ACTION matches T0.14 (lessons is a
    soft-deletable parent).
    """

    __tablename__ = "lesson_prerequisites"
    __table_args__ = (
        CheckConstraint(
            "lesson_id <> prereq_lesson_id",
            name="ck_lesson_prerequisites_not_self",
        ),
    )

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        primary_key=True,
    )
    prereq_lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        primary_key=True,
    )


class LessonResource(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "lesson_resources"
    __table_args__ = (
        UniqueConstraint("lesson_id", "position", name="uq_lesson_resources_position"),
        CheckConstraint(
            "resource_type IN ('pdf', 'zip', 'mp4', 'xlsx', 'pptx', 'docx', 'link', 'other')",
            name="ck_lesson_resources_resource_type",
        ),
    )

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(20), nullable=False)
    storage_object_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="SET NULL"),
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    visible_to_students: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )

    lesson: Mapped[Lesson] = relationship(back_populates="resources")


class ModuleItem(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Polymorphic ordering row inside a :class:`Module`.

    Per §A3, exactly one of ``lesson_id`` / ``quiz_id`` /
    ``interview_config_id`` is non-NULL — enforced by the XOR CHECK
    constraint copied verbatim from baseline migration line 921-925.
    """

    __tablename__ = "module_items"
    __table_args__ = (
        UniqueConstraint("module_id", "position", name="uq_module_items_position"),
        CheckConstraint(
            "item_type IN ('lesson', 'quiz', 'interview')",
            name="ck_module_items_item_type_enum",
        ),
        CheckConstraint(
            _MODULE_ITEM_TYPE_CHECK,
            name="ck_module_items_item_type",
        ),
    )

    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    item_type: Mapped[str] = mapped_column(String(20), nullable=False)
    lesson_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
    )
    quiz_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quizzes.id", ondelete="NO ACTION"),
    )
    interview_config_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_configs.id", ondelete="NO ACTION"),
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    unlock_rule_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    module: Mapped[Module] = relationship(
        back_populates="items",
        foreign_keys=[module_id],
    )
    lesson: Mapped[Lesson | None] = relationship(
        foreign_keys=[lesson_id],
        lazy="noload",
    )
    quiz: Mapped[Any] = relationship(
        "Quiz",
        foreign_keys=[quiz_id],
        lazy="noload",
    )
    interview_config: Mapped[Any] = relationship(
        "InterviewConfig",
        foreign_keys=[interview_config_id],
        lazy="noload",
    )


__all__ = [
    "Course",
    "CourseLearningOutcome",
    "CourseTag",
    "Lesson",
    "LessonPrerequisite",
    "LessonResource",
    "Module",
    "ModuleItem",
    "ModulePrerequisite",
    "Tag",
]
