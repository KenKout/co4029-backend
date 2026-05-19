"""Teacher-facing (authoring) DTOs for the courses aggregate.

Each Authoring schema inherits from its Public counterpart and widens /
adds fields that are only safe to expose to course authors:

* widened ``status`` enum (``draft`` / ``archived`` in addition to
  ``published``);
* audit columns from :class:`~abridgeai.core.db.AuditedByMixin` and
  :class:`~abridgeai.core.db.SoftDeleteMixin`
  (``created_by`` / ``updated_by`` / ``created_at`` / ``updated_at`` /
  ``deleted_at``);
* unlock-config columns on :class:`LessonAuthoring` per §A4 (the
  structured ``ef_min_unlock`` / ``tau_unlock`` /
  ``requires_interview_pass`` columns plus the ``unlock_rule_json``
  extension carrier);
* lesson / module prerequisite collections, hydrated by the service
  layer from ``ModulePrerequisite`` and ``LessonPrerequisite`` link
  tables.

Field drops vs plan §3970-3991 (per T3.1 ORM ground truth, §A13)
----------------------------------------------------------------
The plan body suggested several "draft_notes" / "internal_notes" columns
that do NOT exist in the baseline DDL. Per §A13 ("baseline > spec") we
do NOT fabricate columns to match the spec; the dropped fields are:

* ``CourseAuthoring.internal_notes``  — Course has no such column.
* ``ModuleAuthoring.draft_notes``     — Module has no such column.
* ``LessonAuthoring.draft_notes``     — Lesson has no such column.
* ``LessonResourceAuthoring.internal_notes`` — LessonResource has none.
* ``CourseLearningOutcomeAuthoring.status``  — outcomes have no status
  column (only ``position`` + ``outcome_text``).
* ``TagAuthoring.organization_id``    — Tag has no org scoping.

If a future migration adds them, this module is the only place that
needs updating (Public schemas continue to hide them automatically).

The :class:`InstructorAuthoring` extension exposes ``primary_email``
(joined from ``users.primary_email``); :class:`InstructorRead` from
``public.py`` deliberately omits it. ``test_public_excludes_internal_fields``
verifies the leak guard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from abridgeai.features.courses.schemas.public import (
    CourseLearningOutcomePublic,
    CoursePublic,
    InstructorRead,
    LessonPublic,
    LessonResourcePublic,
    ModuleItemPublic,
    ModulePublic,
    TagPublic,
)


class InstructorAuthoring(InstructorRead):
    """Instructor view for authoring screens — exposes the user's primary email."""

    primary_email: str


class TagAuthoring(TagPublic):
    """Authoring projection of :class:`Tag`.

    Tag is NOT in ``SOFT_DELETE_TABLES`` (§A13) — only the base
    ``created_at`` / ``updated_at`` apply. ``organization_id`` is NOT a
    column in the baseline DDL and is therefore not exposed here.
    """

    created_at: datetime
    updated_at: datetime


class CourseLearningOutcomeAuthoring(CourseLearningOutcomePublic):
    """Authoring projection of a course learning outcome.

    Outcome rows carry the standard audit + soft-delete column set.
    """

    course_id: UUID
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class CourseAuthoring(CoursePublic):
    """Authoring projection of :class:`Course`.

    Inherits the public schema and widens ``status`` to the full
    Literal (the ORM CHECK constraint allows all three values). Field
    types for ``tags`` / ``outcomes`` are rebound to their authoring
    variants — Pydantic v2 supports re-typing inherited fields in
    sub-classes.
    """

    status: Literal["draft", "published", "archived"]  # type: ignore[assignment]
    org_unit_id: UUID | None = None
    owner_user_id: UUID
    level: str | None = None
    thumbnail_object_id: UUID | None = None
    estimated_minutes: int | None = None
    expected_completion_days: int | None = None
    enrollment_cap: int | None = None
    instructor: InstructorAuthoring | None = None
    tags: list[TagAuthoring] = []  # type: ignore[assignment]
    outcomes: list[CourseLearningOutcomeAuthoring] = []  # type: ignore[assignment]
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class ModuleAuthoring(ModulePublic):
    description: str | None = None
    status: Literal["draft", "published", "archived"]
    estimated_minutes: int | None = None
    requires_all_lessons_unlocked: bool = False
    prerequisites: list[UUID] = []
    items: list[ModuleItemAuthoring] = Field(default_factory=list)  # type: ignore[assignment]
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def _strip_orm_items_relationship(cls, data: Any) -> Any:  # noqa: ANN401 - Pydantic before-validator: opaque input
        """Skip the ``Module.items`` ORM relationship during from_attributes.

        Pydantic v2 with ``from_attributes=True`` getattrs every field;
        for the ``items`` relationship that triggers a sync SQLAlchemy
        lazy-load inside an async context (``MissingGreenlet``). The
        authoring content tree composer populates ``items`` explicitly
        from a JSON dict; bare-module fetches just want an empty list.

        Same applies to ``prerequisites`` — that's a separately-loaded
        join the service hydrates only when needed.
        """
        if isinstance(data, dict | BaseModel):
            return data
        if hasattr(data, "_sa_instance_state"):
            shim: dict[str, Any] = {}
            for name in cls.model_fields:
                if name in {"items", "prerequisites"}:
                    continue
                shim[name] = getattr(data, name, None)
            shim["items"] = []
            shim["prerequisites"] = []
            return shim
        return data


class LessonAuthoring(LessonPublic):
    """Authoring projection of :class:`Lesson`.

    Carries the full unlock-config column set per §A4 (structured
    columns + the ``unlock_rule_json`` extension carrier) plus the
    ``prereq_lesson_ids`` collection joined from ``LessonPrerequisite``.

    Note: the legacy ORM uses ``summary`` (not ``description``) and has
    no ``draft_notes`` column — both are reflected here.
    """

    module_id: UUID
    slug: str
    summary: str | None = None
    notes_markdown: str | None = None
    primary_material_id: UUID | None = None
    lesson_type: str = "video"
    difficulty: str | None = None
    estimated_minutes: int | None = None
    status: Literal["draft", "published", "archived"]
    ef_min_unlock: float = 2.0
    tau_unlock: float = 0.8
    requires_interview_pass: bool = False
    unlock_rule_json: dict[str, Any] = {}
    prereq_lesson_ids: list[UUID] = []
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class LessonResourceAuthoring(LessonResourcePublic):
    storage_object_id: UUID | None = None
    position: int
    visible_to_students: bool = True
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class ModuleItemTarget(BaseModel):
    """Polymorphic target for authoring items — wider than LessonPublic."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    slug: str | None = None
    lesson_type: str | None = None
    status: str | None = None
    summary: str | None = None
    estimated_minutes: int | None = None
    difficulty: str | None = None


class ModuleItemAuthoring(ModuleItemPublic):
    target: ModuleItemTarget | None = None  # type: ignore[assignment]
    lesson_id: UUID | None = None
    quiz_id: UUID | None = None
    interview_config_id: UUID | None = None
    unlock_rule_json: dict[str, Any] = {}
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class CourseContentAuthoring(BaseModel):
    """Composed authoring tree — course + draft + published modules.

    The nested item / lesson / resource trees are composed by the
    service layer (T3.5); this DTO carries the top-level slices that
    every authoring view shares.
    """

    model_config = ConfigDict(from_attributes=True)

    course: CourseAuthoring
    modules: list[ModuleAuthoring] = []


__all__ = [
    "CourseAuthoring",
    "CourseContentAuthoring",
    "CourseLearningOutcomeAuthoring",
    "InstructorAuthoring",
    "LessonAuthoring",
    "LessonResourceAuthoring",
    "ModuleAuthoring",
    "ModuleItemAuthoring",
    "TagAuthoring",
]
