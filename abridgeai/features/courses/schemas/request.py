"""Request body schemas for the courses aggregate (T3.2 §3989-3991).

These DTOs are used by the learner / authoring routers (T3.6 / T3.7)
and validated at the API boundary BEFORE the service layer is called.
The unlock-config range gates duplicate the ranges enforced as DB
CHECK constraints (`ck_lessons_ef_min_unlock_range`,
`ck_lessons_tau_unlock_range`) and as Pydantic validators on
:class:`~abridgeai.features.courses.models.LessonUnlockConfig`. The
duplication is intentional — the API layer rejects malformed input
without paying a round-trip to Postgres.

State transitions for ``CourseUpdate.status`` (draft → published →
archived) are validated in the service layer, not here. The schema
only narrows the value to one of the three legal Literal options.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictRequest(BaseModel):
    """Reject unknown keys to surface typos in API payloads early."""

    model_config = ConfigDict(extra="forbid")


class LessonUnlockConfig(_StrictRequest):
    """Pydantic v2 sibling that re-asserts the lesson-unlock range gates.

    Per plan §3894 / §3927, the same ``ef_min_unlock`` and
    ``tau_unlock`` ranges enforced as DB CHECK constraints on
    :class:`~abridgeai.features.courses.models.Lesson` are also enforced
    at the application boundary so request handlers / services can reject
    malformed input before issuing INSERT / UPDATE.
    """

    ef_min_unlock: float = 2.0
    tau_unlock: float = 0.8
    requires_interview_pass: bool = False

    @field_validator("ef_min_unlock")
    @classmethod
    def _check_ef_min_unlock_range(cls, value: float) -> float:
        if not (1.3 <= value <= 2.5):
            raise ValueError("ef_min_unlock must be in [1.3, 2.5]")
        return value

    @field_validator("tau_unlock")
    @classmethod
    def _check_tau_unlock_range(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError("tau_unlock must be in (0.0, 1.0]")
        return value


class CourseCreate(_StrictRequest):
    org_unit_id: UUID | None = None
    slug: str = Field(max_length=100)
    title: str = Field(max_length=255)
    description: str | None = None
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    thumbnail_object_id: UUID | None = None
    estimated_minutes: int | None = None
    expected_completion_days: int | None = None
    enrollment_cap: int | None = None
    # Teacher contact info surfaced on the student landing page (all optional).
    contact_email: str | None = Field(default=None, max_length=320)
    contact_phone: str | None = Field(default=None, max_length=50)
    contact_website_url: str | None = Field(default=None, max_length=500)
    contact_social_url: str | None = Field(default=None, max_length=500)


class CourseUpdate(_StrictRequest):
    """Partial update for a course. State transitions checked in service."""

    org_unit_id: UUID | None = None
    slug: str | None = Field(default=None, max_length=100)
    title: str | None = Field(default=None, max_length=255)
    description: str | None = None
    status: Literal["draft", "published", "archived"] | None = None
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    thumbnail_object_id: UUID | None = None
    estimated_minutes: int | None = None
    expected_completion_days: int | None = None
    enrollment_cap: int | None = None
    # Teacher contact info surfaced on the student landing page. Empty string
    # is normalised to None so clearing a field in the form actually blanks the
    # column rather than storing "".
    contact_email: str | None = Field(default=None, max_length=320)
    contact_phone: str | None = Field(default=None, max_length=50)
    contact_website_url: str | None = Field(default=None, max_length=500)
    contact_social_url: str | None = Field(default=None, max_length=500)

    @field_validator(
        "contact_email",
        "contact_phone",
        "contact_website_url",
        "contact_social_url",
    )
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class ModuleCreate(_StrictRequest):
    course_id: UUID
    title: str = Field(max_length=255)
    description: str | None = None
    position: int = Field(ge=0)
    estimated_minutes: int | None = None
    requires_all_lessons_unlocked: bool = False


class ModuleUpdate(_StrictRequest):
    title: str | None = Field(default=None, max_length=255)
    description: str | None = None
    position: int | None = Field(default=None, ge=0)
    status: Literal["draft", "published", "archived"] | None = None
    estimated_minutes: int | None = None
    requires_all_lessons_unlocked: bool | None = None


def _check_ef_min_unlock(value: float) -> float:
    if not (1.3 <= value <= 2.5):
        raise ValueError("ef_min_unlock must be in [1.3, 2.5]")
    return value


def _check_tau_unlock(value: float) -> float:
    if not (0.0 < value <= 1.0):
        raise ValueError("tau_unlock must be in (0.0, 1.0]")
    return value


class LessonCreate(_StrictRequest):
    """Create a new lesson under ``module_id``.

    Per Reconciliation §A5, the service layer auto-creates the
    accompanying :class:`~abridgeai.features.courses.models.ModuleItem`
    row that links this lesson into the module's ordering — callers do
    NOT pass a ``ModuleItem`` payload.
    """

    module_id: UUID
    slug: str = Field(max_length=100)
    title: str = Field(max_length=255)
    summary: str | None = None
    notes_markdown: str | None = None
    primary_material_id: UUID | None = None
    lesson_type: Literal["video", "reading"] = "video"
    difficulty: str | None = Field(default=None, max_length=20)
    estimated_minutes: int | None = None
    ef_min_unlock: float = 2.0
    tau_unlock: float = 0.8
    requires_interview_pass: bool = False
    unlock_rule_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ef_min_unlock")
    @classmethod
    def _validate_ef_min(cls, value: float) -> float:
        return _check_ef_min_unlock(value)

    @field_validator("tau_unlock")
    @classmethod
    def _validate_tau(cls, value: float) -> float:
        return _check_tau_unlock(value)


class LessonUpdate(_StrictRequest):
    """Partial update for a lesson — mirrors :class:`LessonCreate` (all optional)."""

    slug: str | None = Field(default=None, max_length=100)
    title: str | None = Field(default=None, max_length=255)
    summary: str | None = None
    notes_markdown: str | None = None
    primary_material_id: UUID | None = None
    lesson_type: Literal["video", "reading"] | None = None
    difficulty: str | None = Field(default=None, max_length=20)
    estimated_minutes: int | None = None
    status: Literal["draft", "published", "archived"] | None = None
    ef_min_unlock: float | None = None
    tau_unlock: float | None = None
    requires_interview_pass: bool | None = None
    unlock_rule_json: dict[str, Any] | None = None

    @field_validator("ef_min_unlock")
    @classmethod
    def _validate_ef_min(cls, value: float | None) -> float | None:
        return None if value is None else _check_ef_min_unlock(value)

    @field_validator("tau_unlock")
    @classmethod
    def _validate_tau(cls, value: float | None) -> float | None:
        return None if value is None else _check_tau_unlock(value)


class LessonResourceCreate(_StrictRequest):
    lesson_id: UUID
    title: str = Field(max_length=255)
    resource_type: Literal["pdf", "zip", "mp4", "xlsx", "pptx", "docx", "link", "other"]
    storage_object_id: UUID
    position: int = Field(ge=0)
    visible_to_students: bool = True


class LessonResourceUpdate(_StrictRequest):
    title: str | None = Field(default=None, max_length=255)
    resource_type: Literal["pdf", "zip", "mp4", "xlsx", "pptx", "docx", "link", "other"] | None = (
        None
    )
    storage_object_id: UUID | None = None
    position: int | None = Field(default=None, ge=0)
    visible_to_students: bool | None = None


class ModulePrerequisiteSet(_StrictRequest):
    """Replace the full list of prerequisites for a module (idempotent)."""

    module_id: UUID
    prerequisite_module_ids: list[UUID]


class ModuleItemReorder(_StrictRequest):
    """Reorder ModuleItem rows under a single module.

    ``new_order`` is the full list of ``ModuleItem.id`` values in the
    target order — the service applies the
    ``_OFFSET=100_000`` two-phase swap pattern (see locked decision
    "Architectural flows to PRESERVE") to avoid violating the
    ``uq_module_items_position`` unique constraint mid-update.
    """

    module_id: UUID
    new_order: list[UUID]


class ModuleReorder(_StrictRequest):
    """Reorder Module rows under a single course.

    ``new_order`` is the full list of ``Module.id`` values in the target
    order — the service applies the ``_OFFSET=100_000`` two-phase swap
    pattern to avoid violating the ``uq_modules_course_position`` unique
    constraint mid-update (mirrors :class:`ModuleItemReorder`).
    """

    course_id: UUID
    new_order: list[UUID]


class ModuleItemUpdate(_StrictRequest):
    """Patch payload for ``PATCH /teacher/module-items/{item_id}``.

    Only the unstructured ``unlock_rule_json`` carrier is mutable; the
    polymorphic identity (lesson_id / quiz_id / interview_config_id),
    item_type, and position are managed via reorder + create / delete
    flows so a teacher cannot use PATCH to violate the XOR check or
    the position UNIQUE.
    """

    unlock_rule_json: dict[str, Any] | None = None


class CoursePublishRequest(_StrictRequest):
    """Explicit confirmation gate before transitioning a course to published."""

    confirm: bool = False


class CourseArchiveRequest(_StrictRequest):
    """Explicit confirmation gate before archiving a course."""

    confirm: bool = False


__all__ = [
    "CourseArchiveRequest",
    "CourseCreate",
    "CoursePublishRequest",
    "CourseUpdate",
    "LessonCreate",
    "LessonResourceCreate",
    "LessonResourceUpdate",
    "LessonUnlockConfig",
    "LessonUpdate",
    "ModuleCreate",
    "ModuleItemReorder",
    "ModuleItemUpdate",
    "ModulePrerequisiteSet",
    "ModuleUpdate",
]
