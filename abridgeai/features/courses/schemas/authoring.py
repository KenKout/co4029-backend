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
from typing import Annotated, Any, Literal
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
    ``question_count`` is how many live quiz questions currently map to
    this outcome (via ``quiz_questions.learning_outcome_id``) — surfaced
    so the delete confirmation can say exactly what loses its mapping.
    """

    course_id: UUID
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None
    question_count: int = 0


class CourseLearningOutcomeCreate(BaseModel):
    """Body for ``POST /teacher/courses/{course_id}/outcomes`` (§LO-1).

    ``outcome_text`` is required. ``parent_id`` is optional — omit (or
    null) for a top-level outcome, or pass an existing outcome's id to
    nest this one beneath it (arbitrary depth). ``position`` is assigned
    server-side (append at the end of the parent's children); the dotted
    ``L.O.x.y`` code is derived from the parent chain at display time and
    is never stored.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_text: Annotated[str, Field(min_length=1, max_length=1000)]
    parent_id: UUID | None = None


class CourseLearningOutcomeUpdate(BaseModel):
    """Body for ``PATCH /teacher/courses/{course_id}/outcomes/{outcome_id}``.

    ``outcome_text`` edits the statement. ``parent_id`` re-parents the
    outcome (move within the tree); pass null to promote it to top-level.
    ``position`` reorders it among its (possibly new) siblings — 1-based,
    the slot the outcome should occupy after the move, so the outliner can
    express "drop between rows 3 and 4" as ``position=4``.

    The dotted ``L.O.x.y`` code is display-only: derived from the parent
    chain and sibling positions at read time, never stored. All internal
    references key on the stable UUID ``id``, so reorders and re-parents
    change what a code *renders as* without ever changing identity — an
    external syllabus quoting "L.O.2" points at the outcome's id, not at
    a persisted code string.

    ``parent_id`` and ``position`` use sentinels so "omitted" (leave
    unchanged) is distinguishable from "explicit null" (promote to
    top-level / append at the end): ``model_fields_set`` is consulted in
    the service layer. Re-parenting is cycle-checked server-side (an
    outcome may not become its own descendant).
    """

    model_config = ConfigDict(extra="forbid")

    outcome_text: Annotated[str | None, Field(min_length=1, max_length=1000)] = None
    parent_id: UUID | None = None
    position: Annotated[int | None, Field(ge=1)] = None


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
    # Short-TTL presigned GET URL for the course thumbnail, minted by the
    # service layer from the thumbnail's storage object. None when no
    # thumbnail is set (the SPA falls back to the gradient banner). Not
    # persisted — a projection.
    thumbnail_url: str | None = None
    estimated_minutes: int | None = None
    expected_completion_days: int | None = None
    enrollment_cap: int | None = None
    # Course-health projections computed by the service layer for the "My
    # courses" grid — active enrollments and non-deleted module count. Not
    # persisted; default 0 so single-course reads that don't populate them
    # still validate.
    student_count: int = 0
    module_count: int = 0
    instructor: InstructorAuthoring | None = None
    tags: list[TagAuthoring] = []  # type: ignore[assignment]
    outcomes: list[CourseLearningOutcomeAuthoring] = []  # type: ignore[assignment]
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class TeacherDashboardStats(BaseModel):
    """Actionable counts for the teacher dashboard's clickable widgets.

    Scoped to every course the caller can author. ``draft_courses`` =
    courses in draft status; ``ungraded_quizzes`` = submitted-but-ungraded
    quiz attempts; ``pending_interviews`` = interview sessions awaiting
    evaluation. All are \"needs attention\" signals that deep-link into the
    relevant page.

    The review-queue block surfaces the human-in-the-loop backlog for
    AI-generated content, and the retention block summarises the spaced-
    repetition health of the caller's students. Every field is an aggregate
    over the same authorable course-id set, batched in the query layer.
    """

    model_config = ConfigDict(extra="forbid")

    draft_courses: int = 0
    ungraded_quizzes: int = 0
    pending_interviews: int = 0
    #: ``{course_id: pending_review_count}`` for AI-generated quiz + interview
    #: questions still awaiting review. Courses with nothing pending are omitted,
    #: so consumers treat a missing key as zero. Drives the pending-review dot on
    #: the dashboard's course cards — the totals say work exists, this says where.
    pending_review_by_course: dict[UUID, int] = Field(default_factory=dict)

    # --- Human-in-the-loop review queue -------------------------------
    # Quiz questions still at review_status='pending' (AI-generated cards
    # awaiting teacher approval).
    quiz_cards_pending_review: int = 0
    # Interview questions still at review_status='pending'.
    interview_questions_pending_review: int = 0
    # Published quizzes that carry APPROVED questions with no usable
    # expected_response_time_ms (NULL or <= 0). SM-2 grading needs t_exp,
    # so these are published-but-uncalibrated.
    published_quizzes_missing_texp: int = 0
    # Material versions whose ingestion pipeline completed but whose lesson
    # has not been used to generate a quiz yet — ready for quiz generation.
    materials_ready_for_quiz_gen: int = 0

    # --- Spaced-repetition retention signal ---------------------------
    # Students whose AVERAGE easiness factor across in-scope cards is below
    # the 2.0 struggling threshold.
    students_below_ef_threshold: int = 0
    # Mean easiness factor across in-scope cards, rounded to 2dp. 0.0 when
    # the caller's courses have no card state yet.
    avg_retention_ef: float = 0.0
    # In-scope cards whose due_at is already in the past.
    cards_overdue: int = 0


class ReviewQueueItem(BaseModel):
    """One navigable group inside a review-queue category."""

    course_id: UUID
    course_title: str
    module_id: UUID | None = None
    module_title: str | None = None
    target_id: UUID
    target_title: str
    count: int


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


class SlugAvailability(BaseModel):
    available: bool


class OutlineSection(BaseModel):
    """One section row in ``GET /lessons/{id}/outline``.

    Mirrors the SPA's ``OutlineSectionRead`` interface verbatim.

    The ``id`` is a deterministic slug like ``sec_<lesson8>_<title-slug>_<page>``
    (see :func:`abridgeai.features.quizzes.ai.outline._section_id`), NOT a
    database UUID. Stable across re-runs against the same chunks so
    ``coverage_options.section_ids`` keeps working between calls.
    """

    id: str
    title: str
    depth: int = 0
    chunk_count: int = 0
    char_count: int = 0
    page_range: tuple[int, int] = (0, 0)
    content_role: Literal["body", "summary", "review", "front_matter"] = "body"
    preview: str = ""


class LessonOutline(BaseModel):
    lesson_id: UUID
    lesson_title: str
    sections: list[OutlineSection]
    suggested_question_count: int = 0
    min_for_full_coverage: int = 0


class StreamUrlResponse(BaseModel):
    """Response shape for ``GET /teacher/lesson-resources/{id}/download-url``.

    Field name ``stream_url`` (not ``url``) matches the SPA's
    ``StreamUrlResponse`` TypeScript interface so the existing
    ``fetchTeacherResourceDownloadUrl`` helper works unchanged.
    """

    stream_url: str
    expires_at: datetime


__all__ = [
    "CourseAuthoring",
    "CourseContentAuthoring",
    "CourseLearningOutcomeAuthoring",
    "InstructorAuthoring",
    "LessonAuthoring",
    "LessonOutline",
    "LessonResourceAuthoring",
    "ModuleAuthoring",
    "ModuleItemAuthoring",
    "OutlineSection",
    "SlugAvailability",
    "StreamUrlResponse",
    "TagAuthoring",
]
