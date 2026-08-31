"""Student-facing (public) DTOs for the courses aggregate.

These schemas back ``routers/learner.py`` (T3.6) and represent the
narrowest projection of the courses ORM models — no internal notes,
no audit columns, no ``draft_notes``. Authoring schemas (``authoring.py``)
inherit from these and widen as needed for ``routers/authoring.py``.

Reconciliation directives (HIGHER PRECEDENCE — see plan §266-585):

* §A2 — ``Course.published_at`` and ``Course.instructor_summary`` do
  NOT exist as columns on T3.1's :class:`~abridgeai.features.courses.models.Course`,
  so they are intentionally absent from :class:`CoursePublic`. The
  instructor profile is composed via :class:`InstructorRead` joined
  from ``UserProfile`` keyed off ``Course.owner_user_id`` (the service
  layer T3.5 performs the join). Likewise ``Lesson.position`` is
  absent — ordering lives on the parent ``ModuleItem.position`` link.
* §A12 — ``CourseLearningOutcome`` has its own (course_id, position)
  ordering and is exposed as a sibling collection on the course payload.
* §A13 — baseline DDL is ground truth. ``Tag`` has ``slug`` + ``name``
  only (no ``organization_id``, no ``display_name``); :class:`TagPublic`
  reflects that.

Status narrowing
----------------
:class:`CoursePublic` declares ``status: Literal["published"]`` — the
public API only ever sees published courses. The authoring variant
widens this to ``Literal["draft", "published", "archived"]`` so the
narrowing is type-safe at the Python level. ``CourseAuthoring`` rebinds
``status`` to the wider ``Literal`` (legal under Pydantic v2 model
inheritance).

Forward references
------------------
:class:`ModuleItemPublic.target` is currently typed as
``LessonPublic | None``. Phase 5 (T5.x) widens it to
``LessonPublic | QuizPublic | InterviewPublic`` once those public
schemas exist; for T3.2 the lesson polymorph is the only one with a
matching schema, and quiz / interview ``ModuleItem`` rows surface
``target = None`` plus the discriminator ``item_type``.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _ORMModel(BaseModel):
    """Shared base — Pydantic v2 ORM-mode equivalent (`from_attributes=True`)."""

    model_config = ConfigDict(from_attributes=True)


class InstructorRead(_ORMModel):
    """Instructor profile composed from ``UserProfile`` (joined via ``owner_user_id``).

    Per Reconciliation §A2, ``Course`` does NOT carry an
    ``instructor_summary`` column — the public payload composes this
    object from the ``UserProfile`` row of the course owner instead.
    """

    user_id: UUID
    display_name: str
    avatar_url: str | None = None
    headline: str | None = None
    # Course-scoped title flags (Course Instructor / Teacher Assistant),
    # surfaced so the student page can label the instructors up front and the
    # TAs behind. Both true = one teacher holding both titles (user decision
    # 2026-08-30). Also set on the instructor block (`CoursePublic.instructor`
    # is the longest-serving CI).
    is_instructor: bool = False
    is_assistant: bool = False


class TagPublic(_ORMModel):
    """Public projection of :class:`~abridgeai.features.courses.models.Tag`.

    Baseline DDL only carries ``slug`` and ``name`` (§A13) — no
    ``display_name``, no ``organization_id``. We expose ``name`` as-is.
    """

    id: UUID
    slug: str
    name: str


class CourseLearningOutcomePublic(_ORMModel):
    """Public projection of a course learning outcome (§A12).

    Hierarchy (arbitrary depth): ``parent_id`` is the self-referential
    parent (NULL = top-level) and ``position`` is the sibling order within
    that parent. ``code`` is the dotted display path (e.g. ``1.2.1`` →
    rendered ``L.O.1.2.1``) derived from the parent chain at read time;
    ``depth`` (0 = top-level) is provided so clients can indent the tree
    without recomputing the chain. Both are projection-only — filled by the
    service layer, not stored — so they default when validating a bare ORM
    row.
    """

    id: UUID
    position: int
    outcome_text: str
    parent_id: UUID | None = None
    code: str | None = None
    depth: int = 0


class CourseCareerPlacementPublic(_ORMModel):
    """One place where this course sits on a career path.

    The student-facing LEVEL is derived from these placements rather than a
    user-defined property: a course's "level" is shown as \"Stage {position} —
    {stage_title}\" (career path: {path_name}). A course may sit on several
    paths; the FE shows the first and flags the rest.
    """

    career_path_id: UUID
    career_path_name: str
    stage_id: UUID
    stage_title: str | None = None
    stage_position: int


class CoursePublic(_ORMModel):
    """Student-facing course summary.

    Fields intentionally absent (per §A2):

    * ``published_at`` — column does not exist on the ORM.
    * ``instructor_summary`` — column does not exist; use
      :attr:`instructor` composed from ``UserProfile``.
    """

    id: UUID
    slug: str
    title: str
    description: str | None = None
    organization_id: UUID
    instructor: InstructorRead | None = None
    # Every teacher on the course, ordered Course Instructor first then Teacher
    # Assistants (see assignment service). Each carries `course_role` so the
    # learner page can show the CI prominently and the TAs behind. Empty (and
    # `instructor` null) when the course has no assigned teachers.
    instructors: list[InstructorRead] = []
    # Where the course sits on career paths — the DERIVED level label is built
    # from these (\"Stage {position} — {stage_title}\"). Empty when the course is
    # on no path (no stage label to show).
    career_paths: list[CourseCareerPlacementPublic] = []
    status: Literal["published"]
    # Difficulty / effort exposed for the landing-page meta line (e.g.
    # "Nam · 10 modules · ~18h"). `estimated_minutes` is a plain Course
    # column surfaced verbatim — the SPA formats it. The level is DERIVED
    # from `career_paths` (the level column was dropped).
    estimated_minutes: int | None = None
    # Short-TTL presigned GET URL for the course thumbnail, minted by the
    # service layer from the thumbnail's storage object. None when no thumbnail
    # is set (the SPA falls back to the gradient banner). Not persisted — a
    # projection.
    thumbnail_url: str | None = None
    # Teacher contact info shown on the landing page. All optional — the SPA
    # renders a contact block only for the fields that are set.
    contact_email: str | None = None
    contact_phone: str | None = None
    contact_website_url: str | None = None
    contact_social_url: str | None = None
    tags: list[TagPublic] = []
    outcomes: list[CourseLearningOutcomePublic] = []


class ModulePublic(_ORMModel):
    id: UUID
    course_id: UUID
    title: str
    position: int
    items: list[ModuleItemPublic] = []


class LessonPublic(_ORMModel):
    """Public projection of a lesson.

    Per §A2, ``position`` is absent here — lesson ordering is determined
    by the parent :class:`ModuleItemPublic.position` field.

    ``lesson_type`` drives the student-side renderer (``video`` → video
    player, ``reading`` → notes / primary-material reader, etc). The
    DDL CHECK constraint ``lessons_lesson_type_check`` keeps the column
    inside the ``video|reading|interactive|interview|quiz`` enum.

    ``summary`` / ``notes_markdown`` / ``primary_material_id`` are
    surfaced because the reader pane needs them to render the lesson
    body — without them the student-side fell back to a video-shaped
    skeleton even for ``reading`` lessons.
    """

    id: UUID
    title: str
    slug: str | None = None
    lesson_type: str
    summary: str | None = None
    notes_markdown: str | None = None
    primary_material_id: UUID | None = None
    estimated_minutes: int | None = None
    difficulty: str | None = None


class QuizSummaryPublic(_ORMModel):
    """Slim quiz projection embedded inside :class:`ModuleItemPublic.target`.

    Mirrors :class:`abridgeai.features.quizzes.schemas.public.QuizPublic`
    on the wire (``id`` + ``title``) but lives in the courses schema
    package so the cross-feature import-linter contract stays intact.
    The field surface intentionally matches :class:`LessonPublic` so
    frontend code can address ``item.target.{id,title}`` polymorphically.

    ``slug`` carries the item's URL slug so the student curriculum tree
    can build breadcrumb links without per-item fetches.
    """

    id: UUID
    title: str
    slug: str


class InterviewSummaryPublic(_ORMModel):
    """Slim interview-config projection embedded inside
    :class:`ModuleItemPublic.target`.

    Mirrors :class:`QuizSummaryPublic` (``id`` + ``title``) so frontend
    code can address ``item.target.{id,title}`` polymorphically; lives
    in the courses schema package so the cross-feature import-linter
    contract stays intact.

    ``slug`` carries the item's URL slug so the student curriculum tree
    can build breadcrumb links without per-item fetches.
    """

    id: UUID
    title: str
    slug: str


class LessonResourcePublic(_ORMModel):
    """Public projection of a lesson resource.

    ``presigned_url`` is populated by the service layer (T3.5) when the
    caller requested resource access; raw ``storage_object_id`` is NOT
    exposed in public payloads.
    """

    id: UUID
    lesson_id: UUID
    title: str
    resource_type: str
    presigned_url: str | None = None


class ModuleItemPublic(_ORMModel):
    """Polymorphic ordering row — public projection.

    :attr:`target` widens with each phase as content types come online:

    * ``LessonPublic`` — Phase 3 (T3.2) baseline.
    * ``QuizSummaryPublic`` — Phase 5 (FR-5); slim shape mirroring
      :class:`abridgeai.features.quizzes.schemas.public.QuizPublic`.
    * ``InterviewSummaryPublic`` — Phase 6 (FR-6); slim interview-config
      shape with the same ``{id, title}`` surface.

    Pydantic v2 discriminates dict-shape on a best-fit basis; the
    branches share ``{id, title}`` so ``item_type`` is the effective
    discriminator for consumers.
    """

    id: UUID
    module_id: UUID
    item_type: Literal["lesson", "quiz", "interview"]
    position: int
    target: LessonPublic | QuizSummaryPublic | InterviewSummaryPublic | None = None


class CourseContentPublic(_ORMModel):
    """Composed course tree (course + modules) returned by the learner router.

    The nested ``items[target]`` graph is populated by the service layer
    (T3.5); this DTO carries the top-level course + module slices.
    """

    course: CoursePublic
    modules: list[ModulePublic] = []


class CoursePage(BaseModel):
    items: list[CoursePublic]
    next_cursor: str | None = None


class ResourceDownloadUrlResponse(BaseModel):
    url: str
    expires_at: str


__all__ = [
    "CourseContentPublic",
    "CourseLearningOutcomePublic",
    "CoursePage",
    "CoursePublic",
    "InstructorRead",
    "InterviewSummaryPublic",
    "LessonPublic",
    "LessonResourcePublic",
    "ModuleItemPublic",
    "ModulePublic",
    "QuizSummaryPublic",
    "ResourceDownloadUrlResponse",
    "TagPublic",
]
