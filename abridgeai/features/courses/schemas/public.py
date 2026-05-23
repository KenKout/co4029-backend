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


class TagPublic(_ORMModel):
    """Public projection of :class:`~abridgeai.features.courses.models.Tag`.

    Baseline DDL only carries ``slug`` and ``name`` (§A13) — no
    ``display_name``, no ``organization_id``. We expose ``name`` as-is.
    """

    id: UUID
    slug: str
    name: str


class CourseLearningOutcomePublic(_ORMModel):
    """Public projection of a course learning outcome (§A12)."""

    id: UUID
    position: int
    outcome_text: str


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
    status: Literal["published"]
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
    """

    id: UUID
    title: str


class QuizSummaryPublic(_ORMModel):
    """Slim quiz projection embedded inside :class:`ModuleItemPublic.target`.

    Mirrors :class:`abridgeai.features.quizzes.schemas.public.QuizPublic`
    on the wire (``id`` + ``title``) but lives in the courses schema
    package so the cross-feature import-linter contract stays intact.
    The field surface intentionally matches :class:`LessonPublic` so
    frontend code can address ``item.target.{id,title}`` polymorphically.
    """

    id: UUID
    title: str


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
    * Phase 6 will add ``InterviewPublic``.

    Pydantic v2 discriminates dict-shape on a best-fit basis; both
    branches share ``{id, title}`` so tagging is unambiguous.
    """

    id: UUID
    module_id: UUID
    item_type: Literal["lesson", "quiz", "interview"]
    position: int
    target: LessonPublic | QuizSummaryPublic | None = None


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
    "LessonPublic",
    "LessonResourcePublic",
    "ModuleItemPublic",
    "ModulePublic",
    "QuizSummaryPublic",
    "ResourceDownloadUrlResponse",
    "TagPublic",
]
