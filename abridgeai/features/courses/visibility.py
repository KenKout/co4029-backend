"""Visibility clause builders for the courses aggregate.

These are **pure SQL clause fragments** (SQLAlchemy ``ColumnElement[bool]``)
intended to be composed into ``Select`` statements by the queries layer
(T3.4, ``features/courses/queries/published.py``). They are not Python
predicates: every function returns an unevaluated SQL expression that
references the mapped columns of the courses aggregate (Course / Module
/ Lesson / LessonResource / ModuleItem from T3.1).

Design notes
------------

* **Pure SQL, no Python booleans.** Per plan §4068 / §3768. Each builder
  returns a ``ColumnElement[bool]``; tests assert the compiled SQL
  shape, never call DB.
* **Audience-agnostic naming.** Per plan §4070, names are
  ``published_*`` rather than ``student_visible_*``. The single
  exception, :func:`student_visible_resource_clause`, is named that
  way because the underlying column is literally
  ``LessonResource.visible_to_students`` — the predicate is
  column-level, not role-level.
* **Soft-delete is orthogonal.** ``deleted_at IS NULL`` filtering is
  injected globally by the T0.7 ``with_loader_criteria`` chain on every
  ``Mapped`` class that uses :class:`SoftDeleteMixin`. These visibility
  clauses **add** to that — they're status-based publish gates, not
  deletion filters.
* **Forward references for Phase 4/5/6 entities.** Quizzes, interview
  configs, and learning-material visibility live in features that
  T3.3 cannot import (would break the import-linter contracts and
  introduce a circular ref). Their clause builders raise
  :class:`NotImplementedError` until Phase 4/5/6 lands the relevant
  models. T3.4 must not call them yet.

Composition example (used by T3.4)
----------------------------------

>>> from sqlalchemy import select
>>> stmt = select(Course).where(published_course_clause())
>>> # ... or for a content-tree join:
>>> stmt = (
...     select(ModuleItem, Lesson)
...     .join(Lesson, ModuleItem.lesson_id == Lesson.id, isouter=True)
...     .where(module_item_visible_clause())
... )
"""

from __future__ import annotations

from sqlalchemy import false
from sqlalchemy.sql import case
from sqlalchemy.sql.elements import ColumnElement

from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
)

__all__ = [
    "module_item_visible_clause",
    "published_course_clause",
    "published_interview_clause",
    "published_lesson_clause",
    "published_material_clause",
    "published_module_clause",
    "published_quiz_clause",
    "student_visible_resource_clause",
]


def published_course_clause() -> ColumnElement[bool]:
    """SQL: ``courses.status = 'published'``.

    Assumes ``Course`` is in the ``FROM`` clause of the composing query.
    """
    return Course.status == "published"


def published_module_clause() -> ColumnElement[bool]:
    """SQL: ``modules.status = 'published'``.

    Assumes ``Module`` is in the ``FROM`` clause of the composing query.
    """
    return Module.status == "published"


def published_lesson_clause() -> ColumnElement[bool]:
    """SQL: ``lessons.status = 'published'``.

    Assumes ``Lesson`` is in the ``FROM`` clause of the composing query.
    """
    return Lesson.status == "published"


def student_visible_resource_clause() -> ColumnElement[bool]:
    """SQL: ``lesson_resources.visible_to_students IS TRUE``.

    Named after the underlying column (T3.1 baseline column on
    :class:`LessonResource`), not after a role-based predicate.
    """
    return LessonResource.visible_to_students.is_(True)


def published_quiz_clause() -> ColumnElement[bool]:
    """SQL: ``quizzes.status = 'published'``.

    Assumes :class:`Quiz` is in the ``FROM`` clause of the composing
    query. Phase-5 implementation lit; quiz items now flow through
    student-facing content trees on the same predicate as lessons.
    """
    from abridgeai.features.quizzes.models import Quiz

    return Quiz.status == "published"


def published_interview_clause() -> ColumnElement[bool]:
    """Forward reference — raises until Phase 6 lands ``features.interviews.models``.

    The Phase-6 implementation will return
    ``InterviewConfig.status == 'published'``.
    """
    raise NotImplementedError(
        "published_interview_clause: features.interviews.models pending Phase 6 (T6.x)"
    )


def published_material_clause() -> ColumnElement[bool]:
    """Forward reference — raises until Phase 4 lands ``features.materials.models``.

    The Phase-4 implementation will return
    ``LearningMaterial.visible_to_students.is_(True)`` (per plan §4062).
    """
    raise NotImplementedError(
        "published_material_clause: features.materials.models pending Phase 4 (T4.x)"
    )


def module_item_visible_clause() -> ColumnElement[bool]:
    """Polymorphic visibility predicate for :class:`ModuleItem`.

    Branches on ``module_items.item_type``:

    * ``'lesson'`` → ``lessons.status = 'published'`` (the composing
      query MUST join :class:`Lesson` for this branch to make sense; if
      the join is missing the predicate degrades to NULL on those rows
      which Postgres treats as falsy in ``WHERE``).
    * ``'quiz'`` → ``false()`` placeholder (Phase 5 swaps in the real
      ``Quiz.status = 'published'`` predicate).
    * ``'interview'`` → ``false()`` placeholder (Phase 6 swaps in
      ``InterviewConfig.status = 'published'``).
    * else → ``false()`` (defensive — the
      ``ck_module_items_item_type_enum`` CHECK constraint already
      restricts the column to those three values).

    Until Phase 5/6 land, only lesson-typed module items are visible to
    student-facing queries — quiz / interview items appear in authoring
    queries (which don't apply this clause) but are filtered out of
    published content trees.
    """
    return case(
        (ModuleItem.item_type == "lesson", Lesson.status == "published"),
        (ModuleItem.item_type == "quiz", published_quiz_clause()),
        (ModuleItem.item_type == "interview", false()),
        else_=false(),
    )
