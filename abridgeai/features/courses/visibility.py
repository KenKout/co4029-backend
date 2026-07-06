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
* **Lazy imports for cross-feature entities.** Quizzes, interview
  configs, and learning materials live in features that this module
  cannot import at load time (would break the import-linter contracts
  and introduce a circular ref). Their clause builders import the
  mapped class inside the function body — the composing query must
  join the corresponding table for the predicate to bind.

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

from sqlalchemy import and_, false, select
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
    """SQL: ``interview_configs.status = 'published'``.

    Assumes :class:`InterviewConfig` is in the ``FROM`` clause of the
    composing query (outer-joined on ``module_items.interview_config_id``);
    without the join the predicate degrades to NULL → falsy in ``WHERE``.
    """
    from abridgeai.features.interviews.models import InterviewConfig

    return InterviewConfig.status == "published"


def published_material_clause() -> ColumnElement[bool]:
    """SQL: material is student-visible AND its current version is ready.

    ``learning_materials.visible_to_students IS TRUE`` (plan §4062) AND
    EXISTS a ``learning_material_versions`` row that is both
    ``is_current`` and ``processing_status = 'ready'`` — learners must
    never see a material whose ingestion pipeline has not completed.

    Assumes :class:`LearningMaterial` is in the ``FROM`` clause of the
    composing query; the version check is a correlated EXISTS subquery.
    """
    from abridgeai.features.materials.models import (
        LearningMaterial,
        LearningMaterialVersion,
    )

    ready_current_version = (
        select(LearningMaterialVersion.id)
        .where(
            LearningMaterialVersion.material_id == LearningMaterial.id,
            LearningMaterialVersion.is_current.is_(True),
            LearningMaterialVersion.processing_status == "ready",
        )
        .exists()
    )
    return and_(
        LearningMaterial.visible_to_students.is_(True),
        ready_current_version,
    )


def module_item_visible_clause() -> ColumnElement[bool]:
    """Polymorphic visibility predicate for :class:`ModuleItem`.

    Branches on ``module_items.item_type``:

    * ``'lesson'`` → ``lessons.status = 'published'`` (the composing
      query MUST join :class:`Lesson` for this branch to make sense; if
      the join is missing the predicate degrades to NULL on those rows
      which Postgres treats as falsy in ``WHERE``).
    * ``'quiz'`` → ``quizzes.status = 'published'`` (requires a
      :class:`Quiz` join on ``module_items.quiz_id``).
    * ``'interview'`` → ``interview_configs.status = 'published'``
      (requires an :class:`InterviewConfig` join on
      ``module_items.interview_config_id``).
    * else → ``false()`` (defensive — the
      ``ck_module_items_item_type_enum`` CHECK constraint already
      restricts the column to those three values).
    """
    return case(
        (ModuleItem.item_type == "lesson", Lesson.status == "published"),
        (ModuleItem.item_type == "quiz", published_quiz_clause()),
        (ModuleItem.item_type == "interview", published_interview_clause()),
        else_=false(),
    )
