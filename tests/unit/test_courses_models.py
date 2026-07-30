"""Unit tests for the courses aggregate ORM models (T3.1).

Covers Reconciliation §A1-§A13 invariants:
* All 8 ORM models + 2 self-edge association tables import cleanly.
* Soft-deletable models carry the 7 audit columns supplied by the
  ``UUIDPrimaryKeyMixin + TimestampMixin + AuditedByMixin +
  SoftDeleteMixin`` stack.
* :class:`Course` declares ``owner_user_id`` as a column distinct from
  ``created_by`` per the locked "Naming convention" decision.
* Status enums on :class:`Course`, :class:`Module`, :class:`Lesson`
  carry the canonical ``{draft, published, archived}`` CHECK matching
  baseline DDL.
* Lesson unlock-config columns (``ef_min_unlock``, ``tau_unlock``,
  ``requires_interview_pass``, ``unlock_rule_json``) are present.
* :class:`Module.requires_all_lessons_unlocked` defaults to FALSE
  (loose semantics per §3900).
* :class:`ModuleItem` keeps the canonical
  ``lesson_id``/``quiz_id``/``interview_config_id`` polymorphism per
  §A3 — NOT the hypothetical ``target_*`` variants.
* :class:`Tag` and :class:`CourseTag` follow baseline DDL exactly
  (Tag has no ``organization_id``; CourseTag has no ``created_by``).
* ``CareerCourseItem`` is intentionally NOT importable from
  ``features.courses.models`` (deferred to ``features/career_paths/``
  in Phase 7 / T7.3).
"""

from __future__ import annotations

import importlib

import pytest

from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    CourseTag,
    Lesson,
    LessonPrerequisite,
    LessonResource,
    Module,
    ModuleItem,
    ModulePrerequisite,
    Tag,
)


def test_courses_models_importable() -> None:
    models = [
        Course,
        Module,
        Lesson,
        ModuleItem,
        LessonResource,
        Tag,
        CourseTag,
        CourseLearningOutcome,
    ]
    assert len(models) == 8
    table_names = {m.__tablename__ for m in models}
    assert table_names == {
        "courses",
        "modules",
        "lessons",
        "module_items",
        "lesson_resources",
        "tags",
        "course_tags",
        "course_learning_outcomes",
    }
    assert ModulePrerequisite.__tablename__ == "module_prerequisites"
    assert LessonPrerequisite.__tablename__ == "lesson_prerequisites"


@pytest.mark.parametrize(
    "model",
    [Course, Module, Lesson, ModuleItem, LessonResource, CourseLearningOutcome],
    ids=lambda m: m.__name__,
)
def test_audit_columns_present(model: type) -> None:
    cols = {c.name for c in model.__table__.columns}
    expected = {
        "id",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
    assert expected.issubset(cols), f"{model.__name__} missing audit columns: {expected - cols}"


def test_course_owner_field_separate_from_audit() -> None:
    cols = {c.name for c in Course.__table__.columns}
    assert "owner_user_id" in cols
    assert "created_by" in cols
    assert "updated_by" in cols
    owner_col = Course.__table__.c.owner_user_id
    assert owner_col.nullable is False


def _check_constraint_text(model: type, name: str) -> str:
    for constraint in model.__table__.constraints:
        if getattr(constraint, "name", None) == name:
            return str(constraint.sqltext)
    raise AssertionError(f"{model.__name__} missing CHECK constraint {name}")


@pytest.mark.parametrize(
    ("model", "constraint_name"),
    [
        (Course, "ck_courses_status"),
        (Module, "ck_modules_status"),
        (Lesson, "ck_lessons_status"),
    ],
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_status_enum_values(model: type, constraint_name: str) -> None:
    sqltext = _check_constraint_text(model, constraint_name)
    assert "'draft'" in sqltext
    assert "'published'" in sqltext
    assert "'archived'" in sqltext


def test_lesson_unlock_config_columns_present() -> None:
    cols = {c.name for c in Lesson.__table__.columns}
    assert "ef_min_unlock" in cols
    assert "tau_unlock" in cols
    assert "requires_interview_pass" in cols
    assert "unlock_rule_json" in cols, "Lesson.unlock_rule_json must be kept per §A4"


def test_lesson_unlock_config_check_constraints_present() -> None:
    ef_text = _check_constraint_text(Lesson, "ck_lessons_ef_min_unlock_range")
    assert "1.3" in ef_text
    assert "2.5" in ef_text
    tau_text = _check_constraint_text(Lesson, "ck_lessons_tau_unlock_range")
    assert "0.0" in tau_text
    assert "1.0" in tau_text


def test_module_requires_all_lessons_unlocked_default_false() -> None:
    cols = {c.name: c for c in Module.__table__.columns}
    col = cols["requires_all_lessons_unlocked"]
    assert col.nullable is False
    server_default_text = str(col.server_default.arg).upper() if col.server_default else ""
    assert "FALSE" in server_default_text, (
        "requires_all_lessons_unlocked must default FALSE per §3900 (loose semantics)"
    )


def test_module_item_polymorphism_keeps_canonical_field_names() -> None:
    cols = {c.name for c in ModuleItem.__table__.columns}
    assert "lesson_id" in cols, "§A3: keep lesson_id, NOT target_lesson_id"
    assert "quiz_id" in cols, "§A3: keep quiz_id, NOT target_quiz_id"
    assert "interview_config_id" in cols, "§A3: keep interview_config_id, NOT target_interview_id"
    assert "target_lesson_id" not in cols
    assert "target_quiz_id" not in cols
    assert "target_interview_id" not in cols


def test_module_item_xor_check_present() -> None:
    sqltext = _check_constraint_text(ModuleItem, "ck_module_items_item_type")
    assert "lesson_id IS NOT NULL" in sqltext
    assert "quiz_id IS NOT NULL" in sqltext
    assert "interview_config_id IS NOT NULL" in sqltext


def test_course_slug_no_unique_kwarg() -> None:
    slug_col = Course.__table__.c.slug
    assert slug_col.unique is None or slug_col.unique is False, (
        "§A11: Course.slug uniqueness lives in T0.16 partial index, not unique=True"
    )


def test_tag_minimal_shape_per_baseline() -> None:
    cols = {c.name for c in Tag.__table__.columns}
    assert cols == {"id", "slug", "name", "created_at", "updated_at"}, (
        "Tag should match baseline DDL exactly: id/slug/name + timestamps"
    )


def test_course_tag_minimal_shape_per_baseline() -> None:
    cols = {c.name for c in CourseTag.__table__.columns}
    assert cols == {"course_id", "tag_id", "created_at"}, (
        "CourseTag should match baseline DDL exactly: composite PK + created_at only"
    )


def test_course_learning_outcome_position_uniqueness_is_db_partial_index() -> None:
    # Migration 0059 (LO hierarchy) replaced the course-wide
    # ``uq_course_learning_outcomes_position`` UNIQUE constraint with a partial
    # expression index over COALESCE(parent_id, course_id) so uniqueness is
    # per-parent (siblings), not per-course. That is NOT expressible as a simple
    # ``UniqueConstraint`` in ``__table_args__``, so the model intentionally
    # declares none — the guarantee lives in the migration DDL. Assert the model
    # reflects that: no course-wide position UNIQUE constraint remains, and the
    # self-referential ``parent_id`` is indexed (the hierarchy access path).
    constraint_names = {
        getattr(c, "name", None) for c in CourseLearningOutcome.__table__.constraints
    }
    assert "uq_course_learning_outcomes_position" not in constraint_names

    parent_id_col = CourseLearningOutcome.__table__.c.parent_id
    assert parent_id_col.index is True


def test_career_course_item_not_in_courses_models() -> None:
    module = importlib.import_module("abridgeai.features.courses.models")
    with pytest.raises(AttributeError):
        _ = module.CareerCourseItem
    with pytest.raises(ImportError):
        from abridgeai.features.courses.models import CareerCourseItem  # noqa: F401


def test_enrollment_not_in_courses_models() -> None:
    module = importlib.import_module("abridgeai.features.courses.models")
    with pytest.raises(AttributeError):
        _ = module.Enrollment
    with pytest.raises(AttributeError):
        _ = module.CourseEnrollment
    with pytest.raises(AttributeError):
        _ = module.InvitationCode
