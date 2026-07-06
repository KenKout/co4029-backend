"""Unit tests for visibility clause builders (T3.3).

Each test compiles the clause to a literal Postgres SQL string via
``str(clause.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))``
and asserts the generated SQL contains the expected fragments. No DB is
involved — these are pure SQL builders.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from abridgeai.features.courses.models import Course, Lesson
from abridgeai.features.courses.visibility import (
    module_item_visible_clause,
    published_course_clause,
    published_interview_clause,
    published_lesson_clause,
    published_material_clause,
    published_module_clause,
    published_quiz_clause,
    student_visible_resource_clause,
)


def _compile(clause: object) -> str:
    return str(
        clause.compile(  # type: ignore[attr-defined]
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_published_course_clause_compiles_to_correct_sql() -> None:
    sql = _compile(published_course_clause())
    assert "courses.status" in sql
    assert "'published'" in sql


def test_published_module_clause_compiles_to_correct_sql() -> None:
    sql = _compile(published_module_clause())
    assert "modules.status" in sql
    assert "'published'" in sql


def test_published_lesson_clause_compiles_to_correct_sql() -> None:
    sql = _compile(published_lesson_clause())
    assert "lessons.status" in sql
    assert "'published'" in sql


def test_student_visible_resource_clause_compiles_to_correct_sql() -> None:
    sql = _compile(student_visible_resource_clause())
    assert "lesson_resources.visible_to_students" in sql
    assert "true" in sql.lower()


def test_published_quiz_clause_compiles_to_correct_sql() -> None:
    sql = _compile(published_quiz_clause())
    assert "quizzes.status" in sql
    assert "'published'" in sql


def test_published_interview_clause_compiles_to_correct_sql() -> None:
    sql = _compile(published_interview_clause())
    assert "interview_configs.status" in sql
    assert "'published'" in sql


def test_published_material_clause_compiles_to_correct_sql() -> None:
    sql = _compile(published_material_clause())
    assert "learning_materials.visible_to_students" in sql
    assert "true" in sql.lower()
    # Current-version readiness gate: correlated EXISTS subquery.
    assert "EXISTS" in sql
    assert "learning_material_versions.is_current" in sql
    assert "learning_material_versions.processing_status" in sql
    assert "'ready'" in sql


def test_module_item_visible_clause_polymorphic() -> None:
    sql = _compile(module_item_visible_clause()).lower()
    assert "case" in sql
    assert "module_items.item_type" in sql
    assert "'lesson'" in sql
    assert "'quiz'" in sql
    assert "'interview'" in sql
    assert "lessons.status" in sql
    assert "quizzes.status" in sql
    assert "interview_configs.status" in sql
    assert "false" in sql


def test_clauses_are_pure_sql_no_db_call() -> None:
    """Each builder returns a SQL ColumnElement without DB I/O."""
    from sqlalchemy.sql.elements import ColumnElement

    for builder in (
        published_course_clause,
        published_module_clause,
        published_lesson_clause,
        published_quiz_clause,
        published_interview_clause,
        published_material_clause,
        student_visible_resource_clause,
        module_item_visible_clause,
    ):
        result = builder()
        assert isinstance(result, ColumnElement)


def test_clauses_compose_in_select_where() -> None:
    stmt = select(Course).where(published_course_clause())
    sql = _compile(stmt).lower()
    assert "from courses" in sql
    assert "where courses.status = 'published'" in sql


def test_module_item_clause_composes_with_lesson_join() -> None:
    """Smoke: the polymorphic CASE composes cleanly into a join + WHERE."""
    from abridgeai.features.courses.models import ModuleItem

    stmt = (
        select(ModuleItem)
        .join(Lesson, ModuleItem.lesson_id == Lesson.id, isouter=True)
        .where(module_item_visible_clause())
    )
    sql = _compile(stmt).lower()
    assert "left outer join lessons" in sql
    assert "case" in sql
    assert "module_items.item_type = 'lesson'" in sql
