"""Unit tests for the quizzes aggregate ORM models (T5.1).

Covers Reconciliation §C1-§C5 + plan §5400-5418 invariants:
* All 7 ORM models import cleanly.
* Soft-deletable models (``Quiz``, ``QuizQuestion``,
  ``QuizQuestionOption``) carry the 7 audit columns supplied by the
  ``UUIDPrimaryKeyMixin + TimestampMixin + AuditedByMixin +
  SoftDeleteMixin`` stack.
* ``QuizAttempt`` does NOT carry soft-delete cols (plan §5378 — attempts
  are historical record).
* Status enums on ``Quiz`` carry the canonical
  ``{draft, published, archived}`` CHECK matching baseline DDL.
* §C1 renames applied:
  - ``QuizQuestion.expected_response_time_ms`` (was
    ``expected_response_ms``) is Integer + nullable.
  - ``QuizQuestion.source_refs`` (was ``source_refs_json``) is JSONB.
  - ``QuizQuestion.question_type`` CHECK contains ``multiple_choice``
    and NOT ``mcq``.
  - ``QuizQuestion.hint_text`` is Text (NOT String(500)).
  - ``QuizAttemptAnswer.t_actual_ms`` (was ``response_time_ms``)
    present + Integer + nullable.
  - ``QuizAttemptAnswer.hint_used`` is Boolean default FALSE.
* No ``Quiz.mode`` column (plan §5375 + §5381).
* §C4 ``QuizAttempt.idempotency_key`` carries UNIQUE.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Boolean, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB

from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
    QuizSourceLesson,
)


def test_quiz_models_importable() -> None:
    models = [
        Quiz,
        QuizSourceLesson,
        QuizQuestion,
        QuizQuestionOption,
        QuizQuestionRevision,
        QuizAttempt,
        QuizAttemptAnswer,
    ]
    assert len(models) == 7
    table_names = {m.__tablename__ for m in models}
    assert table_names == {
        "quizzes",
        "quiz_source_lessons",
        "quiz_questions",
        "quiz_question_options",
        "quiz_question_revisions",
        "quiz_attempts",
        "quiz_attempt_answers",
    }


@pytest.mark.parametrize(
    "model",
    [Quiz, QuizQuestion, QuizQuestionOption],
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


def test_quiz_attempt_has_no_soft_delete() -> None:
    cols = {c.name for c in QuizAttempt.__table__.columns}
    assert "deleted_at" not in cols, "plan §5378: QuizAttempt is historical record, no soft-delete"
    assert "deleted_by" not in cols


def test_quiz_attempt_answer_has_no_soft_delete() -> None:
    cols = {c.name for c in QuizAttemptAnswer.__table__.columns}
    assert "deleted_at" not in cols
    assert "deleted_by" not in cols


def _check_constraint_text(model: type, name: str) -> str:
    for constraint in model.__table__.constraints:
        if getattr(constraint, "name", None) == name:
            return str(constraint.sqltext)
    raise AssertionError(f"{model.__name__} missing CHECK constraint {name}")


def test_quiz_status_check_constraint() -> None:
    sqltext = _check_constraint_text(Quiz, "ck_quizzes_status")
    assert "'draft'" in sqltext
    assert "'published'" in sqltext
    assert "'archived'" in sqltext


def test_question_type_renamed_from_mcq() -> None:
    sqltext = _check_constraint_text(QuizQuestion, "ck_quiz_questions_question_type")
    assert "multiple_choice" in sqltext, "§C1: question_type CHECK must include 'multiple_choice'"
    assert "'mcq'" not in sqltext, "§C1: legacy 'mcq' must be removed — migration 0007 renames data"
    for value in ("true_false", "short_answer", "fill_blank", "code"):
        assert value in sqltext, f"question_type CHECK missing '{value}'"


def test_expected_response_time_ms_nullable() -> None:
    cols = {c.name: c for c in QuizQuestion.__table__.columns}
    assert "expected_response_ms" not in cols, (
        "§C1: expected_response_ms must be renamed to expected_response_time_ms"
    )
    col = cols["expected_response_time_ms"]
    assert isinstance(col.type, Integer)
    assert col.nullable is True, (
        "§C1: T_exp must be nullable in draft; T7.5.9 publish gate enforces NOT NULL"
    )


def test_hint_text_is_text_type() -> None:
    cols = {c.name: c for c in QuizQuestion.__table__.columns}
    col = cols["hint_text"]
    assert isinstance(col.type, Text), (
        "§C1: hint_text is Text (unbounded), NOT String(500) per plan body"
    )
    assert col.nullable is True


def test_source_refs_no_json_suffix() -> None:
    cols = {c.name: c for c in QuizQuestion.__table__.columns}
    assert "source_refs_json" not in cols, "§C1: source_refs_json must be renamed to source_refs"
    col = cols["source_refs"]
    assert isinstance(col.type, JSONB)
    assert col.nullable is False


def test_quiz_question_preserved_columns() -> None:
    cols = {c.name for c in QuizQuestion.__table__.columns}
    for preserved in (
        "expected_ef_ceiling",
        "original_generated_payload",
        "reviewed_by",
        "reviewed_at",
        "published_at",
        "bloom_level",
        "difficulty",
        "review_status",
        "explanation",
    ):
        assert preserved in cols, f"§C1 PRESERVE: QuizQuestion.{preserved} must be kept"


def test_review_status_check_constraint() -> None:
    sqltext = _check_constraint_text(QuizQuestion, "ck_quiz_questions_review_status")
    for value in ("pending", "approved", "edited", "rejected"):
        assert f"'{value}'" in sqltext


def test_t_actual_ms_and_hint_used_present() -> None:
    cols = {c.name: c for c in QuizAttemptAnswer.__table__.columns}
    assert "response_time_ms" not in cols, (
        "plan §5371: response_time_ms must be renamed to t_actual_ms"
    )
    t_actual = cols["t_actual_ms"]
    assert isinstance(t_actual.type, Integer)
    assert t_actual.nullable is True
    hint_used = cols["hint_used"]
    assert isinstance(hint_used.type, Boolean)
    assert hint_used.nullable is False
    server_default_text = (
        str(hint_used.server_default.arg).upper() if hint_used.server_default else ""
    )
    assert "FALSE" in server_default_text


def test_no_quiz_mode_field() -> None:
    cols = {c.name for c in Quiz.__table__.columns}
    assert "mode" not in cols, "plan §5375 + §5381: every quiz is SR; no Quiz.mode field"


def test_no_quiz_generation_mode_column() -> None:
    cols = {c.name for c in Quiz.__table__.columns}
    assert "generation_mode" not in cols, (
        "§C1: generation_mode lives in generation_runs.config_json, not on quizzes table"
    )


def test_quiz_attempt_idempotency_key_unique() -> None:
    cols = {c.name: c for c in QuizAttempt.__table__.columns}
    assert "idempotency_key" in cols, "§C4: QuizAttempt.idempotency_key must be preserved"
    assert cols["idempotency_key"].unique is True


def test_quiz_source_lesson_preserved() -> None:
    cols = {c.name for c in QuizSourceLesson.__table__.columns}
    assert cols == {"quiz_id", "lesson_id", "created_at"}, (
        "§C3: QuizSourceLesson is composite PK + created_at link table"
    )
    pk_cols = {c.name for c in QuizSourceLesson.__table__.primary_key.columns}
    assert pk_cols == {"quiz_id", "lesson_id"}


def test_quiz_attempt_status_includes_graded() -> None:
    sqltext = _check_constraint_text(QuizAttempt, "ck_quiz_attempts_status")
    for value in ("in_progress", "submitted", "graded"):
        assert f"'{value}'" in sqltext, f"§C1: QuizAttempt.status must include '{value}'"


def test_quiz_question_revision_preserved() -> None:
    cols = {c.name for c in QuizQuestionRevision.__table__.columns}
    expected = {
        "id",
        "question_id",
        "revision_no",
        "source_kind",
        "payload_json",
        "created_by",
        "created_at",
    }
    assert expected.issubset(cols)


def test_quiz_question_option_unique_constraints() -> None:
    constraint_names = {getattr(c, "name", None) for c in QuizQuestionOption.__table__.constraints}
    assert "uq_quiz_question_options_position" in constraint_names
    assert "uq_quiz_question_options_key" in constraint_names


def test_quiz_passing_score_is_numeric_percent() -> None:
    cols = {c.name for c in Quiz.__table__.columns}
    assert "passing_score_percent" in cols, (
        "§A13 baseline-canon: column is passing_score_percent NUMERIC(5,2)"
    )
    assert "passing_score" not in cols
