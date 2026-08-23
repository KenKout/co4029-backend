"""Unit tests for the interviews aggregate ORM models (T6.1).

Covers Reconciliation §A13 + plan §6.1 invariants:
* All 9 ORM models import cleanly (5 in plan body + 4 baseline-canon
  additions: InterviewOutcome, InterviewSessionQuestion,
  InterviewSessionMessage, InterviewOutcomeEvaluation,
  AssessmentIntegrityEvent).
* Authoring-side soft-deletable models (``InterviewConfig``,
  ``InterviewOutcome``, ``InterviewQuestion``) carry the 7 audit
  columns supplied by the
  ``UUIDPrimaryKeyMixin + TimestampMixin + AuditedByMixin +
  SoftDeleteMixin`` stack.
* Session/runtime models (``InterviewSession``,
  ``InterviewSessionMessage``, ``InterviewOutcomeEvaluation``) are
  historical record — no soft-delete cols (plan §6.1 explicit).
* Append-only models (``InterviewSessionQuestion``,
  ``AssessmentIntegrityEvent``) carry only ``created_at``.
* Status / persona / question_type / role / etc.
  CHECK constraints match baseline DDL verbatim.
* ``UNIQUE (interview_config_id, student_id, attempt_number)`` on
  sessions is preserved (one row per attempt).
* Voice forward-compat FKs (``transcript_object_id``,
  ``recording_object_id``, ``audio_object_id``) are nullable.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB

from abridgeai.features.interviews.models import (
    AssessmentIntegrityEvent,
    GapReport,
    InterviewConfig,
    InterviewOutcome,
    InterviewOutcomeEvaluation,
    InterviewQuestion,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)


def test_interview_models_importable() -> None:
    models = [
        InterviewConfig,
        InterviewOutcome,
        InterviewQuestion,
        InterviewSession,
        InterviewSessionQuestion,
        InterviewSessionMessage,
        InterviewOutcomeEvaluation,
        GapReport,
        AssessmentIntegrityEvent,
    ]
    assert len(models) == 9
    table_names = {m.__tablename__ for m in models}
    assert table_names == {
        "interview_configs",
        "interview_outcomes",
        "interview_questions",
        "interview_sessions",
        "interview_session_questions",
        "interview_session_messages",
        "interview_outcome_evaluations",
        "gap_reports",
        "assessment_integrity_events",
    }


@pytest.mark.parametrize(
    "model",
    [InterviewConfig, InterviewOutcome, InterviewQuestion],
    ids=lambda m: m.__name__,
)
def test_authoring_audit_columns_present(model: type) -> None:
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


@pytest.mark.parametrize(
    "model",
    [InterviewSession, InterviewSessionMessage, InterviewOutcomeEvaluation, GapReport],
    ids=lambda m: m.__name__,
)
def test_session_models_have_no_soft_delete(model: type) -> None:
    cols = {c.name for c in model.__table__.columns}
    assert "deleted_at" not in cols, (
        f"plan §6.1: {model.__name__} is historical record, no soft-delete"
    )
    assert "deleted_by" not in cols


def test_interview_session_question_is_append_only() -> None:
    cols = {c.name for c in InterviewSessionQuestion.__table__.columns}
    assert "created_at" in cols
    assert "updated_at" not in cols, (
        "§A13: interview_session_questions is immutable record (CreatedAtMixin only)"
    )


def test_assessment_integrity_event_is_append_only() -> None:
    cols = {c.name for c in AssessmentIntegrityEvent.__table__.columns}
    assert "created_at" in cols
    assert "updated_at" not in cols, (
        "§A13: assessment_integrity_events is append-only proctoring log"
    )


def _check_constraint_text(model: type, name: str) -> str:
    for constraint in model.__table__.constraints:
        if getattr(constraint, "name", None) == name:
            return str(constraint.sqltext)
    raise AssertionError(f"{model.__name__} missing CHECK constraint {name}")


def test_interview_config_status_check() -> None:
    sqltext = _check_constraint_text(InterviewConfig, "ck_interview_configs_status")
    for value in ("draft", "published", "archived"):
        assert f"'{value}'" in sqltext


def test_interview_config_persona_check_baseline_canon() -> None:
    sqltext = _check_constraint_text(InterviewConfig, "ck_interview_configs_persona")
    for value in ("strict", "neutral", "supportive"):
        assert f"'{value}'" in sqltext, (
            f"§A13 baseline-canon: persona must include '{value}' (NOT plan-body 'friendly')"
        )
    assert "'friendly'" not in sqltext, (
        "§A13 baseline-canon: 'friendly' is plan-body fabrication; baseline uses 'supportive'"
    )





def test_interview_question_type_check_baseline_canon() -> None:
    sqltext = _check_constraint_text(InterviewQuestion, "ck_interview_questions_question_type")
    for value in (
        "conceptual",
        "behavioral",
        "technical",
        "situational",
        "system_design",
    ):
        assert f"'{value}'" in sqltext, f"§A13 baseline-canon: question_type must include '{value}'"


def test_interview_question_difficulty_check() -> None:
    sqltext = _check_constraint_text(InterviewQuestion, "ck_interview_questions_difficulty")
    for value in ("junior", "mid_level", "senior"):
        assert f"'{value}'" in sqltext


def test_interview_question_review_status_check() -> None:
    sqltext = _check_constraint_text(InterviewQuestion, "ck_interview_questions_review_status")
    for value in ("pending", "approved", "edited", "rejected"):
        assert f"'{value}'" in sqltext


def test_interview_session_status_baseline_canon() -> None:
    sqltext = _check_constraint_text(InterviewSession, "ck_interview_sessions_status")
    for value in ("in_progress", "completed", "timed_out", "abandoned", "failed"):
        assert f"'{value}'" in sqltext, (
            f"§A13 baseline-canon: session status must include '{value}' (5 values, not 3)"
        )


def test_interview_session_input_mode_check() -> None:
    sqltext = _check_constraint_text(InterviewSession, "ck_interview_sessions_input_mode")
    for value in ("voice", "text", "hybrid"):
        assert f"'{value}'" in sqltext


def test_interview_session_attempt_unique_constraint() -> None:
    constraint_names = {getattr(c, "name", None) for c in InterviewSession.__table__.constraints}
    assert "uq_interview_sessions_number" in constraint_names, (
        "§A13: UNIQUE (interview_config_id, student_id, attempt_number) required"
    )


def test_interview_session_voice_columns_nullable() -> None:
    cols = {c.name: c for c in InterviewSession.__table__.columns}
    for column_name in ("transcript_object_id", "recording_object_id"):
        assert cols[column_name].nullable is True, (
            f"plan §6.1 must-not: {column_name} stays nullable for voice forward-compat"
        )


def test_interview_session_message_audio_nullable() -> None:
    cols = {c.name: c for c in InterviewSessionMessage.__table__.columns}
    assert cols["audio_object_id"].nullable is True, (
        "voice forward-compat: audio_object_id must be nullable"
    )


def test_interview_session_message_role_baseline_canon() -> None:
    sqltext = _check_constraint_text(InterviewSessionMessage, "ck_interview_session_messages_role")
    for value in ("ai", "user", "system"):
        assert f"'{value}'" in sqltext, (
            f"§A13 baseline-canon: role must include '{value}' (NOT 'assistant')"
        )
    assert "'assistant'" not in sqltext, (
        "§A13 baseline-canon: baseline uses 'ai' not OpenAI's 'assistant'"
    )


def test_interview_outcome_unique_position() -> None:
    constraint_names = {getattr(c, "name", None) for c in InterviewOutcome.__table__.constraints}
    assert "uq_interview_outcomes_position" in constraint_names


def test_interview_outcome_evaluation_unique() -> None:
    constraint_names = {
        getattr(c, "name", None) for c in InterviewOutcomeEvaluation.__table__.constraints
    }
    assert "uq_interview_outcome_evaluations" in constraint_names


def test_interview_outcome_evaluation_verdict_columns() -> None:
    cols = {c.name: c for c in InterviewOutcomeEvaluation.__table__.columns}
    assert isinstance(cols["verdict_met"].type, Boolean)
    assert cols["verdict_met"].nullable is False
    assert cols["hidden_reasoning"].nullable is True
    assert cols["evidence_excerpt"].nullable is True


def test_interview_question_source_refs_baseline_name() -> None:
    cols = {c.name: c for c in InterviewQuestion.__table__.columns}
    assert "source_refs_json" in cols, (
        "§A13: interview_questions keeps baseline name source_refs_json "
        "(quizzes-only rename via migration 0007)"
    )
    assert isinstance(cols["source_refs_json"].type, JSONB)
    assert cols["source_refs_json"].nullable is False


def test_gap_report_columns() -> None:
    cols = {c.name: c for c in GapReport.__table__.columns}
    assert cols["module_id"].nullable is True, "gap may span entire course"
    assert cols["source_quiz_attempt_id"].nullable is True
    assert cols["source_interview_session_id"].nullable is True
    assert isinstance(cols["report_json"].type, JSONB)
    assert cols["report_json"].nullable is False


def test_assessment_integrity_polymorphic_check() -> None:
    sqltext = _check_constraint_text(AssessmentIntegrityEvent, "ck_assessment_integrity_parent_ref")
    assert "quiz_attempt_id" in sqltext
    assert "interview_session_id" in sqltext


def test_assessment_integrity_event_type_check() -> None:
    sqltext = _check_constraint_text(AssessmentIntegrityEvent, "ck_assessment_integrity_event_type")
    for value in (
        "focus_lost",
        "tab_switch",
        "fullscreen_exit",
        "warning_issued",
        "reconnect",
        "disconnect",
    ):
        assert f"'{value}'" in sqltext


def test_interview_config_preserved_columns() -> None:
    cols = {c.name for c in InterviewConfig.__table__.columns}
    for preserved in (
        "max_attempts",
        "min_outcomes_to_pass",
        "time_limit_minutes",
        "supplementary_instructions",
        "generation_run_id",
        "published_at",
    ):
        assert preserved in cols, (
            f"§A13 baseline-canon: InterviewConfig.{preserved} must be preserved"
        )


def test_interview_config_time_limit_in_minutes() -> None:
    cols = {c.name: c for c in InterviewConfig.__table__.columns}
    assert "time_limit_minutes" in cols, (
        "§A13 baseline-canon: interviews use minutes (NOT seconds like quizzes)"
    )
    assert "time_limit_seconds" not in cols
    assert isinstance(cols["time_limit_minutes"].type, Integer)
