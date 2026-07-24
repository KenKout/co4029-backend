"""Serialization allowlists for learner REST and LiveKit contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

from abridgeai.api import create_app
from abridgeai.features.interviews.models import InterviewSecurityEvent
from abridgeai.features.interviews.schemas.public import (
    InterviewConfigPublic,
    InterviewForTakingPublic,
    InterviewQuestionPublic,
)
from abridgeai.features.interviews.schemas.session import InterviewSubmitAnswerResponse
from abridgeai.features.interviews.services.real_time import build_agent_metadata

_HIDDEN_FIELDS = {
    "model_answer",
    "expected_evidence",
    "common_misconceptions",
    "importance_weight",
    "min_outcomes_to_pass",
    "supplementary_instructions",
    "rubric_scores",
    "candidate_question_scores",
    "internal_summary_json",
    "phase",
    "action",
    "reason_code",
    "target_outcome_id",
    "state_version",
    "security_attempt_count",
    "session_security_flagged",
}


def test_question_public_serialization_is_an_explicit_allowlist() -> None:
    source = SimpleNamespace(
        id=uuid4(),
        prompt_text="Explain transaction isolation.",
        question_type="conceptual",
        model_answer="Serializable execution is the hidden answer.",
        expected_evidence=["mentions anomalies"],
        common_misconceptions=["confuses atomicity"],
        difficulty="senior",
        linked_outcome_id=uuid4(),
        source_refs_json=[{"chunk": "secret"}],
    )
    payload = InterviewQuestionPublic.model_validate(source).model_dump(mode="json")
    assert set(payload) == {"id", "prompt_text", "question_type"}
    assert not (_HIDDEN_FIELDS & set(payload))


def test_taking_payload_cannot_serialize_outcome_or_bank_metadata() -> None:
    config = InterviewConfigPublic(
        id=uuid4(),
        course_id=uuid4(),
        module_id=uuid4(),
        title="Secure interview",
        status="published",
        supported_modes="hybrid",
        lock_quiz_ef_until_pass=False,
    )
    question = InterviewQuestionPublic(
        id=uuid4(),
        prompt_text="Explain transaction isolation.",
        question_type="conceptual",
    )
    payload = InterviewForTakingPublic(config=config, first_question=question).model_dump(
        mode="json"
    )
    # outcome_count is a SAFE count-only signal (how many criteria this
    # interview assesses) — it carries no outcome text / weight / threshold, so
    # it's an allowed field. The contract still forbids the raw bank/outcomes.
    assert set(payload) == {"config", "first_question", "outcome_count"}
    assert isinstance(payload["outcome_count"], int)
    assert "questions" not in payload
    assert "outcomes" not in payload
    assert not (_HIDDEN_FIELDS & set(payload["config"]))


def test_submit_response_drops_internal_orchestrator_fields() -> None:
    payload = InterviewSubmitAnswerResponse(
        next_question=None,
        is_finished=False,
        ai_turn_text="Please answer the current question.",
        language="en",
        should_narrate=True,
        should_await_response=True,
        should_finish=False,
        phase="core",
        action="transition_topic",
        reason_code="sufficient_evidence",
        target_outcome_id=str(uuid4()),
        state_version=7,
        candidate_question_scores=[{"id": "secret"}],
    ).model_dump(mode="json")
    assert not (_HIDDEN_FIELDS & set(payload))
    assert set(payload) == {
        "next_question",
        "is_finished",
        "ai_followup_text",
        "time_remaining_seconds",
        "ai_turn_text",
        "language",
        "should_narrate",
        "should_await_response",
        "should_finish",
        "assistance_kind",
        # Natural Interview Transitions — additive, safe (no hidden internals).
        "transition_id",
        "transition_text",
        "transition_target",
        # End-confirmation gate (Slice 4) — additive, safe public response fields
        # that drive the client's confirm-end UX (not internal orchestrator state).
        "pending_confirmation",
        "interaction_state",
    }


def test_livekit_dispatch_metadata_contains_only_routing_fields() -> None:
    session_id = uuid4()
    student_id = uuid4()
    metadata = json.loads(build_agent_metadata(session_id, student_id, language="vi-VN"))
    assert metadata == {
        "session_id": str(session_id),
        "student_id": str(student_id),
        "language": "vi",
    }
    assert not (_HIDDEN_FIELDS & set(metadata))


def test_security_event_schema_is_redacted_and_uses_confidence_band_only() -> None:
    columns = set(InterviewSecurityEvent.__table__.columns.keys())
    assert {"session_id", "turn_id", "category", "confidence_band", "action"} <= columns
    assert {
        "student_response",
        "raw_response",
        "redacted_excerpt",
        "confidence",
        "reasoning",
        "prompt",
        "tool_arguments",
    }.isdisjoint(columns)


def test_openapi_learner_components_exclude_hidden_security_and_rubric_fields() -> None:
    schemas = create_app().openapi()["components"]["schemas"]
    learner_components = (
        "InterviewConfigPublic",
        "InterviewForTakingPublic",
        "InterviewQuestionPublic",
        "InterviewSubmitAnswerResponse",
        "RealtimeTokenResponse",
    )
    for component in learner_components:
        properties = set(schemas[component].get("properties", {}))
        assert not (_HIDDEN_FIELDS & properties), component
    assert "outcomes" not in schemas["InterviewForTakingPublic"]["properties"]
    assert "questions" not in schemas["InterviewForTakingPublic"]["properties"]


def test_authoring_openapi_exposes_policy_knobs_but_not_detector_rules() -> None:
    schemas = create_app().openapi()["components"]["schemas"]
    properties = set(schemas["InterviewConfigUpdate"]["properties"])
    assert {
        "security_response_policy",
        "security_max_consecutive_attempts",
        "security_custom_refusal_en",
        "security_custom_refusal_vi",
        "security_incident_summary_enabled",
    } <= properties
    assert "security_detection_rules" not in properties
    assert "interview_security_guard_mode" not in properties
