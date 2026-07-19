"""Unit tests for the adaptive-interviewer runtime state schema (Phase 1).

Pure (de)serialization — no DB, no LLM. These lock in the compatibility
guarantees that make lazy initialisation of pre-existing sessions safe:

* an empty dict deserializes to a valid default state,
* unknown keys (written by newer code) are ignored,
* missing keys (old rows) fall back to defaults,
* a full round-trip preserves every field.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.state import (
    STATE_SCHEMA_VERSION,
    CandidateSignals,
    CoverageStatus,
    InterviewPhase,
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)


def test_empty_dict_yields_valid_default_state() -> None:
    # This is the lazy-init safety guarantee: an old session with no payload
    # (or a `{}` JSONB default) must produce a usable default state.
    data = InterviewRuntimeStateData.from_dict({})
    assert data.phase is InterviewPhase.OPENING
    assert data.asked_question_ids == []
    assert data.outcome_coverage == {}
    assert data.candidate_signals.requested_repeat is False
    assert data.version == STATE_SCHEMA_VERSION


def test_none_yields_valid_default_state() -> None:
    data = InterviewRuntimeStateData.from_dict(None)
    assert data.phase is InterviewPhase.OPENING
    assert data.total_follow_up_count == 0


def test_unknown_keys_are_ignored() -> None:
    # Newer code may write keys this version doesn't know; they must not crash.
    data = InterviewRuntimeStateData.from_dict(
        {"phase": "core", "some_future_field": {"nested": 1}}
    )
    assert data.phase is InterviewPhase.CORE


def test_invalid_phase_falls_back_to_opening() -> None:
    data = InterviewRuntimeStateData.from_dict({"phase": "not_a_phase"})
    assert data.phase is InterviewPhase.OPENING


def test_full_round_trip_preserves_fields() -> None:
    original = InterviewRuntimeStateData(
        phase=InterviewPhase.DEEP_PROBE,
        started_at="2026-07-14T10:00:00+00:00",
        remaining_time_seconds=600,
        current_question_id="q-1",
        current_outcome_id="o-1",
        asked_question_ids=["q-1", "q-2"],
        skipped_question_ids=["q-3"],
        completed_question_ids=["q-1"],
        current_question_follow_up_count=2,
        total_follow_up_count=5,
        outcome_coverage={
            "o-1": OutcomeCoverageState(
                outcome_id="o-1",
                evidence_count=3,
                provisional_score=0.75,
                confidence=0.8,
                status=CoverageStatus.SUFFICIENT,
                last_updated_at="2026-07-14T10:05:00+00:00",
                supporting_turn_ids=["t-1", "t-2"],
                missing_evidence=["tradeoff analysis"],
            )
        },
        last_student_intent={"intent": "answer", "confidence": 0.9},
        last_answer_analysis={"relevance": "relevant"},
        consecutive_weak_answers=1,
        consecutive_strong_answers=2,
        candidate_signals=CandidateSignals(requested_clarification=True),
    )

    restored = InterviewRuntimeStateData.from_dict(original.to_dict())

    assert restored.phase is InterviewPhase.DEEP_PROBE
    assert restored.remaining_time_seconds == 600
    assert restored.asked_question_ids == ["q-1", "q-2"]
    assert restored.skipped_question_ids == ["q-3"]
    assert restored.current_question_follow_up_count == 2
    assert restored.total_follow_up_count == 5
    assert restored.last_student_intent == {"intent": "answer", "confidence": 0.9}
    assert restored.consecutive_strong_answers == 2
    assert restored.candidate_signals.requested_clarification is True

    cov = restored.outcome_coverage["o-1"]
    assert cov.evidence_count == 3
    assert cov.provisional_score == 0.75
    assert cov.status is CoverageStatus.SUFFICIENT
    assert cov.supporting_turn_ids == ["t-1", "t-2"]
    assert cov.missing_evidence == ["tradeoff analysis"]


def test_coverage_invalid_status_falls_back() -> None:
    cov = OutcomeCoverageState.from_dict({"outcome_id": "o-9", "status": "garbage"})
    assert cov.status is CoverageStatus.NOT_STARTED
    assert cov.outcome_id == "o-9"


def test_to_dict_is_json_serializable() -> None:
    import json

    data = InterviewRuntimeStateData(
        phase=InterviewPhase.CLOSING,
        outcome_coverage={"o-1": OutcomeCoverageState(outcome_id="o-1")},
    )
    # Must not raise — the payload is stored as JSONB.
    dumped = json.dumps(data.to_dict())
    assert "closing" in dumped
