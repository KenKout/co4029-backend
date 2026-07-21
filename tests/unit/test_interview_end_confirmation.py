"""Unit tests for the end-confirmation gate (Slice 4).

Covers the three pure layers the gate touches:

1. ``decide_next_action`` — a fresh end request asks to confirm (never closes
   immediately); while pending, confirm/end closes and anything else cancels;
   a confirm/cancel reply with NO pending confirmation is not an end signal.
2. ``classify_confirmation_reply`` — context-scoped yes/no classification used
   only while a confirmation is pending (EN + VI).
3. ``InterviewRuntimeStateData`` (v3) — the new ``interaction_state`` /
   ``pending_confirmation`` fields round-trip and tolerate old (v2) payloads.

All three are pure (no DB, no LLM), so they're exercised with plain objects.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
)
from abridgeai.features.interviews.orchestrator.decision import (
    DecisionInputs,
    InterviewerActionType,
    ReasonCode,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
    classify_confirmation_reply,
)
from abridgeai.features.interviews.orchestrator.state import (
    STATE_SCHEMA_VERSION,
    InteractionState,
    InterviewRuntimeStateData,
)


def _intent(kind: StudentIntent) -> IntentClassification:
    return IntentClassification(intent=kind, confidence=0.9, rationale="test")


def _analysis(
    *,
    relevance: Relevance = Relevance.RELEVANT,
    correctness: Correctness = Correctness.MOSTLY_CORRECT,
    probe: ProbeType = ProbeType.NONE,
) -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=relevance,
        correctness=correctness,
        recommended_probe_type=probe,
        confidence=0.7,
    )


def _inputs(**overrides: object) -> DecisionInputs:
    base: dict[str, object] = {
        "intent": _intent(StudentIntent.ANSWER),
        "analysis": _analysis(),
        "current_question_follow_up_count": 0,
        "total_follow_up_count": 0,
        "time_fraction_remaining": 0.8,
        "has_next_question": True,
        "all_required_outcomes_covered": False,
    }
    base.update(overrides)
    return DecisionInputs(**base)  # type: ignore[arg-type]


# ── decision policy: the confirmation gate ───────────────────────────────────


def test_fresh_end_request_asks_to_confirm_not_close() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.END_INTERVIEW)))
    assert d.action is InterviewerActionType.REQUEST_END_CONFIRMATION
    assert d.reason_code is ReasonCode.END_CONFIRMATION_REQUESTED
    # Critical: a fresh end request must NOT close immediately.
    assert d.should_end_session is False
    assert d.should_advance_question is False
    assert d.should_record_academic_evidence is False


def test_pending_confirm_end_closes() -> None:
    d = decide_next_action(
        _inputs(intent=_intent(StudentIntent.CONFIRM_END), pending_confirmation=True)
    )
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.END_CONFIRMED
    assert "end_confirmed" in d.tags


def test_pending_repeated_end_request_also_closes() -> None:
    # A second natural "end it" while pending is treated as confirmation.
    d = decide_next_action(
        _inputs(intent=_intent(StudentIntent.END_INTERVIEW), pending_confirmation=True)
    )
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.END_CONFIRMED


def test_pending_cancel_resumes_same_question() -> None:
    d = decide_next_action(
        _inputs(intent=_intent(StudentIntent.CANCEL_END), pending_confirmation=True)
    )
    assert d.action is InterviewerActionType.CANCEL_END
    assert d.reason_code is ReasonCode.END_CANCELLED
    assert d.should_advance_question is False
    assert d.should_end_session is False
    assert d.should_record_academic_evidence is False


def test_pending_any_other_intent_cancels_and_resumes() -> None:
    # While a confirmation is pending, an ordinary answer is treated as a
    # cancel (the candidate kept talking) — never scored, never advanced.
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.ANSWER), pending_confirmation=True))
    assert d.action is InterviewerActionType.CANCEL_END
    assert d.reason_code is ReasonCode.END_CANCELLED
    assert d.should_record_academic_evidence is False


def test_confirm_reply_without_pending_is_not_an_end_signal() -> None:
    # A bare confirm/cancel intent with NO pending confirmation must fall
    # through to normal answer handling (not close, not cancel).
    confirm = decide_next_action(
        _inputs(intent=_intent(StudentIntent.CONFIRM_END), pending_confirmation=False)
    )
    assert confirm.action is not InterviewerActionType.BEGIN_CLOSING
    assert confirm.action is not InterviewerActionType.CANCEL_END
    cancel = decide_next_action(
        _inputs(intent=_intent(StudentIntent.CANCEL_END), pending_confirmation=False)
    )
    assert cancel.action is not InterviewerActionType.CANCEL_END


def test_request_end_confirmation_never_advances_or_ends() -> None:
    # Property: the confirmation-request action holds on the current question.
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.END_INTERVIEW)))
    assert d.should_advance_question is False
    assert d.should_end_session is False


# ── context-scoped confirm/cancel classification ─────────────────────────────


def test_confirm_replies_classify_as_confirm_end() -> None:
    for text in ("yes", "confirm", "end and submit", "yes, end the interview", "có", "đồng ý"):
        reply = classify_confirmation_reply(text)
        assert reply is not None, text
        assert reply.intent is StudentIntent.CONFIRM_END, text


def test_cancel_replies_classify_as_cancel_end() -> None:
    for text in ("no", "continue", "not yet", "keep going", "không", "tiếp tục"):
        reply = classify_confirmation_reply(text)
        assert reply is not None, text
        assert reply.intent is StudentIntent.CANCEL_END, text


def test_substantive_answer_is_not_a_confirmation_reply() -> None:
    # A real answer while pending must NOT be hijacked as yes/no — it returns
    # None so the decision layer's "anything else cancels" rule handles it.
    assert classify_confirmation_reply("because a fact table stores measures") is None


def test_empty_reply_is_none() -> None:
    assert classify_confirmation_reply("") is None
    assert classify_confirmation_reply("   ") is None


# ── state serialization (schema v3) ──────────────────────────────────────────


def test_new_fields_round_trip() -> None:
    data = InterviewRuntimeStateData(
        interaction_state=InteractionState.CONFIRMING_END,
        pending_confirmation=True,
    )
    restored = InterviewRuntimeStateData.from_dict(data.to_dict())
    assert restored.interaction_state is InteractionState.CONFIRMING_END
    assert restored.pending_confirmation is True


def test_schema_version_is_three() -> None:
    assert STATE_SCHEMA_VERSION == 3
    assert InterviewRuntimeStateData().version == 3


def test_v2_payload_without_new_fields_defaults_safely() -> None:
    # A payload persisted before Slice 4 has neither field. Deserialization must
    # default to a non-pending, awaiting-answer state (never crash, never leave
    # a phantom confirmation pending).
    legacy = {"phase": "core", "version": 2}
    restored = InterviewRuntimeStateData.from_dict(legacy)
    assert restored.pending_confirmation is False
    assert restored.interaction_state is InteractionState.AWAITING_ANSWER


def test_unknown_interaction_state_defaults_to_awaiting() -> None:
    restored = InterviewRuntimeStateData.from_dict({"interaction_state": "bogus"})
    assert restored.interaction_state is InteractionState.AWAITING_ANSWER
