"""Unit tests for the canonical legacy↔adaptive response mapper (Slice 4).

Proves safeguards #4 (single source of truth — legacy + structured fields can
never contradict) and #5 (anti-double-render: on advance the question text lives
ONLY in next_question, never duplicated into followup_text).
"""

from __future__ import annotations

from types import SimpleNamespace

from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.mapping import canonical_step_result
from abridgeai.features.interviews.orchestrator.utterance import Utterance


def _utt(ack: str = "", transition: str = "", qp: str = "") -> Utterance:
    combined = " ".join(p for p in (ack, transition, qp) if p).strip()
    return Utterance(
        acknowledgement=ack, transition=transition, question_or_probe=qp, ai_turn_text=combined
    )


def _decision(
    action: InterviewerActionType, reason: ReasonCode, **kw: object
) -> InterviewerDecision:
    return InterviewerDecision(action=action, reason_code=reason, **kw)  # type: ignore[arg-type]


def test_advance_puts_question_only_in_next_question() -> None:
    """Anti-double-render: on advance, followup_text is ack+transition ONLY."""
    q = SimpleNamespace(id="q-123", prompt_text="What is a fact table?")
    utt = _utt(ack="Thank you.", transition="Let's move on.", qp="What is a fact table?")
    d = _decision(
        InterviewerActionType.TRANSITION_TOPIC,
        ReasonCode.OUTCOME_SUFFICIENTLY_COVERED,
        should_advance_question=True,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
    )
    res = canonical_step_result(
        decision=d,
        utterance=utt,
        selected_question=q,
        language="en",
        state_version=5,
        ai_turn_id="ai-1",
        utterance_status="fallback",
    )
    # Legacy followup carries ack+transition, NOT the question.
    assert res["followup_text"] == "Thank you. Let's move on."
    assert "fact table" not in (res["followup_text"] or "")
    # Question rides in next_question only.
    assert res["next_question"] is q
    assert res["is_finished"] is False
    # Structured mirror.
    assert res["action"] == "transition_topic"
    assert res["should_await_response"] is True
    assert res["should_finish"] is False
    assert res["current_question_id"] == "q-123"
    assert res["state_version"] == 5


def test_probe_puts_full_utterance_in_followup_and_no_next_question() -> None:
    utt = _utt(ack="I understand your reasoning.", qp="Could you give a concrete example?")
    d = _decision(
        InterviewerActionType.ASK_FOR_EXAMPLE,
        ReasonCode.MISSING_EXAMPLE,
        should_advance_question=False,
        should_record_academic_evidence=True,
    )
    res = canonical_step_result(
        decision=d,
        utterance=utt,
        selected_question=None,
        language="en",
        state_version=2,
        ai_turn_id="ai-2",
        utterance_status="llm",
    )
    assert res["next_question"] is None
    assert res["is_finished"] is False
    assert res["followup_text"] == "I understand your reasoning. Could you give a concrete example?"
    assert res["ai_turn_text"] == res["followup_text"]
    assert res["action"] == "ask_for_example"
    assert res["should_await_response"] is True
    assert res["_should_record_evidence"] is True


def test_closing_sets_finished_but_keeps_utterance_for_render() -> None:
    """Safeguard #6: closing utterance survives even though is_finished=True."""
    utt = _utt(qp="That concludes the interview. Anything to add?")
    d = _decision(
        InterviewerActionType.BEGIN_CLOSING,
        ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED,
    )
    res = canonical_step_result(
        decision=d,
        utterance=utt,
        selected_question=None,
        language="en",
        state_version=9,
        ai_turn_id="ai-3",
        utterance_status="fallback",
    )
    assert res["is_finished"] is True
    assert res["should_finish"] is True
    assert res["should_await_response"] is False
    # The closing message MUST still be present so the client renders + narrates it.
    assert res["followup_text"] == "That concludes the interview. Anything to add?"
    assert res["ai_turn_text"] == "That concludes the interview. Anything to add?"
    assert res["next_question"] is None


def test_repeat_carries_full_utterance_no_advance_no_score() -> None:
    utt = _utt(transition="Of course. Here is the question again:", qp="What is normalization?")
    d = _decision(
        InterviewerActionType.REPEAT_QUESTION,
        ReasonCode.STUDENT_REQUESTED_REPEAT,
        should_record_academic_evidence=False,
    )
    res = canonical_step_result(
        decision=d,
        utterance=utt,
        selected_question=None,
        language="en",
        state_version=1,
        ai_turn_id="ai-4",
        utterance_status="fallback",
    )
    assert res["next_question"] is None
    assert res["is_finished"] is False
    assert res["action"] == "repeat_question"
    assert res["_should_record_evidence"] is False
    assert "What is normalization?" in res["followup_text"]


def test_language_and_narrate_flags_propagate() -> None:
    utt = _utt(ack="Cảm ơn bạn.", transition="Chúng ta tiếp tục nhé.", qp="Fact table là gì?")
    q = SimpleNamespace(id="q-9", prompt_text="Fact table là gì?")
    d = _decision(
        InterviewerActionType.TRANSITION_TOPIC,
        ReasonCode.PARTIAL_OUTCOME_COVERAGE,
        should_advance_question=True,
    )
    res = canonical_step_result(
        decision=d,
        utterance=utt,
        selected_question=q,
        language="vi",
        state_version=3,
        ai_turn_id="ai-5",
        utterance_status="llm",
    )
    assert res["language"] == "vi"
    assert res["should_narrate"] is True
    # No double-render in Vietnamese either.
    assert "Fact table là gì?" not in (res["followup_text"] or "")


def test_empty_utterance_sets_narrate_false() -> None:
    utt = _utt()  # everything empty
    d = _decision(
        InterviewerActionType.HANDLE_TECHNICAL_ISSUE,
        ReasonCode.TECHNICAL_ISSUE,
    )
    res = canonical_step_result(
        decision=d,
        utterance=utt,
        selected_question=None,
        language="en",
        state_version=1,
        ai_turn_id=None,
        utterance_status="fallback",
    )
    assert res["should_narrate"] is False
    assert res["followup_text"] is None
    assert res["ai_turn_id"] is None
