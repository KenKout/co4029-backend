"""Unit tests for the Phase 9 interviewer utterance generator.

Pure — no DB, no LLM. Covers the deterministic fallback (the guaranteed path):
persona/language matrix, the acknowledgement+transition-only split used to
avoid double-rendering the question on an advance, answer-safe generic probes,
and self-contained templates for non-question actions.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.utterance import (
    Persona,
    build_fallback_utterance,
    persona_from,
)


def _decision(
    action: InterviewerActionType,
    *,
    ack: AcknowledgementStyle = AcknowledgementStyle.NONE,
    reason: ReasonCode = ReasonCode.PARTIAL_OUTCOME_COVERAGE,
) -> InterviewerDecision:
    return InterviewerDecision(action=action, reason_code=reason, acknowledgement_style=ack)


def test_persona_from_defaults_to_neutral() -> None:
    assert persona_from("strict") is Persona.STRICT
    assert persona_from("supportive") is Persona.SUPPORTIVE
    assert persona_from(None) is Persona.NEUTRAL
    assert persona_from("bogus") is Persona.NEUTRAL


def test_advance_splits_ack_transition_from_question() -> None:
    """On an advance, ack+transition must NOT contain the question (no doubling)."""
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    u = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="What is a fact table?"
    )
    # Combined contains everything...
    assert "What is a fact table?" in u.ai_turn_text
    # ...but the ack+transition-only view (legacy ai_followup_text on advance) does NOT.
    assert "What is a fact table?" not in u.acknowledgement_and_transition()
    assert u.question_or_probe == "What is a fact table?"


def test_vietnamese_templates_used_for_vi() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    u = build_fallback_utterance(
        d, persona=Persona.SUPPORTIVE, language="vi-VN", question_text="Fact table là gì?"
    )
    assert "Cảm ơn" in u.acknowledgement  # VI acknowledgement
    assert "Fact table là gì?" in u.ai_turn_text


def test_unknown_language_falls_back_to_english() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    u = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="fr", question_text="Q?")
    assert u.acknowledgement == "Thank you."


def _model_answer_leak_guard(text: str) -> None:
    """Depth probes must never contain answer/rubric content — only ask to expand."""
    lowered = text.lower()
    for banned in ("the answer is", "correct answer", "you should say", "rubric"):
        assert banned not in lowered


def test_depth_probe_extend_answer_renders_bilingual_and_answer_safe() -> None:
    """Slice 8: EXTEND_ANSWER has non-empty, answer-safe EN + VI fallbacks."""
    d = _decision(
        InterviewerActionType.EXTEND_ANSWER,
        ack=AcknowledgementStyle.POSITIVE,
        reason=ReasonCode.STRONG_ANSWER_DEPTH_PROBE,
    )
    en = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    vi = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="vi", question_text=None)
    assert en.question_or_probe.strip()
    assert vi.question_or_probe.strip()
    assert en.question_or_probe != vi.question_or_probe  # genuinely localized
    _model_answer_leak_guard(en.ai_turn_text)
    _model_answer_leak_guard(vi.ai_turn_text)


def test_depth_probe_edge_case_renders_bilingual_and_answer_safe() -> None:
    d = _decision(
        InterviewerActionType.PROBE_EDGE_CASE,
        ack=AcknowledgementStyle.POSITIVE,
        reason=ReasonCode.STRONG_ANSWER_DEPTH_PROBE,
    )
    en = build_fallback_utterance(d, persona=Persona.STRICT, language="en", question_text=None)
    vi = build_fallback_utterance(d, persona=Persona.STRICT, language="vi", question_text=None)
    assert en.question_or_probe.strip()
    assert vi.question_or_probe.strip()
    _model_answer_leak_guard(en.ai_turn_text)
    _model_answer_leak_guard(vi.ai_turn_text)


def test_strict_persona_is_terse() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    strict = build_fallback_utterance(d, persona=Persona.STRICT, language="en", question_text="Q?")
    supportive = build_fallback_utterance(
        d, persona=Persona.SUPPORTIVE, language="en", question_text="Q?"
    )
    # Strict acknowledgement is shorter than supportive.
    assert len(strict.acknowledgement) < len(supportive.acknowledgement)


def test_repeat_question_has_no_acknowledgement_and_carries_question() -> None:
    d = _decision(InterviewerActionType.REPEAT_QUESTION, reason=ReasonCode.STUDENT_REQUESTED_REPEAT)
    u = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Original Q?"
    )
    assert u.acknowledgement == ""
    assert "Original Q?" in u.question_or_probe
    # Standardized repeat signpost (Natural Interview Transitions spec).
    assert "repeat the question" in u.ai_turn_text.lower()


def test_ask_for_example_uses_generic_probe_when_no_text() -> None:
    d = _decision(InterviewerActionType.ASK_FOR_EXAMPLE, reason=ReasonCode.MISSING_EXAMPLE)
    u = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    assert "example" in u.question_or_probe.lower()
    # Never leaks an answer — it asks the student to supply the example.
    assert u.question_or_probe != ""


def test_clarify_never_empty_and_answer_safe() -> None:
    d = _decision(
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        reason=ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
    )
    u = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    assert u.question_or_probe  # non-empty
    # Asks which part to rephrase — does not answer the question.
    assert "rephrase" in u.ai_turn_text.lower() or "part" in u.ai_turn_text.lower()


def test_technical_issue_is_self_contained_no_question() -> None:
    d = _decision(InterviewerActionType.HANDLE_TECHNICAL_ISSUE, reason=ReasonCode.TECHNICAL_ISSUE)
    u = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    assert u.question_or_probe == ""
    assert u.ai_turn_text  # still says something reassuring


def test_closing_asks_for_final_addition() -> None:
    d = _decision(InterviewerActionType.BEGIN_CLOSING, reason=ReasonCode.CLOSING_REQUIRED)
    en = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    vi = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="vi", question_text=None)
    assert "concludes" in en.ai_turn_text.lower()
    assert en.ai_turn_text  # rendered before is_finished transition (safeguard #6)
    assert "kết thúc" in vi.ai_turn_text.lower()


def test_combined_text_has_no_double_spaces() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NONE)
    # No acknowledgement → combine must not leave a leading/double space.
    u = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text="Q?")
    assert "  " not in u.ai_turn_text
    assert not u.ai_turn_text.startswith(" ")


# ── affect-aware tone (Slice 10) ─────────────────────────────────────────────


def test_affect_none_leaves_utterance_unchanged() -> None:
    """v1 parity: affect=None (default) must not alter the utterance at all."""
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(InterviewerActionType.PROBE_DEEPER, ack=AcknowledgementStyle.NEUTRAL)
    base = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text="Q?")
    neutral = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", affect=Affect.NEUTRAL
    )
    assert base.ai_turn_text == neutral.ai_turn_text


def test_nervous_affect_adds_reassuring_lead_in() -> None:
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(InterviewerActionType.PROBE_DEEPER, ack=AcknowledgementStyle.NEUTRAL)
    base = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text="Q?")
    nervous = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", affect=Affect.NERVOUS
    )
    # A warmer lead-in is prepended; the question/probe is preserved verbatim.
    assert nervous.ai_turn_text != base.ai_turn_text
    assert len(nervous.ai_turn_text) > len(base.ai_turn_text)
    assert nervous.question_or_probe == base.question_or_probe


def test_nervous_affect_bilingual() -> None:
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(InterviewerActionType.PROBE_DEEPER, ack=AcknowledgementStyle.NEUTRAL)
    vi = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="vi", question_text="Q?", affect=Affect.NERVOUS
    )
    en = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", affect=Affect.NERVOUS
    )
    # The lead-in is localized (EN and VI differ) and never double-spaces.
    assert vi.ai_turn_text != en.ai_turn_text
    assert "  " not in vi.ai_turn_text
    assert "  " not in en.ai_turn_text
