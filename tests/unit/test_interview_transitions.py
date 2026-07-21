"""Unit tests for standardized Natural Interview Transitions wording.

Pure — no DB, no LLM. Covers the persona-aware EN/VI transition_text() used by
both the legacy-mode deterministic advance and transcript persistence, plus the
repeat / clarify / follow-up signposts in the deterministic fallback utterance.
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.utterance import (
    Persona,
    build_fallback_utterance,
    transition_text,
)


def _decision(
    action: InterviewerActionType,
    *,
    ack: AcknowledgementStyle = AcknowledgementStyle.NONE,
    reason: ReasonCode = ReasonCode.PARTIAL_OUTCOME_COVERAGE,
) -> InterviewerDecision:
    return InterviewerDecision(action=action, reason_code=reason, acknowledgement_style=ack)


@pytest.mark.parametrize("persona", list(Persona))
def test_next_question_transition_is_thankful_and_signposts_move_on(persona: Persona) -> None:
    en = transition_text(persona, "en")
    vi = transition_text(persona, "vi")
    # EN: thanks + explicit next-question signpost, never the question itself.
    assert en.lower().startswith("thank you")
    assert "next question" in en.lower()
    # VI: thanks + move-on signpost.
    assert "cảm ơn" in vi.lower()
    assert "câu hỏi tiếp theo" in vi.lower()


@pytest.mark.parametrize("persona", list(Persona))
def test_final_question_transition_matches_spec_wording(persona: Persona) -> None:
    en = transition_text(persona, "en", final=True)
    vi = transition_text(persona, "vi", final=True)
    assert "that was the final question" in en.lower()
    assert "câu hỏi cuối cùng" in vi.lower()


def test_unknown_language_falls_back_to_english_neutral() -> None:
    assert transition_text(Persona.NEUTRAL, "fr") == transition_text(Persona.NEUTRAL, "en")
    assert transition_text(Persona.STRICT, None).lower().startswith("thank you")


def test_transition_never_contains_a_question_mark() -> None:
    # A transition is a signpost only; the question rides in its own field.
    for persona in Persona:
        for lang in ("en", "vi"):
            assert "?" not in transition_text(persona, lang)
            assert "?" not in transition_text(persona, lang, final=True)


def test_repeat_signpost_is_standardized_and_keeps_question_verbatim() -> None:
    d = _decision(InterviewerActionType.REPEAT_QUESTION, reason=ReasonCode.STUDENT_REQUESTED_REPEAT)
    en = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="What is a star schema?"
    )
    vi = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="vi", question_text="Star schema là gì?"
    )
    assert "i'll repeat the question" in en.ai_turn_text.lower()
    assert "What is a star schema?" in en.question_or_probe  # verbatim, unchanged
    assert "nhắc lại câu hỏi" in vi.ai_turn_text.lower()


def test_clarify_signpost_is_standardized_rephrase() -> None:
    d = _decision(
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        reason=ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
    )
    en = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    vi = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="vi", question_text=None)
    assert "let me rephrase the question" in en.ai_turn_text.lower()
    assert "diễn đạt lại câu hỏi" in vi.ai_turn_text.lower()


def test_followup_signpost_does_not_claim_correctness() -> None:
    d = _decision(
        InterviewerActionType.PROBE_DEEPER,
        ack=AcknowledgementStyle.NEUTRAL,
        reason=ReasonCode.PARTIAL_OUTCOME_COVERAGE,
    )
    u = build_fallback_utterance(
        d, persona=Persona.SUPPORTIVE, language="en", question_text="Why did you choose that index?"
    )
    lowered = u.ai_turn_text.lower()
    # Neutral follow-up lead-in; must not assert the answer was right.
    assert "follow up" in lowered
    for forbidden in ("correct", "right answer", "that's right", "well done"):
        assert forbidden not in lowered


def test_hint_signpost_precedes_safe_assistance() -> None:
    d = _decision(
        InterviewerActionType.PROVIDE_NEUTRAL_HINT,
        reason=ReasonCode.STUDENT_REQUESTED_HINT,
    )
    en = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    assert "hint" in en.ai_turn_text.lower()
    assert en.question_or_probe  # the safe assistance body is present
