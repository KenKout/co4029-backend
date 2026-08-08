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


# ── communication polish: time-pressure + recovery lead-ins (Slice 20) ───────


def test_time_pressure_lead_in_prepended_when_enabled() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    base = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text="Q?")
    pressed = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", time_pressure=True
    )
    # A short "we're short on time, let's prioritise" lead-in is prepended; the
    # question/probe is preserved verbatim.
    assert pressed.ai_turn_text != base.ai_turn_text
    assert len(pressed.ai_turn_text) > len(base.ai_turn_text)
    assert pressed.question_or_probe == base.question_or_probe


def test_time_pressure_lead_in_bilingual() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    en = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", time_pressure=True
    )
    vi = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="vi", question_text="Q?", time_pressure=True
    )
    assert en.ai_turn_text != vi.ai_turn_text  # localized
    assert "  " not in en.ai_turn_text
    assert "  " not in vi.ai_turn_text


def test_recovery_lead_in_prepended_when_enabled() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    base = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text="Q?")
    recover = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", recovery=True
    )
    assert recover.ai_turn_text != base.ai_turn_text
    assert len(recover.ai_turn_text) > len(base.ai_turn_text)
    assert recover.question_or_probe == base.question_or_probe


def test_recovery_lead_in_bilingual() -> None:
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    en = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", recovery=True
    )
    vi = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="vi", question_text="Q?", recovery=True
    )
    assert en.ai_turn_text != vi.ai_turn_text
    assert "  " not in en.ai_turn_text
    assert "  " not in vi.ai_turn_text


def test_lead_in_precedence_recovery_over_time_pressure_over_affect() -> None:
    # Only ONE lead-in is ever prepended (no stacking). Precedence is
    # recovery > time_pressure > affect, so a struggling candidate is rebuilt
    # rather than also told "we're short on time" or "you're doing fine".
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    recovery_only = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", recovery=True
    )
    all_three = build_fallback_utterance(
        d,
        persona=Persona.NEUTRAL,
        language="en",
        question_text="Q?",
        recovery=True,
        time_pressure=True,
        affect=Affect.NERVOUS,
    )
    # Recovery wins outright — identical to recovery-only (no stacked lead-ins).
    assert all_three.ai_turn_text == recovery_only.ai_turn_text


def test_time_pressure_wins_over_affect() -> None:
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    tp_only = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", time_pressure=True
    )
    tp_and_affect = build_fallback_utterance(
        d,
        persona=Persona.NEUTRAL,
        language="en",
        question_text="Q?",
        time_pressure=True,
        affect=Affect.NERVOUS,
    )
    assert tp_and_affect.ai_turn_text == tp_only.ai_turn_text


def test_no_polish_flags_is_byte_for_byte_v1() -> None:
    # Parity: with neither new signal, the utterance is unchanged from v1.
    d = _decision(InterviewerActionType.TRANSITION_TOPIC, ack=AcknowledgementStyle.NEUTRAL)
    base = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text="Q?")
    same = build_fallback_utterance(
        d,
        persona=Persona.NEUTRAL,
        language="en",
        question_text="Q?",
        time_pressure=False,
        recovery=False,
    )
    assert base.ai_turn_text == same.ai_turn_text


# ── laddered hints & rephrasing variants (Slice 11) ──────────────────────────


def _hint_leak_guard(text: str) -> None:
    lowered = text.lower()
    for banned in ("the answer is", "correct answer", "you should say"):
        assert banned not in lowered


def test_hint_ladder_escalates_and_stays_answer_safe() -> None:
    """Each hint level yields a DIFFERENT, answer-safe EN hint."""
    d = _decision(
        InterviewerActionType.PROVIDE_NEUTRAL_HINT, reason=ReasonCode.STUDENT_REQUESTED_HINT
    )
    texts = []
    for level in (0, 1, 2):
        u = build_fallback_utterance(
            d, persona=Persona.NEUTRAL, language="en", question_text=None, hint_level=level
        )
        assert u.question_or_probe.strip()
        _hint_leak_guard(u.ai_turn_text)
        texts.append(u.question_or_probe)
    assert len(set(texts)) == 3  # three distinct escalating hints


def test_hint_ladder_bilingual() -> None:
    d = _decision(
        InterviewerActionType.PROVIDE_NEUTRAL_HINT, reason=ReasonCode.STUDENT_REQUESTED_HINT
    )
    vi = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="vi", question_text=None, hint_level=1
    )
    assert vi.question_or_probe.strip()
    _hint_leak_guard(vi.ai_turn_text)


def test_hint_level_zero_matches_v1_default() -> None:
    """Parity: level 0 (default) is the original single flat hint."""
    d = _decision(
        InterviewerActionType.PROVIDE_NEUTRAL_HINT, reason=ReasonCode.STUDENT_REQUESTED_HINT
    )
    base = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
    lvl0 = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text=None, hint_level=0
    )
    assert base.ai_turn_text == lvl0.ai_turn_text


def test_reframe_variants_differ_by_count() -> None:
    d = _decision(
        InterviewerActionType.REFRAME_QUESTION, reason=ReasonCode.STUDENT_REQUESTED_CLARIFICATION
    )
    v0 = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text=None, reframe_count=0
    )
    v1 = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text=None, reframe_count=1
    )
    assert v0.ai_turn_text != v1.ai_turn_text


# ── rich closing templates (Slice 13) ────────────────────────────────────────


def test_closing_substeps_render_bilingual_and_answer_safe() -> None:
    """Self-reflection / invite-questions / answer-question all render EN + VI."""
    for action in (
        InterviewerActionType.PROMPT_SELF_REFLECTION,
        InterviewerActionType.INVITE_CANDIDATE_QUESTIONS,
        InterviewerActionType.ANSWER_CANDIDATE_QUESTION,
    ):
        d = _decision(action, reason=ReasonCode.CLOSING_REQUIRED)
        en = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="en", question_text=None)
        vi = build_fallback_utterance(d, persona=Persona.NEUTRAL, language="vi", question_text=None)
        assert en.ai_turn_text.strip()
        assert vi.ai_turn_text.strip()
        assert en.ai_turn_text != vi.ai_turn_text  # localized
        # Never leaks rubric/answer content.
        for text in (en.ai_turn_text.lower(), vi.ai_turn_text.lower()):
            for banned in ("the answer is", "correct answer", "rubric", "score is"):
                assert banned not in text


# ── terse lead-in must not contradict an advance (live-transcript bug) ────────


def test_terse_lead_in_suppressed_when_the_turn_advances() -> None:
    # Observed live: "Feel free to expand. Thank you. Now let's move on to the
    # next question." — the terse lead-in invites elaboration on a turn that has
    # already closed the chance to elaborate. The ack table and the transition
    # table do not know about each other; suppress the invite when advancing.
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(
        InterviewerActionType.TRANSITION_TOPIC,
        ack=AcknowledgementStyle.NEUTRAL,
        reason=ReasonCode.OUTCOME_NOT_COVERED,
    )
    d.should_advance_question = True
    advancing = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", affect=Affect.TERSE
    )
    assert "Feel free to expand" not in advancing.ai_turn_text
    for lang, phrase in (("en", "Feel free to expand"), ("vi", "trình bày thêm")):
        out = build_fallback_utterance(
            d, persona=Persona.NEUTRAL, language=lang, question_text="Q?", affect=Affect.TERSE
        )
        assert phrase not in out.ai_turn_text


def test_terse_lead_in_kept_when_staying_on_the_same_question() -> None:
    # Staying on the question is exactly when "say more" is useful — must survive.
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(InterviewerActionType.PROBE_DEEPER, ack=AcknowledgementStyle.NEUTRAL)
    d.should_advance_question = False
    staying = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", affect=Affect.TERSE
    )
    assert "Feel free to expand" in staying.ai_turn_text


def test_other_affect_lead_ins_survive_an_advance() -> None:
    # Only the "expand" invite contradicts an advance. Reassurance does not.
    from abridgeai.features.interviews.orchestrator.affect import Affect

    d = _decision(
        InterviewerActionType.TRANSITION_TOPIC,
        ack=AcknowledgementStyle.NEUTRAL,
        reason=ReasonCode.OUTCOME_NOT_COVERED,
    )
    d.should_advance_question = True
    nervous = build_fallback_utterance(
        d, persona=Persona.NEUTRAL, language="en", question_text="Q?", affect=Affect.NERVOUS
    )
    assert "No rush" in nervous.ai_turn_text


# ── clarify must rephrase, not interrogate the confused candidate ─────────────


def test_clarify_does_not_ask_the_candidate_which_part() -> None:
    # Live transcript: candidate said "I don't understand the question" and got
    # "Which part of the question would you like me to clarify?" — twice. Asking a
    # confused candidate to self-diagnose is the opposite of guiding them. The
    # scaffold must offer a rephrasing instead.
    for lang, banned in (
        ("en", ("which part", "which specific part")),
        ("vi", ("phần nào",)),
    ):
        d = _decision(
            InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
            reason=ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
        )
        out = build_fallback_utterance(
            d, persona=Persona.NEUTRAL, language=lang, question_text=None
        )
        lowered = out.ai_turn_text.lower()
        for phrase in banned:
            assert phrase not in lowered, f"{lang}: still asking the candidate to self-diagnose"
        assert lowered.strip(), f"{lang}: clarify produced no text"


def test_clarify_stays_answer_safe() -> None:
    d = _decision(
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        reason=ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
    )
    for lang in ("en", "vi"):
        out = build_fallback_utterance(
            d, persona=Persona.NEUTRAL, language=lang, question_text=None
        )
        lowered = out.ai_turn_text.lower()
        for banned in ("the answer is", "correct answer", "rubric", "score is"):
            assert banned not in lowered
