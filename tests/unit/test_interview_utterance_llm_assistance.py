"""Regression tests for LLM utterance phrasing on ASSISTANCE turns.

Two bugs are pinned here, both observed in a live interview where a candidate
said "I'm not sure" and received the generic template
"A small hint: organize your answer around the main concepts in the question…" —
a hint that cannot possibly reference the question, because:

  1. ``validated_rewrite`` required the fallback's ``question_or_probe`` to
     survive VERBATIM. That is correct when the text IS an exam question from the
     bank (the selector is authoritative and the model must not reword it, which
     would change difficulty/fairness). It is wrong on a HINT / CLARIFY / REFRAME
     turn, where ``question_or_probe`` is a CANNED TEMPLATE — so every genuinely
     rephrased hint was rejected and the template shipped. The LLM call was
     billed and thrown away, which is also why ``utterance_status`` sat
     permanently at "fallback" and was useless as a signal.

  2. The prompt payload never carried the actual interview question, so even an
     accepted rewrite had nothing to ground the hint in.

The guard is keyed on DATA (did this text come from the question bank?) rather
than on a list of action types, so it cannot drift as new actions are added.

These tests exercise the real ``validated_rewrite`` / ``build_prompt_payload``
rather than a copy, so a regression cannot pass by re-implementing the rule.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.decision import (
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.utterance import Utterance
from abridgeai.features.interviews.orchestrator.utterance_logic import (
    build_prompt_payload,
    validated_rewrite,
)

_QUESTION = (
    "What is the primary difference between operational processing and "
    "information processing in an organizational context?"
)
_HINT_TEMPLATE = (
    "A small hint: organize your answer around the main concepts in the question "
    "and how they relate."
)


def _decision(action: InterviewerActionType) -> InterviewerDecision:
    return InterviewerDecision(action=action, reason_code=ReasonCode.PARTIAL_OUTCOME_COVERAGE)


def _hint_fallback() -> Utterance:
    return Utterance(
        acknowledgement="",
        transition="",
        question_or_probe=_HINT_TEMPLATE,
        ai_turn_text=_HINT_TEMPLATE,
    )


def _advance_fallback() -> Utterance:
    return Utterance(
        acknowledgement="Thank you.",
        transition="Now let's move on.",
        question_or_probe=_QUESTION,
        ai_turn_text=f"Thank you. Now let's move on. {_QUESTION}",
    )


# ── Bug 1: a template is not authoritative, so a rewrite may replace it ───────


def test_hint_rewrite_accepted_when_text_is_not_a_bank_question() -> None:
    rewritten = validated_rewrite(
        {
            "ai_turn_text": (
                "Think about who uses each one day to day: one keeps the business "
                "running transaction by transaction, the other looks back across "
                "many transactions to inform a decision."
            )
        },
        _hint_fallback(),
        require_verbatim=False,
    )
    assert rewritten is not None, "LLM hint rejected — the verbatim guard fired on a template"
    assert "A small hint" not in rewritten.ai_turn_text


def test_hint_rewrite_still_rejected_when_empty() -> None:
    assert (
        validated_rewrite({"ai_turn_text": "   "}, _hint_fallback(), require_verbatim=False) is None
    )
    assert validated_rewrite({}, _hint_fallback(), require_verbatim=False) is None


# ── The guard MUST still hold for an authoritative bank question ──────────────


def test_advance_rejects_a_reworded_exam_question() -> None:
    rewritten = validated_rewrite(
        {"ai_turn_text": "So, tell me how operational and informational systems differ."},
        _advance_fallback(),
        require_verbatim=True,
    )
    assert rewritten is None, "a reworded exam question was accepted"


def test_advance_accepts_rewrite_that_keeps_the_question_verbatim() -> None:
    rewritten = validated_rewrite(
        {"ai_turn_text": f"Great, thanks for that. Here's the next one. {_QUESTION}"},
        _advance_fallback(),
        require_verbatim=True,
    )
    assert rewritten is not None
    assert _QUESTION in rewritten.ai_turn_text


def test_answer_leak_length_guard_applies_to_both_modes() -> None:
    fallback = _hint_fallback()
    bloated = "x" * (len(fallback.ai_turn_text) * 3 + 500)
    assert validated_rewrite({"ai_turn_text": bloated}, fallback, require_verbatim=False) is None
    assert (
        validated_rewrite({"ai_turn_text": bloated}, _advance_fallback(), require_verbatim=True)
        is None
    )


# ── Bug 2: the model must be told which question it is assisting with ─────────


def test_payload_grounds_a_hint_in_the_question_being_answered() -> None:
    # THE case the bug report is about. On a hint turn the authoritative text
    # (`question_text`) is None — `probe_seed_text` only returns a value for
    # REPEAT_QUESTION — so the question must reach the model through
    # `grounding_question` or the hint has nothing to be about.
    payload = build_prompt_payload(
        _decision(InterviewerActionType.PROVIDE_NEUTRAL_HINT),
        fallback=_hint_fallback(),
        persona_value="supportive",
        persona_traits={},
        language="en",
        question_text=None,
        grounding_question=_QUESTION,
        hint_level=1,
    )
    assert payload.get("current_question") == _QUESTION, (
        "hint prompt has no question to ground itself in"
    )
    assert payload.get("hint_level") == 1, "escalation rung not passed to the model"
    approved = payload.get("approved_parts")
    assert isinstance(approved, dict)
    assert approved.get("question_or_probe") == _HINT_TEMPLATE


def test_payload_falls_back_to_the_authoritative_text_for_grounding() -> None:
    # On an advance the bank question is both authoritative AND the grounding
    # context, so it must still appear without a separate grounding argument.
    payload = build_prompt_payload(
        _decision(InterviewerActionType.TRANSITION_TOPIC),
        fallback=_advance_fallback(),
        persona_value="neutral",
        persona_traits={},
        language="en",
        question_text=_QUESTION,
        grounding_question=None,
        hint_level=0,
    )
    assert payload.get("current_question") == _QUESTION
    assert "hint_level" not in payload


def test_payload_omits_question_when_nothing_is_known() -> None:
    payload = build_prompt_payload(
        _decision(InterviewerActionType.HANDLE_TECHNICAL_ISSUE),
        fallback=_hint_fallback(),
        persona_value="neutral",
        persona_traits={},
        language="en",
        question_text=None,
        grounding_question=None,
        hint_level=0,
    )
    assert "current_question" not in payload
    assert "hint_level" not in payload
