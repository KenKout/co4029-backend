"""Natural interviewer utterance generation (Phase 9).

Turns the authoritative deterministic :class:`InterviewerDecision` into what the
interviewer actually *says*. The decision (action, reason code, selected
question, advance/end/record flags) is the source of truth; this layer only
phrases it. An optional LLM pass may rewrite the phrasing for naturalness, but
it can NEVER change the decision — if it fails, returns garbage, or is disabled,
we fall back to a deterministic per-action, per-persona template.

Structure (safeguard #5): every utterance is decomposed into
``acknowledgement`` + ``transition`` + ``question_or_probe``, plus a combined
``ai_turn_text``. The canonical response mapper (Slice 4) decides how those map
onto legacy vs. new response fields so no client renders the question twice.

Persona (requirement #7) shapes *language only* — tone, acknowledgement
frequency, directness — never scoring or passing logic:

* strict     — concise, direct, evidence-seeking, professional, never hostile.
* neutral    — balanced, professional, efficient.
* supportive — warm, encouraging, gently scaffolded, never reveals answers.

Bilingual: every template ships EN + VI; ``language`` selects. Unknown
languages fall back to EN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    InterviewerActionType,
    InterviewerDecision,
)


class Persona(str, Enum):  # noqa: UP042 -- match codebase convention
    STRICT = "strict"
    NEUTRAL = "neutral"
    SUPPORTIVE = "supportive"


def persona_from(value: str | None) -> Persona:
    try:
        return Persona(value)
    except (ValueError, TypeError):
        return Persona.NEUTRAL


def _lang(language: str | None) -> str:
    """Normalize a language tag to 'vi' or 'en' (default en)."""
    if language and language.strip().lower().startswith("vi"):
        return "vi"
    return "en"


@dataclass(frozen=True)
class Utterance:
    """The decomposed interviewer utterance (safeguard #5).

    ``question_or_probe`` is the actual question/probe text (or "" for actions
    that only acknowledge/transition, e.g. handle_technical_issue). ``ai_turn_text``
    is the combined natural utterance the new client can render directly.
    """

    acknowledgement: str
    transition: str
    question_or_probe: str
    ai_turn_text: str

    def acknowledgement_and_transition(self) -> str:
        """Ack + transition ONLY (no question) — for legacy ai_followup_text on advance."""
        return " ".join(p for p in (self.acknowledgement, self.transition) if p).strip()


# ── Acknowledgement snippets by persona + style + language ───────────────────
# Neutral acknowledgements only — never declare correctness when uncertain
# (requirement #8). Supportive is warmer; strict is sparse.
_ACK: dict[tuple[Persona, AcknowledgementStyle, str], str] = {
    # neutral style
    (Persona.STRICT, AcknowledgementStyle.NEUTRAL, "en"): "Noted.",
    (Persona.STRICT, AcknowledgementStyle.NEUTRAL, "vi"): "Đã ghi nhận.",
    (Persona.NEUTRAL, AcknowledgementStyle.NEUTRAL, "en"): "Thank you.",
    (Persona.NEUTRAL, AcknowledgementStyle.NEUTRAL, "vi"): "Cảm ơn bạn.",
    (Persona.SUPPORTIVE, AcknowledgementStyle.NEUTRAL, "en"): "Thanks for sharing that.",
    (Persona.SUPPORTIVE, AcknowledgementStyle.NEUTRAL, "vi"): "Cảm ơn bạn đã chia sẻ.",
    # positive style (still not declaring correctness)
    (Persona.STRICT, AcknowledgementStyle.POSITIVE, "en"): "Good.",
    (Persona.STRICT, AcknowledgementStyle.POSITIVE, "vi"): "Tốt.",
    (Persona.NEUTRAL, AcknowledgementStyle.POSITIVE, "en"): "That's helpful.",
    (Persona.NEUTRAL, AcknowledgementStyle.POSITIVE, "vi"): "Điều đó hữu ích.",
    (Persona.SUPPORTIVE, AcknowledgementStyle.POSITIVE, "en"): "That's a solid start, well done.",
    (Persona.SUPPORTIVE, AcknowledgementStyle.POSITIVE, "vi"): "Một khởi đầu tốt, làm tốt lắm.",
    # corrective style (neutral, non-shaming)
    (Persona.STRICT, AcknowledgementStyle.CORRECTIVE, "en"): "I understand your reasoning.",
    (Persona.STRICT, AcknowledgementStyle.CORRECTIVE, "vi"): "Tôi hiểu lập luận của bạn.",
    (Persona.NEUTRAL, AcknowledgementStyle.CORRECTIVE, "en"): "I understand your reasoning.",
    (Persona.NEUTRAL, AcknowledgementStyle.CORRECTIVE, "vi"): "Tôi hiểu cách bạn nghĩ.",
    (Persona.SUPPORTIVE, AcknowledgementStyle.CORRECTIVE, "en"): "I see your direction.",
    (Persona.SUPPORTIVE, AcknowledgementStyle.CORRECTIVE, "vi"): "Tôi thấy hướng bạn đang đi.",
}


def _ack_text(persona: Persona, style: AcknowledgementStyle, lang: str) -> str:
    if style is AcknowledgementStyle.NONE:
        return ""
    return _ACK.get((persona, style, lang), _ACK.get((Persona.NEUTRAL, style, "en"), ""))


# ── Transition / action phrasing by action + persona + language ──────────────
# Each entry is a template string. {q} is substituted with the selected question
# or probe text. Where the action carries no question (technical issue, pause),
# the template is self-contained.

# Transitions used when advancing to a new question.
_TRANSITION: dict[tuple[Persona, str], str] = {
    (Persona.STRICT, "en"): "Next question.",
    (Persona.STRICT, "vi"): "Câu hỏi tiếp theo.",
    (Persona.NEUTRAL, "en"): "Let's move on.",
    (Persona.NEUTRAL, "vi"): "Chúng ta tiếp tục nhé.",
    (Persona.SUPPORTIVE, "en"): "Let's move on to the next one.",
    (Persona.SUPPORTIVE, "vi"): "Chúng ta chuyển sang câu tiếp theo nhé.",
}


def _fallback_parts(  # noqa: C901 -- flat per-action dispatch; readability > splitting
    decision: InterviewerDecision,
    persona: Persona,
    lang: str,
    *,
    question_text: str | None,
) -> tuple[str, str, str]:
    """Deterministic (acknowledgement, transition, question_or_probe) per action.

    This is the guaranteed fallback (requirement #9): every action type has a
    persona-aware, bilingual template so a failed LLM call never blocks the turn.
    """
    action = decision.action
    ack = _ack_text(persona, decision.acknowledgement_style, lang)
    q = (question_text or "").strip()

    # Actions that carry the selected question / probe verbatim.
    if action in (
        InterviewerActionType.ASK_MAIN_QUESTION,
        InterviewerActionType.TRANSITION_TOPIC,
        InterviewerActionType.SKIP_QUESTION,
    ):
        transition = _TRANSITION.get((persona, lang), _TRANSITION[(Persona.NEUTRAL, "en")])
        return ack, transition, q

    if action in (
        InterviewerActionType.PROBE_DEEPER,
        InterviewerActionType.ASK_FOR_EXAMPLE,
        InterviewerActionType.CHALLENGE_REASONING,
        InterviewerActionType.EXPLORE_TRADEOFF,
        InterviewerActionType.RESOLVE_CONTRADICTION,
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        InterviewerActionType.PROVIDE_NEUTRAL_HINT,
        InterviewerActionType.REFRAME_QUESTION,
    ):
        # Probe text is supplied by the caller (from analysis / a reframing);
        # when absent we ask a generic, answer-safe probe.
        probe = q or _generic_probe(action, persona, lang)
        return ack, "", probe

    if action is InterviewerActionType.REPEAT_QUESTION:
        prefix = {
            "en": "Of course. Here is the question again:",
            "vi": "Tất nhiên. Câu hỏi được nhắc lại:",
        }[lang]
        return "", prefix, q

    if action is InterviewerActionType.REDIRECT_TO_TOPIC:
        redirect = {
            "en": "Let's refocus on the question.",
            "vi": "Chúng ta hãy tập trung lại vào câu hỏi.",
        }[lang]
        return ack, redirect, q

    if action is InterviewerActionType.OFFER_BRIEF_PAUSE:
        pause = {
            "en": "Take a moment to think — I'll wait.",
            "vi": "Bạn cứ suy nghĩ một chút — tôi sẽ đợi.",
        }[lang]
        return ack, pause, ""

    if action is InterviewerActionType.HANDLE_TECHNICAL_ISSUE:
        tech = {
            "en": "No problem — take your time to sort that out, then we'll continue.",
            "vi": "Không sao — bạn cứ xử lý vấn đề đó, rồi chúng ta tiếp tục.",
        }[lang]
        return "", tech, ""

    if action is InterviewerActionType.OPENING:
        return "", "", q

    if action in (InterviewerActionType.BEGIN_CLOSING, InterviewerActionType.CLOSE_INTERVIEW):
        closing = {
            "en": "Thank you. That concludes the interview.",
            "vi": "Cảm ơn bạn. Buổi phỏng vấn kết thúc tại đây.",
        }[lang]
        return ack, "", closing

    if action is InterviewerActionType.ACKNOWLEDGE:
        return ack, "", q

    # Unknown / unmapped action → neutral, safe.
    return ack, "", q


def _generic_probe(action: InterviewerActionType, persona: Persona, lang: str) -> str:
    """Answer-safe generic probe when no specific probe text is available.

    NEVER reveals expected content (requirement #8) — asks the student to expand.
    """
    table: dict[tuple[InterviewerActionType, str], str] = {
        (InterviewerActionType.ASK_FOR_EXAMPLE, "en"): "Could you give a concrete example?",
        (InterviewerActionType.ASK_FOR_EXAMPLE, "vi"): "Bạn có thể cho một ví dụ cụ thể không?",
        (InterviewerActionType.PROBE_DEEPER, "en"): "Could you explain your reasoning further?",
        (
            InterviewerActionType.PROBE_DEEPER,
            "vi",
        ): "Bạn có thể giải thích rõ hơn lập luận của mình không?",  # noqa: E501
        (InterviewerActionType.CHALLENGE_REASONING, "en"): "What makes you confident in that?",
        (InterviewerActionType.CHALLENGE_REASONING, "vi"): "Điều gì khiến bạn tự tin về điều đó?",
        (InterviewerActionType.EXPLORE_TRADEOFF, "en"): "What trade-offs would that involve?",
        (InterviewerActionType.EXPLORE_TRADEOFF, "vi"): "Điều đó sẽ có những đánh đổi gì?",
        (InterviewerActionType.RESOLVE_CONTRADICTION, "en"): (
            "Earlier you said something that seems different — can you reconcile the two?"
        ),
        (InterviewerActionType.RESOLVE_CONTRADICTION, "vi"): (
            "Trước đó bạn nói điều có vẻ khác — bạn có thể dung hòa hai ý đó không?"
        ),
        (InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER, "en"): (
            "Which part of the question would you like me to rephrase?"
        ),
        (InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER, "vi"): (
            "Bạn muốn tôi diễn đạt lại phần nào của câu hỏi?"
        ),
        (InterviewerActionType.PROVIDE_NEUTRAL_HINT, "en"): (
            "A small hint: organize your answer around the main concepts in the question "
            "and how they relate."
        ),
        (InterviewerActionType.PROVIDE_NEUTRAL_HINT, "vi"): (
            "Gợi ý nhỏ: hãy sắp xếp câu trả lời theo các khái niệm chính trong câu hỏi "
            "và mối quan hệ giữa chúng."
        ),
        (InterviewerActionType.REFRAME_QUESTION, "en"): "Let me put the question another way.",
        (InterviewerActionType.REFRAME_QUESTION, "vi"): "Để tôi diễn đạt câu hỏi theo cách khác.",
    }
    return table.get((action, lang), table.get((action, "en"), "Could you say more?"))


def _combine(acknowledgement: str, transition: str, question_or_probe: str) -> str:
    """Join the parts into one natural utterance, single-spaced, no doubling."""
    return " ".join(p for p in (acknowledgement, transition, question_or_probe) if p).strip()


def build_fallback_utterance(
    decision: InterviewerDecision,
    *,
    persona: Persona,
    language: str | None,
    question_text: str | None = None,
) -> Utterance:
    """Deterministic, persona-aware, bilingual utterance for a decision.

    This is the guaranteed path — no I/O, never fails. The LLM phrasing layer
    (added in the logic module) wraps this and falls back to it on any error.
    """
    lang = _lang(language)
    ack, transition, qp = _fallback_parts(decision, persona, lang, question_text=question_text)
    return Utterance(
        acknowledgement=ack,
        transition=transition,
        question_or_probe=qp,
        ai_turn_text=_combine(ack, transition, qp),
    )


__all__ = [
    "Persona",
    "Utterance",
    "build_fallback_utterance",
    "persona_from",
]
