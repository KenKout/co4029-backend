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

# Transitions used when advancing to a new question. Standardized natural
# wording (Natural Interview Transitions spec): a brief thanks + an explicit
# "move on to the next question" signpost, persona-shaped in tone only. The
# question text itself is appended separately (never duplicated here).
_TRANSITION: dict[tuple[Persona, str], str] = {
    # Deliberately NO leading "Thank you." here: the acknowledgement (_ACK)
    # already opens with a thanks ("Thank you." / "Cảm ơn bạn."), and _combine
    # concatenates ack + transition, which produced an audible/visible
    # "Thank you. Thank you. Now let's move on…" double. The signpost keeps only
    # the "move on" wording; the ack carries the thanks.
    (Persona.STRICT, "en"): "Let's move on to the next question.",
    (Persona.STRICT, "vi"): "Chúng ta sang câu hỏi tiếp theo.",
    (Persona.NEUTRAL, "en"): "Now let's move on to the next question.",
    (Persona.NEUTRAL, "vi"): "Bây giờ chúng ta chuyển sang câu hỏi tiếp theo.",
    (Persona.SUPPORTIVE, "en"): ("Now let's move on to the next question together."),
    (Persona.SUPPORTIVE, "vi"): (
        "Bây giờ chúng ta cùng chuyển sang câu hỏi tiếp theo nhé."
    ),
}

# Final-question transition (spec §ending): a short acknowledgment that the
# last question has been reached. This is a transition-only turn; the separate
# goodbye/closing turn follows via the existing finish flow.
_FINAL_QUESTION_TRANSITION: dict[tuple[Persona, str], str] = {
    # Deliberately NO leading "Thank you." here: this final-question transition
    # is always immediately followed by the closing turn, which itself opens
    # with "Thank you." Keeping both produced an audible "Thank you… Thank you…"
    # double. The closing keeps its thanks so the skip/timeout paths (which have
    # no preceding transition) still thank the candidate.
    (Persona.STRICT, "en"): "That was the final question.",
    (Persona.STRICT, "vi"): "Đó là câu hỏi cuối cùng.",
    (Persona.NEUTRAL, "en"): "That was the final question.",
    (Persona.NEUTRAL, "vi"): "Đó là câu hỏi cuối cùng.",
    (Persona.SUPPORTIVE, "en"): "That was the final question — well done.",
    (Persona.SUPPORTIVE, "vi"): "Đó là câu hỏi cuối cùng — bạn đã làm rất tốt.",
}


def transition_text(persona: Persona, language: str | None, *, final: bool = False) -> str:
    """Standardized persona-aware EN/VI transition wording.

    ``final=True`` returns the final-question acknowledgment (spec §ending);
    otherwise the next-question signpost. Reused by both the legacy-mode
    deterministic path and transcript persistence so wording stays identical.
    """
    lang = _lang(language)
    table = _FINAL_QUESTION_TRANSITION if final else _TRANSITION
    return table.get((persona, lang), table[(Persona.NEUTRAL, "en")])


def _fallback_parts(  # noqa: C901, PLR0911 -- flat per-action dispatch; readability > splitting
    decision: InterviewerDecision,
    persona: Persona,
    lang: str,
    *,
    question_text: str | None,
    hint_level: int = 0,
    reframe_count: int = 0,
) -> tuple[str, str, str]:
    """Deterministic (acknowledgement, transition, question_or_probe) per action.

    This is the guaranteed fallback (requirement #9): every action type has a
    persona-aware, bilingual template so a failed LLM call never blocks the turn.

    ``hint_level`` / ``reframe_count`` (Slice 11, v2) select an escalating hint
    or a rephrasing variant; both default to 0 → the original v1 wording.
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
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        InterviewerActionType.REFRAME_QUESTION,
    ):
        # Spec §wording: clarification/rephrase signpost precedes the safe,
        # answer-preserving rephrasing. The current question is never advanced
        # or scored by this control turn. Slice 11: vary the signpost by
        # reframe_count so a repeated reframe never repeats verbatim.
        signpost = _reframe_signpost(reframe_count, lang)
        probe = q or _generic_probe(action, persona, lang)
        return ack, signpost, probe

    if action is InterviewerActionType.PROVIDE_NEUTRAL_HINT:
        # Laddered hint (Slice 11): escalate neutral nudge → structural →
        # worked-approach by hint_level, NEVER revealing the answer.
        signpost = _probe_signpost(action, persona, lang)
        probe = q or _laddered_hint(hint_level, lang)
        return ack, signpost, probe

    if action in (
        InterviewerActionType.PROBE_DEEPER,
        InterviewerActionType.ASK_FOR_EXAMPLE,
        InterviewerActionType.CHALLENGE_REASONING,
        InterviewerActionType.EXPLORE_TRADEOFF,
        InterviewerActionType.RESOLVE_CONTRADICTION,
        InterviewerActionType.EXTEND_ANSWER,
        InterviewerActionType.PROBE_EDGE_CASE,
    ):
        # Hint/term explanation and follow-up probes get a short natural
        # signpost before the safe assistance (spec §wording). The signpost
        # acknowledges without implying the previous answer was correct.
        signpost = _probe_signpost(action, persona, lang)
        probe = q or _generic_probe(action, persona, lang)
        return ack, signpost, probe

    if action is InterviewerActionType.REPEAT_QUESTION:
        # Spec §wording: repeat signpost is a fixed natural phrase; the current
        # question follows verbatim and unchanged (never advances or scores).
        prefix = {
            "en": "Of course. I'll repeat the question.",
            "vi": "Tất nhiên. Tôi sẽ nhắc lại câu hỏi.",
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

    if action is InterviewerActionType.REQUEST_END_CONFIRMATION:
        # End-confirmation gate (Slice 4): ask the candidate to confirm ending
        # rather than closing immediately. Keeps the current question in play.
        confirm = {
            "en": "Just to confirm — would you like to end and submit for grading, or continue the interview?",  # noqa: E501
            "vi": "Xin xác nhận — bạn muốn kết thúc và nộp bài để chấm điểm, hay tiếp tục buổi phỏng vấn?",  # noqa: E501
        }[lang]
        return "", confirm, ""

    if action is InterviewerActionType.CANCEL_END:
        resume = {
            "en": "No problem — let's continue.",
            "vi": "Không sao — chúng ta tiếp tục nhé.",
        }[lang]
        return "", resume, q

    if action is InterviewerActionType.OPENING:
        return "", "", q

    if action in (InterviewerActionType.BEGIN_CLOSING, InterviewerActionType.CLOSE_INTERVIEW):
        closing = {
            "en": "Thank you. That concludes the interview.",
            "vi": "Cảm ơn bạn. Buổi phỏng vấn kết thúc tại đây.",
        }[lang]
        return ack, "", closing

    # Rich closing sub-steps (Slice 13, v2). Brief, answer-safe, bilingual. Each
    # is a self-contained turn (no question metadata needed).
    if action is InterviewerActionType.PROMPT_SELF_REFLECTION:
        reflection = {
            "en": (
                "Before we wrap up: looking back on the interview, what's one thing "
                "you feel went well, and one you'd approach differently?"
            ),
            "vi": (
                "Trước khi kết thúc: nhìn lại buổi phỏng vấn, bạn thấy điều gì mình "
                "đã làm tốt, và điều gì bạn sẽ làm khác đi?"
            ),
        }[lang]
        return ack, "", reflection

    if action is InterviewerActionType.INVITE_CANDIDATE_QUESTIONS:
        invite = {
            "en": "Thank you for sharing that. Is there anything you'd like to ask me?",
            "vi": "Cảm ơn bạn đã chia sẻ. Bạn có muốn hỏi tôi điều gì không?",
        }[lang]
        return ack, "", invite

    if action is InterviewerActionType.ANSWER_CANDIDATE_QUESTION:
        # Answer-safe acknowledgement — never reveals rubric/answers or makes
        # commitments; defers specifics to the teacher/results.
        reply = {
            "en": (
                "That's a good question. I can't share the evaluation details here, "
                "but your instructor will follow up with feedback and results."
            ),
            "vi": (
                "Đó là một câu hỏi hay. Tôi không thể chia sẻ chi tiết đánh giá ở đây, "
                "nhưng giảng viên của bạn sẽ phản hồi kèm kết quả sau."
            ),
        }[lang]
        return ack, "", reply

    # Frustration de-escalation (Slice 19A, v2). Warm, encouraging, answer-safe;
    # acknowledges the feeling and resumes the SAME question (``q``). Never
    # reveals answer content.
    if action is InterviewerActionType.DEESCALATE:
        reassure = {
            "en": (
                "That's completely okay — take a breath. There's no penalty here; "
                "let's take it one step at a time."
            ),
            "vi": (
                "Không sao đâu — bạn cứ bình tĩnh. Không có điểm trừ gì cả; "
                "chúng ta cứ đi từng bước một nhé."
            ),
        }[lang]
        return reassure, "", q

    # Mid-interview question deferral (Slice 19B, v2). Briefly acknowledge the
    # candidate's question, defer it to the end, and resume the current question.
    if action is InterviewerActionType.DEFER_CANDIDATE_QUESTION:
        defer = {
            "en": (
                "Good question — let's come back to that at the end. For now, "
                "let's stay with the current one."
            ),
            "vi": (
                "Câu hỏi hay — mình sẽ quay lại cuối buổi nhé. Bây giờ, "
                "chúng ta tiếp tục với câu hiện tại."
            ),
        }[lang]
        return defer, "", q

    if action is InterviewerActionType.ACKNOWLEDGE:
        return ack, "", q

    # Unknown / unmapped action → neutral, safe.
    return ack, "", q


def _probe_signpost(action: InterviewerActionType, persona: Persona, lang: str) -> str:
    """Short natural signpost before a follow-up probe or safe assistance.

    Spec §wording: follow-ups acknowledge the answer WITHOUT implying it was
    correct, then ask the probe; hint/term explanations get a brief lead-in
    before the answer-safe assistance. Tone is persona-shaped only.
    """
    if action is InterviewerActionType.PROVIDE_NEUTRAL_HINT:
        table = {
            "en": "Here's a small hint to guide you.",
            "vi": "Đây là một gợi ý nhỏ để bạn định hướng.",
        }
        return table[lang]
    # Depth probes (Slice 8): follow a STRONG answer, so the lead-in genuinely
    # affirms before pushing further (unlike the neutral follow-up lead-in).
    if action in (
        InterviewerActionType.EXTEND_ANSWER,
        InterviewerActionType.PROBE_EDGE_CASE,
    ):
        table = {
            "en": "That's a strong answer — let's go further.",
            "vi": "Đó là một câu trả lời tốt — chúng ta hãy đi xa hơn.",
        }
        return table[lang]
    # Follow-up / deeper probing: neutral lead-in, never affirms correctness.
    followup = {
        (Persona.STRICT, "en"): "Let's dig into that.",
        (Persona.STRICT, "vi"): "Chúng ta hãy đi sâu hơn.",
        (Persona.NEUTRAL, "en"): "Let me follow up on that.",
        (Persona.NEUTRAL, "vi"): "Tôi muốn hỏi thêm về điều đó.",
        (Persona.SUPPORTIVE, "en"): "Thanks — let me follow up on that.",
        (Persona.SUPPORTIVE, "vi"): "Cảm ơn bạn — tôi muốn hỏi thêm một chút.",
    }
    return followup.get((persona, lang), followup[(Persona.NEUTRAL, "en")])


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
        (InterviewerActionType.EXTEND_ANSWER, "en"): (
            "That's solid — can you generalize it or extend it to a broader case?"
        ),
        (InterviewerActionType.EXTEND_ANSWER, "vi"): (
            "Rất tốt — bạn có thể khái quát hóa hoặc mở rộng nó cho một trường hợp rộng hơn không?"
        ),
        (InterviewerActionType.PROBE_EDGE_CASE, "en"): (
            "Where might that break down — what edge cases or failure modes should we consider?"
        ),
        (InterviewerActionType.PROBE_EDGE_CASE, "vi"): (
            "Nó có thể thất bại ở đâu — có trường hợp biên hay tình huống lỗi nào cần cân nhắc không?"  # noqa: E501
        ),
    }
    return table.get((action, lang), table.get((action, "en"), "Could you say more?"))


# Laddered hints (Slice 11, v2). Escalate by hint_level, NEVER revealing the
# answer: level 0 = neutral structural nudge (the original v1 hint), level 1 =
# stronger structural scaffold, level 2+ = a worked-approach hint (how to think,
# not what to say). Level clamps to the last rung.
_HINT_LADDER: dict[str, tuple[str, ...]] = {
    "en": (
        "A small hint: organize your answer around the main concepts in the question "
        "and how they relate.",
        "A bigger hint: break the question into its parts and address each one in turn — "
        "start with the definition, then the 'why', then an example.",
        "Let's approach it together: pick the single most central idea, state it plainly, "
        "then explain one consequence of it. You don't need the whole answer at once.",
    ),
    "vi": (
        "Gợi ý nhỏ: hãy sắp xếp câu trả lời theo các khái niệm chính trong câu hỏi "
        "và mối quan hệ giữa chúng.",
        "Gợi ý rõ hơn: hãy chia câu hỏi thành các phần và trả lời lần lượt — "
        "bắt đầu từ định nghĩa, rồi đến 'tại sao', rồi một ví dụ.",
        "Chúng ta cùng tiếp cận nhé: chọn ý trọng tâm nhất, nêu rõ ràng, "
        "rồi giải thích một hệ quả của nó. Bạn không cần trả lời hết ngay.",
    ),
}


def _laddered_hint(hint_level: int, lang: str) -> str:
    """Answer-safe hint escalating by ``hint_level`` (clamped to the last rung)."""
    rungs = _HINT_LADDER.get(lang, _HINT_LADDER["en"])
    idx = max(0, min(hint_level, len(rungs) - 1))
    return rungs[idx]


def laddered_hint(hint_level: int, language: str | None) -> str:
    """Public answer-safe laddered hint (Slice 11 upgrade).

    Shared by BOTH hint paths so escalation is identical no matter which fires:
    the adaptive decision path (via ``build_fallback_utterance``) and the
    pre-adaptive assistance stage (student types "give me a hint"). Accepts a
    raw ``language`` and normalises it, so callers outside this module don't
    need to know the en/vi keying.
    """
    return _laddered_hint(hint_level, _lang(language))


# Rephrasing signposts (Slice 11, v2). Vary by reframe_count so a repeated
# reframe/clarify never repeats verbatim. Index 0 is the original v1 wording.
_REFRAME_SIGNPOSTS: dict[str, tuple[str, ...]] = {
    "en": (
        "Of course. Let me rephrase the question.",
        "Let me put it a different way.",
        "Here's another way to think about what I'm asking.",
    ),
    "vi": (
        "Tất nhiên. Để tôi diễn đạt lại câu hỏi.",
        "Để tôi nói theo một cách khác.",
        "Đây là một cách khác để hiểu câu hỏi của tôi.",
    ),
}


def _reframe_signpost(reframe_count: int, lang: str) -> str:
    """Rephrasing signpost that differs by ``reframe_count`` (clamped)."""
    variants = _REFRAME_SIGNPOSTS.get(lang, _REFRAME_SIGNPOSTS["en"])
    idx = max(0, min(reframe_count, len(variants) - 1))
    return variants[idx]


def _combine(acknowledgement: str, transition: str, question_or_probe: str) -> str:
    """Join the parts into one natural utterance, single-spaced, no doubling."""
    return " ".join(p for p in (acknowledgement, transition, question_or_probe) if p).strip()


# Affect-aware tone lead-ins (Slice 10, v2). Prepended to the acknowledgement to
# warm the tone for a nervous candidate or gently steer a rambling one. TONE
# ONLY — the action/reason/question are unchanged, so this never affects control
# flow or leaks answer content. Keyed by affect value string (avoids importing
# the Affect enum here and any import cycle); unknown/neutral → no lead-in.
_AFFECT_LEAD_IN: dict[tuple[str, str], str] = {
    ("nervous", "en"): "No rush — you're doing fine.",
    ("nervous", "vi"): "Bạn cứ từ từ — bạn đang làm tốt mà.",
    ("rambling", "en"): "Let's focus in a little.",
    ("rambling", "vi"): "Chúng ta hãy tập trung lại một chút.",
    ("terse", "en"): "Feel free to expand.",
    ("terse", "vi"): "Bạn cứ trình bày thêm nhé.",
}


def _affect_lead_in(affect_value: str | None, lang: str) -> str:
    """Optional tone lead-in for the detected affect (empty when none applies)."""
    if not affect_value:
        return ""
    return _AFFECT_LEAD_IN.get((affect_value, lang), "")


# Communication-polish lead-ins (Slice 20, v2). Same TONE-ONLY mechanism as the
# affect lead-in: prepended to the acknowledgement, never touching the
# question/probe or control flow. ``time_pressure`` signals the candidate to
# prioritise when little time remains; ``recovery`` rebuilds a rattled candidate
# after a weak streak with an encouraging, scoped lead-in.
_TIME_PRESSURE_LEAD_IN: dict[str, str] = {
    "en": "We're a little short on time, so let's prioritise.",
    "vi": "Chúng ta còn hơi ít thời gian, nên hãy tập trung vào điểm chính.",
}
_RECOVERY_LEAD_IN: dict[str, str] = {
    "en": "No problem — let's take a fresh, straightforward one.",
    "vi": "Không sao — mình thử một câu nhẹ nhàng, rõ ràng hơn nhé.",
}


def _polish_lead_in(
    *, recovery: bool, time_pressure: bool, affect_value: str | None, lang: str
) -> str:
    """Pick the single lead-in to prepend (Slice 20, v2).

    Precedence: recovery > time_pressure > affect. Only ONE lead-in is ever
    prepended so the tones never stack (a struggling candidate is rebuilt, not
    also told "we're short on time" and "you're doing fine"). With neither new
    signal set, this falls through to the existing affect lead-in → v1 wording.
    """
    if recovery:
        return _RECOVERY_LEAD_IN.get(lang, "")
    if time_pressure:
        return _TIME_PRESSURE_LEAD_IN.get(lang, "")
    return _affect_lead_in(affect_value, lang)


def build_fallback_utterance(
    decision: InterviewerDecision,
    *,
    persona: Persona,
    language: str | None,
    question_text: str | None = None,
    affect: object | None = None,
    hint_level: int = 0,
    reframe_count: int = 0,
    time_pressure: bool = False,
    recovery: bool = False,
) -> Utterance:
    """Deterministic, persona-aware, bilingual utterance for a decision.

    This is the guaranteed path — no I/O, never fails. The LLM phrasing layer
    (added in the logic module) wraps this and falls back to it on any error.

    ``affect`` (Slice 10, v2) optionally warms the TONE: a short reassuring /
    steering lead-in is prepended for a nervous / rambling / terse candidate.
    It only prepends to the acknowledgement — the question/probe text is
    untouched, so control flow and the answer-leak guard are unaffected. When
    None or NEUTRAL, the utterance is byte-for-byte the v1 result.

    ``hint_level`` / ``reframe_count`` (Slice 11, v2) select an escalating
    answer-safe hint or a rephrasing variant; both default to 0 → v1 wording.
    """
    lang = _lang(language)
    ack, transition, qp = _fallback_parts(
        decision,
        persona,
        lang,
        question_text=question_text,
        hint_level=hint_level,
        reframe_count=reframe_count,
    )
    affect_value = getattr(affect, "value", affect) if affect is not None else None
    lead_in = _polish_lead_in(
        recovery=recovery,
        time_pressure=time_pressure,
        affect_value=affect_value if isinstance(affect_value, str) else None,
        lang=lang,
    )
    ack = _combine(lead_in, "", ack) if lead_in else ack
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
    "laddered_hint",
    "persona_from",
    "transition_text",
]
