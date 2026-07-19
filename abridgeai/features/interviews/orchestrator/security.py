"""Prompt-injection / exfiltration security model for interviews (Phase S1).

Pure types + deterministic EN/VI safe-response templates for the interview
security guard. NO DB access, NO LLM calls, NO policy execution here — the
deterministic rules + normalization live in :mod:`orchestrator.security_logic`
and the control-flow wiring lives in ``services.taking``. Keeping this module
side-effect-free mirrors the ``intent`` / ``decision`` convention and keeps it
trivially unit-testable.

Threat model (concise)
-----------------------
The only untrusted input is the student's utterance (typed in text/hybrid,
STT-transcribed in voice). Everything server-side (question bank, rubric,
decision, selection) is trusted. This module classifies WHAT the student is
attempting so ``services.taking`` can choose a deterministic control-flow
action BEFORE any academic analysis runs. The classifier identifies semantics;
**code** chooses the action (see :class:`SecurityAction`).

Privacy contract
----------------
:class:`SecurityAssessment` carries NO free-form model reasoning or
chain-of-thought and NO raw student content — only a bounded category, a
confidence band, control flags, a response-template key, and an optional
*normalized fingerprint* (a short hash) for dedup/observability. This is what
gets logged; raw utterances never do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SecurityCategory(str, Enum):  # noqa: UP042 -- StrEnum changes value coercion; match codebase convention
    """What the student utterance is attempting, security-wise.

    ``BENIGN`` is the overwhelming common case (a genuine academic answer or a
    legitimate repeat/clarify request). Every other member is an attack class
    the platform must neutralize.
    """

    BENIGN = "benign"
    FUTURE_QUESTION_REQUEST = "future_question_request"
    ANSWER_KEY_REQUEST = "answer_key_request"
    RUBRIC_EXFILTRATION = "rubric_exfiltration"
    SYSTEM_PROMPT_REQUEST = "system_prompt_request"
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLEPLAY_BYPASS = "roleplay_bypass"
    GRADING_MANIPULATION = "grading_manipulation"
    HIDDEN_STATE_REQUEST = "hidden_state_request"
    ENCODED_EXFILTRATION = "encoded_exfiltration"
    CROSS_SESSION_DATA_REQUEST = "cross_session_data_request"


# Every non-benign category. Membership here means "this turn is a security
# event" — the single source of truth so callers never enumerate categories.
ATTACK_CATEGORIES: frozenset[SecurityCategory] = frozenset(
    c for c in SecurityCategory if c is not SecurityCategory.BENIGN
)


class SecurityAction(str, Enum):  # noqa: UP042 -- match codebase convention
    """Deterministic control-flow action chosen by CODE (never by the model).

    The classifier may identify the semantic category, but the mapping from
    category (+ repeat count + policy) to one of these actions is deterministic
    and lives in :mod:`orchestrator.security_logic`. Security actions take
    precedence over any academic action (probe / example / advance).
    """

    ALLOW = "allow"
    REPEAT_CURRENT_QUESTION = "repeat_current_question"
    CLARIFY_CURRENT_QUESTION = "clarify_current_question"
    REFUSE_AND_REDIRECT = "refuse_and_redirect"
    WARN_AND_REDIRECT = "warn_and_redirect"
    END_AND_FLAG = "end_and_flag"


# Actions that BLOCK the academic pipeline for this turn (no intent/analysis,
# no evidence, no follow-up count). ALLOW and the legit repeat/clarify actions
# do NOT block — they flow into the normal pipeline.
BLOCKING_ACTIONS: frozenset[SecurityAction] = frozenset(
    {
        SecurityAction.REFUSE_AND_REDIRECT,
        SecurityAction.WARN_AND_REDIRECT,
        SecurityAction.END_AND_FLAG,
    }
)


@dataclass(frozen=True)
class SecurityAssessment:
    """Result of assessing one student utterance for a security concern.

    Deliberately minimal and privacy-safe:

    * ``category`` — the attack class (or ``BENIGN``).
    * ``detected`` — True iff ``category`` is an attack category.
    * ``confidence`` — 0..1. Deterministic rule hits are high (>= 0.9); the
      ambiguous-case classifier fills the middle band; benign is 0.0.
    * ``should_block`` — whether the academic pipeline must be skipped this
      turn (derived from the chosen action, not the category alone).
    * ``should_record_academic_evidence`` — normally False for any attack; True
      only when a benign answer also carries a *separable* legitimate answer
      (reserved for a later slice — Slice 1 keeps it conservative: attacks never
      record evidence).
    * ``response_key`` — key into the EN/VI safe-response templates, or None for
      ALLOW.
    * ``normalized_fingerprint`` — short hash of the normalized utterance for
      dedup / repeated-attempt tracking. NEVER the raw text.
    * ``source`` — ``"rules"`` | ``"classifier"`` | ``"fallback"`` for audit.
    """

    category: SecurityCategory
    detected: bool
    confidence: float
    should_block: bool
    should_record_academic_evidence: bool
    response_key: str | None
    normalized_fingerprint: str | None
    source: str = "rules"

    def to_dict(self) -> dict[str, object]:
        """Privacy-safe projection for observability / audit (no raw content)."""
        return {
            "category": self.category.value,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "should_block": self.should_block,
            "response_key": self.response_key,
            "normalized_fingerprint": self.normalized_fingerprint,
            "source": self.source,
        }

    @classmethod
    def benign(cls, *, source: str = "rules") -> SecurityAssessment:
        """The safe default: no attack detected, pipeline proceeds normally."""
        return cls(
            category=SecurityCategory.BENIGN,
            detected=False,
            confidence=0.0,
            should_block=False,
            should_record_academic_evidence=True,
            response_key=None,
            normalized_fingerprint=None,
            source=source,
        )


# ── Deterministic safe responses (EN/VI) ─────────────────────────────────────
# Keyed by a stable ``response_key``. These are the fallback texts the student
# sees when a security action fires. The LLM utterance layer MAY rephrase them
# per persona, but it must never reveal protected content, change the action,
# imply academic penalty, explain the detection rule, or become accusatory (a
# later slice enforces that at the utterance layer).
#
# Structure: RESPONSE_TEMPLATES[response_key][language] -> str. English is the
# guaranteed fallback for any missing language.

_REFUSE_EN = (
    "I can't provide hidden interview questions, answers, or grading criteria. "
    "I can repeat or clarify the current question."
)
_REFUSE_VI = (
    "Tôi không thể cung cấp các câu hỏi chưa được hỏi, đáp án hoặc tiêu chí "
    "chấm điểm. Tôi có thể nhắc lại hoặc giải thích câu hỏi hiện tại."
)

# WARN escalates the tone slightly on a repeated attempt — still non-accusatory,
# still no protected content, still offers the legitimate path forward.
_WARN_EN = (
    "I'm not able to share hidden questions, answers, or grading details, and I "
    "can't change how the interview is scored. Let's continue — I'll restate the "
    "current question so you can answer it."
)
_WARN_VI = (
    "Tôi không thể chia sẻ câu hỏi ẩn, đáp án hoặc chi tiết chấm điểm, và cũng "
    "không thể thay đổi cách chấm điểm buổi phỏng vấn. Chúng ta hãy tiếp tục — "
    "tôi sẽ nhắc lại câu hỏi hiện tại để bạn trả lời."
)

# END_AND_FLAG closing — neutral, no accusation, no mention of penalty.
_END_EN = (
    "We'll wrap up the interview here. Thank you for your time."
)
_END_VI = (
    "Chúng ta sẽ kết thúc buổi phỏng vấn tại đây. Cảm ơn bạn đã tham gia."
)

RESPONSE_KEY_REFUSE = "security_refuse_and_redirect"
RESPONSE_KEY_WARN = "security_warn_and_redirect"
RESPONSE_KEY_END = "security_end_and_flag"

RESPONSE_TEMPLATES: dict[str, dict[str, str]] = {
    RESPONSE_KEY_REFUSE: {"en": _REFUSE_EN, "vi": _REFUSE_VI},
    RESPONSE_KEY_WARN: {"en": _WARN_EN, "vi": _WARN_VI},
    RESPONSE_KEY_END: {"en": _END_EN, "vi": _END_VI},
}

# Maps a blocking action to its response-template key. Non-blocking actions have
# no template (they flow into the normal pipeline).
ACTION_RESPONSE_KEY: dict[SecurityAction, str] = {
    SecurityAction.REFUSE_AND_REDIRECT: RESPONSE_KEY_REFUSE,
    SecurityAction.WARN_AND_REDIRECT: RESPONSE_KEY_WARN,
    SecurityAction.END_AND_FLAG: RESPONSE_KEY_END,
}


def safe_response_text(response_key: str | None, language: str | None) -> str:
    """Return the deterministic safe text for a template key + language.

    Falls back to English for any unknown/missing language, and to the refusal
    text for an unknown key (fail safe — never returns protected content or an
    empty string that could surface a raw model output instead).
    """
    if not response_key:
        response_key = RESPONSE_KEY_REFUSE
    lang = (language or "en").lower()
    lang = "vi" if lang.startswith("vi") else "en"
    by_lang = RESPONSE_TEMPLATES.get(response_key) or RESPONSE_TEMPLATES[RESPONSE_KEY_REFUSE]
    return by_lang.get(lang) or by_lang["en"]


__all__ = [
    "ACTION_RESPONSE_KEY",
    "ATTACK_CATEGORIES",
    "BLOCKING_ACTIONS",
    "RESPONSE_KEY_END",
    "RESPONSE_KEY_REFUSE",
    "RESPONSE_KEY_WARN",
    "RESPONSE_TEMPLATES",
    "SecurityAction",
    "SecurityAssessment",
    "SecurityCategory",
    "safe_response_text",
]
