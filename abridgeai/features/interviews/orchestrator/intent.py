"""Student intent classification (Phase 2).

Before an utterance is treated as an academic *answer*, the orchestrator
classifies WHAT the student is doing: answering, asking to repeat, requesting
clarification, saying they can't answer, reporting a technical issue, etc.
This prevents scoring a "can you repeat that?" as a failed answer.

Design (mirrors the follow-up stage conventions):
* Pure types + deterministic fallback + permissive parser here; the LLM call
  lives in :mod:`orchestrator.intent_logic` so this module stays trivially
  unit-testable with no I/O.
* Deterministic fallback rules run FIRST for the obvious cases (cheap, robust,
  language-aware EN + VI) and are also the safety net when the classifier
  fails — the session must never block on a classifier error.
* Structured internal output carries ``confidence`` + ``rationale`` for audit;
  the student never sees it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StudentIntent(str, Enum):  # noqa: UP042 -- StrEnum changes value coercion; match codebase convention
    ANSWER = "answer"
    PARTIAL_ANSWER = "partial_answer"
    ASK_TO_REPEAT = "ask_to_repeat"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"
    ASK_FOR_HINT = "ask_for_hint"
    ASK_FOR_MORE_TIME = "ask_for_more_time"
    SKIP_QUESTION = "skip_question"
    CANNOT_ANSWER = "cannot_answer"
    TECHNICAL_ISSUE = "technical_issue"
    OFF_TOPIC = "off_topic"
    END_INTERVIEW = "end_interview"


# Intents that must NEVER be recorded as an academic answer / scored.
NON_ACADEMIC_INTENTS: frozenset[StudentIntent] = frozenset(
    {
        StudentIntent.ASK_TO_REPEAT,
        StudentIntent.ASK_FOR_CLARIFICATION,
        StudentIntent.ASK_FOR_HINT,
        StudentIntent.ASK_FOR_MORE_TIME,
        StudentIntent.SKIP_QUESTION,
        StudentIntent.TECHNICAL_ISSUE,
        StudentIntent.END_INTERVIEW,
    }
)


@dataclass(frozen=True)
class IntentClassification:
    """Result of classifying a student utterance.

    ``confidence`` is 0..1. ``rationale`` is a short audit string (never shown
    to the student). ``source`` records whether the verdict came from the
    deterministic rules or the LLM, for observability.
    """

    intent: StudentIntent
    confidence: float
    rationale: str
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
        }


# ── Deterministic fallback rules ─────────────────────────────────────────────
# Ordered (intent, patterns) tuples. First match wins. Patterns are matched
# case-insensitively against the whole (stripped) utterance. Kept intentionally
# high-precision — only fire on unambiguous phrasings; everything else defers to
# the LLM (or, on LLM failure, defaults to ANSWER so progression never stalls).

_RULES: tuple[tuple[StudentIntent, tuple[str, ...]], ...] = (
    (
        StudentIntent.ASK_TO_REPEAT,
        (
            r"^(please )?repeat([\s,]+please)?[.!?]*$",
            r"^(again|one more time)[.!?]*$",
            r"\b(can|could) you (please )?repeat\b",
            r"\brepeat the question\b",
            r"\bsay (that|it) again\b",
            r"\bcome again\b",
            r"^(vui lòng |xin )?(nhắc lại|lặp lại)( (đi|giúp (tôi|mình|em)))?[.!?]*$",
            r"\bnhắc lại câu hỏi\b",
            r"\b(bạn )?(có thể )?nhắc lại\b",
            r"\blặp lại (câu hỏi)?\b",
        ),
    ),
    (
        StudentIntent.ASK_FOR_HINT,
        (
            r"\b(can|could|would) you (please )?(give|provide) me (a )?"
            r"(small |brief |little )?hint\b",
            r"\b(can|could) i (please )?(get|have) (a )?(small |brief |little )?hint\b",
            r"\b(give|provide) me (a )?(small |brief |little )?hint\b",
            r"\b(gợi ý|cho (tôi|mình|em) một gợi ý)\b",
        ),
    ),
    (
        StudentIntent.ASK_FOR_CLARIFICATION,
        (
            r"\bwhat do you mean\b",
            r"\bwhat does .{0,30}\bmean\b",
            r"\bcan you clarify\b",
            r"\bcould you clarify\b",
            r"\bi don't understand the question\b",
            r"\bnghĩa là (gì|sao)\b",
            r"\bý (bạn )?là (gì|sao)\b",
            r"\bgiải thích .{0,20}(câu hỏi|nghĩa)\b",
        ),
    ),
    (
        StudentIntent.ASK_FOR_MORE_TIME,
        (
            r"\b(can|could) i (have|get) (a )?(bit )?more time\b",
            r"\bgive me a (minute|moment|second)\b",
            r"\blet me think\b",
            r"\bcho (tôi|mình|em) (thêm )?(chút |một chút |ít )?(thời gian|phút|giây)\b",
            r"\bđể (tôi|mình|em) suy nghĩ\b",
        ),
    ),
    (
        StudentIntent.SKIP_QUESTION,
        (
            r"\b(can|could) (we|i) (skip|move on|move past)\b",
            r"\bskip (this|the) question\b",
            r"\bnext question\b",
            r"\bpass\b",
            r"\b(cho )?(qua|bỏ qua) câu (này|hỏi này)\b",
            r"\bchuyển câu (khác|tiếp)\b",
            r"\bcâu (tiếp theo|kế tiếp)\b",
        ),
    ),
    (
        StudentIntent.CANNOT_ANSWER,
        (
            r"\bi don'?t know\b",
            r"\bi have no idea\b",
            r"\bno idea\b",
            r"\bi'?m not sure\b",
            r"\bi can'?t answer\b",
            r"\bnot sure\b",
            r"\b(tôi|mình|em) không biết\b",
            r"\bchịu\b",
            r"\bkhông rõ\b",
            r"\bkhông chắc\b",
        ),
    ),
    (
        StudentIntent.TECHNICAL_ISSUE,
        (
            r"\b(my )?(microphone|mic|audio|camera|connection|internet) .{0,20}(not working|isn'?t working|broken|down|issue|problem)\b",  # noqa: E501
            r"\bcan'?t hear\b",
            r"\byou'?re (breaking up|cutting out)\b",
            r"\blag(ging)?\b",
            r"\b(micro|mic|âm thanh|kết nối|mạng) .{0,20}(không|lỗi|hỏng|có vấn đề)\b",  # noqa: E501
            r"\bkhông nghe (thấy|được)\b",
        ),
    ),
    (
        StudentIntent.END_INTERVIEW,
        (
            r"\b(i want to |can we |let'?s )?(end|stop|finish|quit) (the |this )?(interview|session)\b",  # noqa: E501
            r"\bi'?m done\b",
            r"\b(kết thúc|dừng|thoát) (buổi |cuộc )?phỏng vấn\b",
            r"\b(tôi|mình|em) (muốn )?(dừng|kết thúc)\b",
        ),
    ),
)


def classify_by_rules(utterance: str) -> IntentClassification | None:
    """Deterministic first-pass classification. Returns None when no rule fires.

    High-precision by design: only unambiguous phrasings match. A non-match is
    NOT "this is an answer" — it means "let the LLM decide" (the caller falls
    back to ANSWER only when the LLM is also unavailable).
    """
    text = (utterance or "").strip().lower()
    if not text:
        # Empty utterance — treat as cannot_answer/silence at the rule layer.
        return IntentClassification(
            intent=StudentIntent.CANNOT_ANSWER,
            confidence=0.5,
            rationale="Empty utterance treated as silence/cannot-answer.",
            source="rules",
        )
    for intent, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                return IntentClassification(
                    intent=intent,
                    confidence=0.9,
                    rationale=f"Deterministic rule match for {intent.value}.",
                    source="rules",
                )
    return None


def parse_intent_response(payload: Mapping[str, Any] | None) -> IntentClassification | None:
    """Coerce the gateway JSON into an :class:`IntentClassification`.

    Contract::

        {"intent": "<one of StudentIntent>", "confidence": 0.0-1.0,
         "rationale": "short string"}

    Returns None (not a fallback verdict) when the payload is unusable, so the
    caller can apply its own deterministic fallback rather than trusting a
    guessed intent. An unknown intent string also yields None.
    """
    if not isinstance(payload, Mapping):
        return None
    raw_intent = payload.get("intent")
    if not isinstance(raw_intent, str):
        return None
    try:
        intent = StudentIntent(raw_intent.strip().lower())
    except ValueError:
        return None
    return IntentClassification(
        intent=intent,
        confidence=_coerce_confidence(payload.get("confidence")),
        rationale=_coerce_rationale(payload.get("rationale")),
        source="llm",
    )


def _coerce_confidence(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, conf))


def _coerce_rationale(value: object) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned[:500]
    return "No rationale provided."


__all__ = [
    "NON_ACADEMIC_INTENTS",
    "IntentClassification",
    "StudentIntent",
    "classify_by_rules",
    "parse_intent_response",
]
