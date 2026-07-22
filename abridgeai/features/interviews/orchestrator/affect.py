"""Deterministic candidate-affect heuristics (Slice 10, v2).

Pure policy — NO DB, NO LLM. Reads lightweight signals from the answer text
(plus the optional structured analysis) to classify the candidate's affect:
terse, rambling, nervous, confident, or neutral.

This is used ONLY to warm the utterance TONE (a nervous candidate gets a gentler
lead-in; a rambling one a gentle steer). It NEVER changes control flow — the
action / reason / question are fixed before tone is applied — so a wrong read is
cosmetic, never harmful. That is what keeps the state machine deterministic
while letting the interviewer feel human.

An optional LLM read may refine this later; the heuristic is the guaranteed
floor and the safety net, mirroring the intent/analysis convention.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator.coverage import is_strong_answer

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis


class Affect(str, Enum):  # noqa: UP042 -- match codebase convention
    NEUTRAL = "neutral"
    NERVOUS = "nervous"
    TERSE = "terse"
    RAMBLING = "rambling"
    CONFIDENT = "confident"


# A very short answer (<= this many words) with no hedging reads as terse.
_TERSE_MAX_WORDS = 3
# A very long answer (>= this many words) with low substance reads as rambling.
_RAMBLING_MIN_WORDS = 60
# Hedging / filler markers (EN + VI) that signal nervousness / low confidence.
_HEDGE_PATTERNS: tuple[str, ...] = (
    r"\bum+\b",
    r"\buh+\b",
    r"\ber+\b",
    r"\bi'?m not (really )?sure\b",
    r"\bnot sure\b",
    r"\bi think maybe\b",
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bi guess\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bà|ừm|ờ\b",
    r"\bkhông chắc\b",
    r"\bcó lẽ\b",
    r"\bchắc là\b",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _hedge_hits(text: str) -> int:
    lowered = text.lower()
    return sum(1 for p in _HEDGE_PATTERNS if re.search(p, lowered))


def detect_affect(*, answer_text: str, analysis: AnswerAnalysis | None) -> Affect:
    """Classify candidate affect from the answer text + optional analysis.

    Precedence (first match wins), chosen so the tone adaptation is safe:
    1. RAMBLING  — very long answer that the analysis found low-substance.
    2. NERVOUS   — multiple hedging / filler markers.
    3. TERSE     — very short, non-hedged answer.
    4. CONFIDENT — a strong answer (relevant + complete + correct + confident).
    5. NEUTRAL   — everything else.
    """
    text = (answer_text or "").strip()
    if not text:
        return Affect.NEUTRAL

    words = _word_count(text)
    hedges = _hedge_hits(text)

    # 1. Rambling: long AND (no analysis substance OR explicitly vague/general).
    low_substance = analysis is None or not is_strong_answer(analysis)
    if words >= _RAMBLING_MIN_WORDS and low_substance:
        return Affect.RAMBLING

    # 2. Nervous: repeated hedging/filler (a single "maybe" is not enough).
    if hedges >= 2:
        return Affect.NERVOUS

    # 3. Terse: very short and not hedged.
    if words <= _TERSE_MAX_WORDS and hedges == 0:
        return Affect.TERSE

    # 4. Confident: the structured analysis says this was a strong answer.
    if is_strong_answer(analysis):
        return Affect.CONFIDENT

    return Affect.NEUTRAL


__all__ = ["Affect", "detect_affect"]
