"""Difficulty streak adaptation (Slice 3).

Pure helpers — NO DB, NO LLM, NO state mutation. They turn the answer-quality
classification (Slice 2's ``is_strong_answer`` / ``is_weak_answer``) into two
things the runtime previously stored but never computed:

1. **Consecutive strong / weak streaks.** A strong answer increments the strong
   streak and resets the weak one; a weak answer does the reverse. Neutral
   (mixed / low-confidence / no analysis) answers leave both untouched, so noise
   never drags the streak around — this mirrors the brief's "treated as
   neutral" rule and Slice 2's strong/weak definitions.

2. **The difficulty level to aim the *next* question at.** Start from the
   current question's difficulty (mid-level when unknown), then step ONE level
   harder after ``STREAK_ADJUST_THRESHOLD`` consecutive strong answers, or one
   level easier after that many consecutive weak answers. Never more than one
   level per turn; always clamped to the junior..senior band.

Keeping this pure lets the streak + difficulty rules be unit-tested with plain
ints/enums, and lets ``run_adaptive_turn`` update streaks *before* it builds the
selection context so this turn's answer shapes the next question's difficulty.
Outcome coverage always outranks difficulty fit in the scorer, so this only
ever breaks ties among otherwise-comparable candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator.coverage import (
    is_strong_answer,
    is_weak_answer,
)
from abridgeai.features.interviews.orchestrator.selection import Difficulty

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis

# Consecutive same-quality answers required before difficulty shifts one level.
STREAK_ADJUST_THRESHOLD = 2

# Difficulty band (1 = junior, 2 = mid-level, 3 = senior). Mid-level is the
# neutral starting point when a question carries no difficulty metadata.
_MIN_RANK = 1
_MID_LEVEL_RANK = 2
_MAX_RANK = 3

_RANK_BY_DIFFICULTY: dict[str, int] = {
    Difficulty.JUNIOR.value: _MIN_RANK,
    Difficulty.MID_LEVEL.value: _MID_LEVEL_RANK,
    Difficulty.SENIOR.value: _MAX_RANK,
}


def difficulty_rank(difficulty: str | None) -> int | None:
    """Map a difficulty label to its 1..3 rank, or None when unknown/missing."""
    if difficulty is None:
        return None
    return _RANK_BY_DIFFICULTY.get(difficulty)


def update_streaks(
    *,
    consecutive_strong: int,
    consecutive_weak: int,
    analysis: AnswerAnalysis | None,
) -> tuple[int, int]:
    """Return the (strong, weak) streaks after folding in one answer.

    * strong answer  → strong + 1, weak reset to 0
    * weak answer    → weak + 1, strong reset to 0
    * neutral / none → both unchanged (noise must not move the streak)

    Strong and weak are mutually exclusive (guaranteed by Slice 2), so at most
    one branch fires.
    """
    if is_strong_answer(analysis):
        return consecutive_strong + 1, 0
    if is_weak_answer(analysis):
        return 0, consecutive_weak + 1
    return consecutive_strong, consecutive_weak


def target_difficulty_level(
    *,
    current_difficulty: str | None,
    consecutive_strong: int,
    consecutive_weak: int,
) -> int:
    """Difficulty rank (1..3) to aim the next question at.

    Base is the current question's difficulty (mid-level when unknown). A
    ``STREAK_ADJUST_THRESHOLD`` strong streak steps one level harder; a weak
    streak of the same length steps one level easier. At most one level of
    movement per call, clamped to the junior..senior band. A strong *and* weak
    streak can never both be at threshold (they're mutually exclusive per turn),
    but strong is checked first for determinism.
    """
    base = difficulty_rank(current_difficulty)
    if base is None:
        base = _MID_LEVEL_RANK
    if consecutive_strong >= STREAK_ADJUST_THRESHOLD:
        return min(_MAX_RANK, base + 1)
    if consecutive_weak >= STREAK_ADJUST_THRESHOLD:
        return max(_MIN_RANK, base - 1)
    return base


# EWMA smoothing factor for the per-outcome competence estimate (Slice 12). A
# single answer moves the estimate by this fraction toward its observed quality,
# so the calibration is responsive but noise-tolerant (needs a few consistent
# answers to swing hard).
_COMPETENCE_ALPHA = 0.4


def update_competence(*, prior: float, analysis: AnswerAnalysis | None) -> float:
    """EWMA-update a per-outcome competence estimate from one answer.

    Quality is 1.0 for a strong answer, 0.0 for a weak one, and neutral
    otherwise — a neutral / low-confidence / absent analysis leaves the estimate
    UNCHANGED (mirrors the streak rule: noise never drags calibration around).
    Returns the new estimate clamped to (0, 1).
    """
    if analysis is None:
        return prior
    if is_strong_answer(analysis):
        quality = 1.0
    elif is_weak_answer(analysis):
        quality = 0.0
    else:
        return prior  # neutral answer → no movement
    updated = (1.0 - _COMPETENCE_ALPHA) * prior + _COMPETENCE_ALPHA * quality
    return max(0.0, min(1.0, updated))


__all__ = [
    "STREAK_ADJUST_THRESHOLD",
    "difficulty_rank",
    "target_difficulty_level",
    "update_competence",
    "update_streaks",
]
