"""Unit tests for difficulty streak adaptation (Slice 3).

The difficulty module is pure (no DB, no LLM, no state save), so these tests
exercise the streak transitions and the target-difficulty derivation directly
with plain enums/ints. Together with Slice 2's strong/weak classification they
cover the brief's rule: escalate/ease difficulty ONLY after a two-answer streak,
one level at a time, clamped to the junior..senior band; neutral/low-confidence
answers never move the streak.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Completeness,
    Correctness,
    Relevance,
    Specificity,
)
from abridgeai.features.interviews.orchestrator.difficulty import (
    STREAK_ADJUST_THRESHOLD,
    difficulty_rank,
    target_difficulty_level,
    update_streaks,
)
from abridgeai.features.interviews.orchestrator.selection import Difficulty


def _strong() -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=Correctness.CORRECT,
        specificity=Specificity.SPECIFIC,
        confidence=0.9,
    )


def _weak() -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.INSUFFICIENT,
        correctness=Correctness.INCORRECT,
        specificity=Specificity.VAGUE,
        confidence=0.9,
    )


def _neutral() -> AnswerAnalysis:
    # Mixed correctness at decent confidence → neither strong nor weak.
    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.PARTIAL,
        correctness=Correctness.MIXED,
        specificity=Specificity.GENERAL,
        confidence=0.9,
    )


def _low_confidence_incorrect() -> AnswerAnalysis:
    # Would be weak at high confidence; low confidence makes it neutral (noise).
    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.INSUFFICIENT,
        correctness=Correctness.INCORRECT,
        specificity=Specificity.VAGUE,
        confidence=0.2,
    )


# ── difficulty_rank ──────────────────────────────────────────────────────────


def test_difficulty_rank_maps_known_labels() -> None:
    assert difficulty_rank(Difficulty.JUNIOR.value) == 1
    assert difficulty_rank(Difficulty.MID_LEVEL.value) == 2
    assert difficulty_rank(Difficulty.SENIOR.value) == 3


def test_difficulty_rank_unknown_or_missing_is_none() -> None:
    assert difficulty_rank(None) is None
    assert difficulty_rank("principal") is None


# ── update_streaks ───────────────────────────────────────────────────────────


def test_strong_answer_increments_strong_and_resets_weak() -> None:
    strong, weak = update_streaks(consecutive_strong=1, consecutive_weak=3, analysis=_strong())
    assert strong == 2
    assert weak == 0


def test_weak_answer_increments_weak_and_resets_strong() -> None:
    strong, weak = update_streaks(consecutive_strong=4, consecutive_weak=0, analysis=_weak())
    assert strong == 0
    assert weak == 1


def test_neutral_answer_leaves_both_streaks_unchanged() -> None:
    strong, weak = update_streaks(consecutive_strong=2, consecutive_weak=1, analysis=_neutral())
    assert strong == 2
    assert weak == 1


def test_low_confidence_answer_is_noise_and_does_not_move_streak() -> None:
    strong, weak = update_streaks(
        consecutive_strong=1, consecutive_weak=1, analysis=_low_confidence_incorrect()
    )
    assert strong == 1
    assert weak == 1


def test_missing_analysis_leaves_streaks_unchanged() -> None:
    strong, weak = update_streaks(consecutive_strong=1, consecutive_weak=1, analysis=None)
    assert strong == 1
    assert weak == 1


# ── target_difficulty_level ──────────────────────────────────────────────────


def test_base_level_is_current_difficulty_without_a_streak() -> None:
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.MID_LEVEL.value,
            consecutive_strong=0,
            consecutive_weak=0,
        )
        == 2
    )


def test_unknown_current_difficulty_defaults_to_mid_level() -> None:
    assert (
        target_difficulty_level(current_difficulty=None, consecutive_strong=0, consecutive_weak=0)
        == 2
    )


def test_strong_streak_at_threshold_steps_one_level_harder() -> None:
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.MID_LEVEL.value,
            consecutive_strong=STREAK_ADJUST_THRESHOLD,
            consecutive_weak=0,
        )
        == 3
    )


def test_weak_streak_at_threshold_steps_one_level_easier() -> None:
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.MID_LEVEL.value,
            consecutive_strong=0,
            consecutive_weak=STREAK_ADJUST_THRESHOLD,
        )
        == 1
    )


def test_streak_below_threshold_does_not_shift() -> None:
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.MID_LEVEL.value,
            consecutive_strong=STREAK_ADJUST_THRESHOLD - 1,
            consecutive_weak=0,
        )
        == 2
    )


def test_escalation_never_exceeds_senior() -> None:
    # Already at senior + a strong streak → clamp at 3, not 4.
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.SENIOR.value,
            consecutive_strong=5,
            consecutive_weak=0,
        )
        == 3
    )


def test_easing_never_drops_below_junior() -> None:
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.JUNIOR.value,
            consecutive_strong=0,
            consecutive_weak=5,
        )
        == 1
    )


def test_only_one_level_of_movement_per_turn() -> None:
    # A long strong streak from junior still only reaches mid-level (one step),
    # never senior in a single turn.
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.JUNIOR.value,
            consecutive_strong=10,
            consecutive_weak=0,
        )
        == 2
    )


def test_two_strong_answers_escalate_end_to_end() -> None:
    # Simulate the streak building across two strong answers from a mid-level
    # base, then confirm the derived target steps up exactly once.
    strong, weak = 0, 0
    strong, weak = update_streaks(
        consecutive_strong=strong, consecutive_weak=weak, analysis=_strong()
    )
    # One strong answer is not yet enough.
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.MID_LEVEL.value,
            consecutive_strong=strong,
            consecutive_weak=weak,
        )
        == 2
    )
    strong, weak = update_streaks(
        consecutive_strong=strong, consecutive_weak=weak, analysis=_strong()
    )
    # Two consecutive strong → escalate one level.
    assert strong == 2
    assert (
        target_difficulty_level(
            current_difficulty=Difficulty.MID_LEVEL.value,
            consecutive_strong=strong,
            consecutive_weak=weak,
        )
        == 3
    )
