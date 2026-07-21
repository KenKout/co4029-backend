"""Unit tests for the adaptive-readiness analyzer (Slice 5).

The analyzer is pure (no DB, no LLM), so these exercise every warning branch
plus the "clean config → no warnings" case directly with plain objects.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.readiness import (
    ReadinessCode,
    ReadinessInputs,
    ReadinessLevel,
    ReadinessOutcome,
    ReadinessQuestion,
    analyze_readiness,
)


def _q(
    qid: str,
    *,
    outcome: str | None = "o1",
    difficulty: str | None = "mid_level",
) -> ReadinessQuestion:
    return ReadinessQuestion(question_id=qid, linked_outcome_id=outcome, difficulty=difficulty)


def _codes(inputs: ReadinessInputs) -> set[ReadinessCode]:
    return {w.code for w in analyze_readiness(inputs)}


def test_clean_config_has_no_warnings() -> None:
    inputs = ReadinessInputs(
        questions=[
            _q("q1", outcome="o1", difficulty="junior"),
            _q("q2", outcome="o2", difficulty="senior"),
        ],
        outcomes=[ReadinessOutcome("o1"), ReadinessOutcome("o2")],
        time_limit_minutes=8,
    )
    assert analyze_readiness(inputs) == []


def test_flags_question_without_outcome() -> None:
    inputs = ReadinessInputs(
        questions=[_q("q1", outcome=None, difficulty="junior")],
        outcomes=[],
    )
    warnings = analyze_readiness(inputs)
    orphan = next(w for w in warnings if w.code is ReadinessCode.QUESTIONS_WITHOUT_OUTCOME)
    assert orphan.level is ReadinessLevel.WARNING
    assert orphan.affected_ids == ["q1"]


def test_flags_outcome_without_question() -> None:
    inputs = ReadinessInputs(
        questions=[_q("q1", outcome="o1", difficulty="junior")],
        outcomes=[ReadinessOutcome("o1"), ReadinessOutcome("o_orphan")],
    )
    warnings = analyze_readiness(inputs)
    uncovered = next(w for w in warnings if w.code is ReadinessCode.OUTCOMES_WITHOUT_QUESTION)
    assert uncovered.affected_ids == ["o_orphan"]


def test_flags_missing_difficulty() -> None:
    inputs = ReadinessInputs(
        questions=[_q("q1", outcome="o1", difficulty=None)],
        outcomes=[ReadinessOutcome("o1")],
    )
    assert ReadinessCode.QUESTIONS_MISSING_DIFFICULTY in _codes(inputs)


def test_flags_low_difficulty_diversity_when_all_same_level() -> None:
    inputs = ReadinessInputs(
        questions=[
            _q("q1", outcome="o1", difficulty="mid_level"),
            _q("q2", outcome="o2", difficulty="mid_level"),
        ],
        outcomes=[ReadinessOutcome("o1"), ReadinessOutcome("o2")],
    )
    diversity = next(
        w for w in analyze_readiness(inputs) if w.code is ReadinessCode.LOW_DIFFICULTY_DIVERSITY
    )
    assert diversity.level is ReadinessLevel.INFO


def test_diverse_difficulty_does_not_flag_diversity() -> None:
    inputs = ReadinessInputs(
        questions=[
            _q("q1", outcome="o1", difficulty="junior"),
            _q("q2", outcome="o2", difficulty="senior"),
        ],
        outcomes=[ReadinessOutcome("o1"), ReadinessOutcome("o2")],
    )
    assert ReadinessCode.LOW_DIFFICULTY_DIVERSITY not in _codes(inputs)


def test_single_labelled_question_does_not_flag_diversity() -> None:
    # Diversity is only meaningful with 2+ labelled questions.
    inputs = ReadinessInputs(
        questions=[_q("q1", outcome="o1", difficulty="mid_level")],
        outcomes=[ReadinessOutcome("o1")],
    )
    assert ReadinessCode.LOW_DIFFICULTY_DIVERSITY not in _codes(inputs)


def test_flags_insufficient_coverage_for_duration() -> None:
    # 40 min / 4 min-per-question = 10 expected; only 2 approved → advise.
    inputs = ReadinessInputs(
        questions=[
            _q("q1", outcome="o1", difficulty="junior"),
            _q("q2", outcome="o2", difficulty="senior"),
        ],
        outcomes=[ReadinessOutcome("o1"), ReadinessOutcome("o2")],
        time_limit_minutes=40,
    )
    coverage = next(
        w
        for w in analyze_readiness(inputs)
        if w.code is ReadinessCode.INSUFFICIENT_QUESTION_COVERAGE
    )
    assert coverage.level is ReadinessLevel.INFO
    assert coverage.count == 2


def test_untimed_interview_skips_coverage_check() -> None:
    inputs = ReadinessInputs(
        questions=[_q("q1", outcome="o1", difficulty="junior")],
        outcomes=[ReadinessOutcome("o1")],
        time_limit_minutes=None,
    )
    assert ReadinessCode.INSUFFICIENT_QUESTION_COVERAGE not in _codes(inputs)


def test_warning_to_dict_shape() -> None:
    inputs = ReadinessInputs(
        questions=[_q("q1", outcome=None, difficulty="junior")],
        outcomes=[],
    )
    payload = analyze_readiness(inputs)[0].to_dict()
    assert payload == {
        "code": "questions_without_outcome",
        "level": "warning",
        "affected_ids": ["q1"],
        "count": 1,
    }
