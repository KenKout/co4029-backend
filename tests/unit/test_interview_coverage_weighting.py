"""Unit tests for weighted provisional coverage + answer-quality policy (Slice 2).

The coverage module is pure (no DB, no LLM, no state save), so these tests
exercise the 2/1/0 weighting, the confidence gate, the sufficiency threshold,
the in-place ``apply_evidence_to_coverage`` mutation, and the strong/weak
answer classification directly with plain enums.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Completeness,
    Correctness,
    EvidenceType,
    OutcomeEvidence,
    Relevance,
    Specificity,
)
from abridgeai.features.interviews.orchestrator.coverage import (
    COVERAGE_SUFFICIENT_POINTS,
    EVIDENCE_CONFIDENCE_MIN,
    STRONG_CONFIDENCE_MIN,
    apply_evidence_to_coverage,
    evidence_points,
    is_provisionally_sufficient,
    is_strong_answer,
    is_weak_answer,
)
from abridgeai.features.interviews.orchestrator.state import (
    CoverageStatus,
    OutcomeCoverageState,
)


def _ev(
    etype: EvidenceType,
    *,
    confidence: float = 0.9,
    outcome_id: str = "o1",
    turn_id: str = "t1",
) -> OutcomeEvidence:
    return OutcomeEvidence(
        outcome_id=outcome_id,
        turn_id=turn_id,
        evidence_type=etype,
        confidence=confidence,
    )


# ── evidence_points weighting ────────────────────────────────────────────────


def test_confident_support_worth_two_points() -> None:
    assert evidence_points(_ev(EvidenceType.SUPPORTS)) == 2


def test_confident_partial_worth_one_point() -> None:
    assert evidence_points(_ev(EvidenceType.PARTIALLY_SUPPORTS)) == 1


def test_contradiction_and_insufficient_worth_zero() -> None:
    assert evidence_points(_ev(EvidenceType.CONTRADICTS)) == 0
    assert evidence_points(_ev(EvidenceType.INSUFFICIENT)) == 0


def test_low_confidence_establishes_no_coverage_regardless_of_type() -> None:
    # Even a "supports" item below the confidence floor is worth nothing.
    low = _ev(EvidenceType.SUPPORTS, confidence=EVIDENCE_CONFIDENCE_MIN - 0.01)
    assert evidence_points(low) == 0
    # At the floor it counts.
    at_floor = _ev(EvidenceType.SUPPORTS, confidence=EVIDENCE_CONFIDENCE_MIN)
    assert evidence_points(at_floor) == 2


# ── sufficiency threshold ────────────────────────────────────────────────────


def test_sufficiency_requires_two_points() -> None:
    assert is_provisionally_sufficient(0) is False
    assert is_provisionally_sufficient(1) is False
    assert is_provisionally_sufficient(2) is True
    assert is_provisionally_sufficient(3) is True


def test_one_partial_is_not_sufficient_but_two_are() -> None:
    # The core anti-flat-count guarantee: a single partial (1pt) does NOT cover;
    # two partials (2pt) do, and one full support (2pt) does on its own.
    assert is_provisionally_sufficient(1) is False
    assert is_provisionally_sufficient(1 + 1) is True
    assert is_provisionally_sufficient(2) is True


# ── apply_evidence_to_coverage (in-place mutation) ───────────────────────────


def test_apply_accumulates_points_and_count_and_status() -> None:
    cov = OutcomeCoverageState(outcome_id="o1")
    apply_evidence_to_coverage(cov, _ev(EvidenceType.PARTIALLY_SUPPORTS), now="ts1")
    assert cov.evidence_count == 1
    assert cov.coverage_points == 1
    assert cov.status is CoverageStatus.PARTIAL
    assert cov.last_updated_at == "ts1"
    assert cov.supporting_turn_ids == ["t1"]

    # A second partial pushes it over the sufficiency threshold.
    apply_evidence_to_coverage(cov, _ev(EvidenceType.PARTIALLY_SUPPORTS, turn_id="t2"))
    assert cov.evidence_count == 2
    assert cov.coverage_points == 2
    assert cov.status is CoverageStatus.SUFFICIENT
    assert cov.supporting_turn_ids == ["t1", "t2"]


def test_apply_full_support_is_immediately_sufficient() -> None:
    cov = OutcomeCoverageState(outcome_id="o1")
    apply_evidence_to_coverage(cov, _ev(EvidenceType.SUPPORTS))
    assert cov.coverage_points == 2
    assert cov.status is CoverageStatus.SUFFICIENT


def test_apply_zero_value_evidence_marks_insufficient_not_notstarted() -> None:
    cov = OutcomeCoverageState(outcome_id="o1")
    apply_evidence_to_coverage(cov, _ev(EvidenceType.CONTRADICTS))
    assert cov.evidence_count == 1
    assert cov.coverage_points == 0
    # Evidence WAS seen, it just established no coverage → INSUFFICIENT, not NOT_STARTED.
    assert cov.status is CoverageStatus.INSUFFICIENT


def test_apply_dedups_turn_ids() -> None:
    cov = OutcomeCoverageState(outcome_id="o1")
    apply_evidence_to_coverage(cov, _ev(EvidenceType.SUPPORTS, turn_id="dup"))
    apply_evidence_to_coverage(cov, _ev(EvidenceType.PARTIALLY_SUPPORTS, turn_id="dup"))
    # Same turn contributed twice to points/count but the turn id is listed once.
    assert cov.supporting_turn_ids == ["dup"]
    assert cov.evidence_count == 2


def test_apply_low_confidence_counts_but_adds_no_points() -> None:
    cov = OutcomeCoverageState(outcome_id="o1")
    apply_evidence_to_coverage(cov, _ev(EvidenceType.SUPPORTS, confidence=0.1))
    # Recorded for traceability, but establishes no coverage.
    assert cov.evidence_count == 1
    assert cov.coverage_points == 0
    assert cov.status is CoverageStatus.INSUFFICIENT


# ── round-trip: coverage_points survives serialization + backfill ────────────


def test_coverage_points_round_trips_through_serialization() -> None:
    cov = OutcomeCoverageState(outcome_id="o1")
    apply_evidence_to_coverage(cov, _ev(EvidenceType.SUPPORTS))
    restored = OutcomeCoverageState.from_dict(cov.to_dict())
    assert restored.coverage_points == 2
    assert restored.status is CoverageStatus.SUFFICIENT


def test_legacy_row_without_points_backfills_from_evidence_count() -> None:
    # A pre-Slice-2 payload has evidence_count but no coverage_points. Backfill
    # seeds points from the raw count so a resumed session keeps its coverage.
    legacy = {"outcome_id": "o1", "evidence_count": 2}
    restored = OutcomeCoverageState.from_dict(legacy)
    assert restored.coverage_points == 2


# ── strong / weak answer classification ──────────────────────────────────────


def _analysis(
    *,
    relevance: Relevance = Relevance.RELEVANT,
    completeness: Completeness = Completeness.COMPLETE,
    correctness: Correctness = Correctness.CORRECT,
    specificity: Specificity = Specificity.SPECIFIC,
    confidence: float = 0.9,
) -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=relevance,
        completeness=completeness,
        correctness=correctness,
        specificity=specificity,
        confidence=confidence,
    )


def test_strong_answer_requires_all_criteria() -> None:
    assert is_strong_answer(_analysis()) is True
    assert is_strong_answer(_analysis(correctness=Correctness.MOSTLY_CORRECT)) is True


def test_strong_answer_fails_below_confidence_floor() -> None:
    assert is_strong_answer(_analysis(confidence=STRONG_CONFIDENCE_MIN - 0.01)) is False
    assert is_strong_answer(_analysis(confidence=STRONG_CONFIDENCE_MIN)) is True


def test_strong_answer_fails_if_not_relevant_or_incomplete() -> None:
    assert is_strong_answer(_analysis(relevance=Relevance.PARTIALLY_RELEVANT)) is False
    assert is_strong_answer(_analysis(completeness=Completeness.PARTIAL)) is False
    assert is_strong_answer(_analysis(correctness=Correctness.MIXED)) is False


def test_none_analysis_is_neither_strong_nor_weak() -> None:
    assert is_strong_answer(None) is False
    assert is_weak_answer(None) is False


def test_weak_answer_confidently_incorrect_or_insufficient_or_vague() -> None:
    assert is_weak_answer(_analysis(correctness=Correctness.INCORRECT)) is True
    assert is_weak_answer(_analysis(completeness=Completeness.INSUFFICIENT)) is True
    # A vague answer is weak only when it isn't already strong — a complete,
    # correct answer that is merely vaguely phrased is strong, not weak (see
    # test_strong_and_weak_are_mutually_exclusive). Pair vagueness with a
    # non-strong correctness so the vague signal is what drives weakness.
    assert (
        is_weak_answer(_analysis(correctness=Correctness.MIXED, specificity=Specificity.VAGUE))
        is True
    )


def test_low_confidence_answer_is_neutral_not_weak() -> None:
    # A confidently-incorrect read is weak; the SAME verdict at low confidence
    # is neutral, so noise can't drag the difficulty streak down.
    weak = _analysis(correctness=Correctness.INCORRECT, confidence=0.9)
    noise = _analysis(correctness=Correctness.INCORRECT, confidence=0.2)
    assert is_weak_answer(weak) is True
    assert is_weak_answer(noise) is False


def test_strong_and_weak_are_mutually_exclusive() -> None:
    # No single analysis can be both strong and weak.
    for correctness in Correctness:
        for completeness in Completeness:
            for specificity in Specificity:
                a = _analysis(
                    correctness=correctness,
                    completeness=completeness,
                    specificity=specificity,
                )
                assert not (is_strong_answer(a) and is_weak_answer(a))


def test_sufficient_points_constant_is_two() -> None:
    # Guards against an accidental threshold change slipping through review.
    assert COVERAGE_SUFFICIENT_POINTS == 2
