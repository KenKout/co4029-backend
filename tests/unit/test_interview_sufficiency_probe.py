"""Unit tests for the fast sufficiency probe and its coverage reconciliation.

Three properties matter here, and they are the ones that make it safe to take the
full analysis off the blocking turn path:

1. The minimal verdict projects onto the EXISTING coverage weighting so
   ``is_provisionally_sufficient`` flips exactly when it should.
2. When the authoritative full analysis disagrees with the probe, coverage ends
   up at the full analysis's number — including downwards, revoking a tick.
3. A probe failure establishes NO coverage and never raises into the turn.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from abridgeai.features.interviews.orchestrator import sufficiency_logic
from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    EvidenceType,
    OutcomeEvidence,
)
from abridgeai.features.interviews.orchestrator.coverage import (
    COVERAGE_SUFFICIENT_POINTS,
    apply_evidence_to_coverage,
    is_provisionally_sufficient,
)
from abridgeai.features.interviews.orchestrator.reconciliation import reconcile_turn_coverage
from abridgeai.features.interviews.orchestrator.state import (
    CoverageStatus,
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.orchestrator.sufficiency import (
    SufficiencyVerdict,
    fallback_verdict,
    parse_sufficiency_response,
    verdict_to_evidence,
)

TARGET = "outcome-target"
OTHER = "outcome-other"
TURN = "turn-1"


class _CapturingDB:
    """A DB whose begin_nested() is an async-context no-op (no real txn)."""

    def begin_nested(self) -> Any:
        class _Ctx:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *a: Any) -> bool:
                return False

        return _Ctx()


def _gateway(payload: dict[str, Any], calls: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    async def _gen(**kwargs: Any) -> SimpleNamespace:
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(content_json=payload)

    return SimpleNamespace(generate_json=AsyncMock(side_effect=_gen))


def _exploding_gateway() -> SimpleNamespace:
    async def _gen(**_kwargs: Any) -> SimpleNamespace:
        raise RuntimeError("gateway down")

    return SimpleNamespace(generate_json=AsyncMock(side_effect=_gen))


def _apply(verdict: SufficiencyVerdict, coverage: OutcomeCoverageState) -> None:
    for ev in verdict_to_evidence(
        verdict, turn_id=TURN, target_outcome_id=TARGET, allowed_other=(OTHER,)
    ):
        apply_evidence_to_coverage(coverage, ev, secondary=ev.secondary)


def _state(*coverages: OutcomeCoverageState) -> InterviewRuntimeStateData:
    data = InterviewRuntimeStateData()
    for cov in coverages:
        data.outcome_coverage[cov.outcome_id] = cov
    return data


def _analysis(*evidence: OutcomeEvidence, confidence: float = 0.9) -> AnswerAnalysis:
    return AnswerAnalysis(evidence=list(evidence), confidence=confidence)


# ── the verdict → coverage projection ────────────────────────────────────────


def test_sufficient_verdict_flips_provisional_sufficiency_in_one_turn() -> None:
    """Given a fresh outcome, when the probe says sufficient, then it is covered."""
    coverage = OutcomeCoverageState(outcome_id=TARGET)

    _apply(
        SufficiencyVerdict(sufficient=True, outcome_ids_touched=[TARGET], confidence=0.9), coverage
    )

    assert coverage.coverage_points == COVERAGE_SUFFICIENT_POINTS
    assert is_provisionally_sufficient(coverage.coverage_points)
    assert coverage.status is CoverageStatus.SUFFICIENT
    assert coverage.supporting_turn_ids == [TURN]


def test_partial_verdict_needs_two_turns_to_reach_sufficiency() -> None:
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    partial = SufficiencyVerdict(sufficient=False, outcome_ids_touched=[TARGET], confidence=0.8)

    _apply(partial, coverage)
    assert coverage.coverage_points == 1
    assert not is_provisionally_sufficient(coverage.coverage_points)
    assert coverage.status is CoverageStatus.PARTIAL

    _apply(partial, coverage)
    assert is_provisionally_sufficient(coverage.coverage_points)


def test_no_outcomes_touched_establishes_no_coverage() -> None:
    """An off-topic or wrong answer must not move coverage at all."""
    verdict = SufficiencyVerdict(sufficient=False, outcome_ids_touched=[], confidence=0.9)

    evidence = verdict_to_evidence(verdict, turn_id=TURN, target_outcome_id=None)

    assert evidence == []


def test_empty_id_list_awards_nothing_even_with_a_known_target() -> None:
    """An answer the model says demonstrated nothing must not earn coverage.

    Crediting the target here let two off-topic answers accumulate to a tick, so
    the candidate gained an outcome they never demonstrated and the interview
    advanced past it.
    """
    verdict = SufficiencyVerdict(sufficient=True, outcome_ids_touched=[], confidence=0.9)

    evidence = verdict_to_evidence(verdict, turn_id=TURN, target_outcome_id=TARGET)

    assert evidence == []


def test_low_confidence_verdict_earns_no_points() -> None:
    """The existing confidence floor gates a hedged probe — no probe-specific rule."""
    coverage = OutcomeCoverageState(outcome_id=TARGET)

    _apply(
        SufficiencyVerdict(sufficient=True, outcome_ids_touched=[TARGET], confidence=0.3), coverage
    )

    assert coverage.coverage_points == 0
    assert coverage.evidence_count == 1
    assert coverage.status is CoverageStatus.INSUFFICIENT


def test_unknown_outcome_id_cannot_create_phantom_coverage() -> None:
    verdict = SufficiencyVerdict(
        sufficient=True, outcome_ids_touched=["hallucinated"], confidence=0.99
    )

    evidence = verdict_to_evidence(
        verdict, turn_id=TURN, target_outcome_id=TARGET, allowed_other=(OTHER,)
    )

    assert evidence == []


def test_secondary_outcome_is_marked_and_clears_the_stricter_bar() -> None:
    verdict = SufficiencyVerdict(sufficient=True, outcome_ids_touched=[OTHER], confidence=0.6)

    evidence = verdict_to_evidence(
        verdict, turn_id=TURN, target_outcome_id=TARGET, allowed_other=(OTHER,)
    )

    assert [e.secondary for e in evidence] == [True]
    coverage = OutcomeCoverageState(outcome_id=OTHER)
    apply_evidence_to_coverage(coverage, evidence[0], secondary=True)
    # 0.6 clears EVIDENCE_CONFIDENCE_MIN but not the emergent bar of 0.75.
    assert coverage.coverage_points == 0


def test_duplicate_ids_are_collapsed() -> None:
    verdict = SufficiencyVerdict(
        sufficient=True, outcome_ids_touched=[TARGET, TARGET], confidence=0.9
    )

    evidence = verdict_to_evidence(verdict, turn_id=TURN, target_outcome_id=TARGET)

    assert len(evidence) == 1


# ── parsing ──────────────────────────────────────────────────────────────────


def test_parse_rejects_a_non_mapping_payload() -> None:
    assert parse_sufficiency_response(None) is None
    assert parse_sufficiency_response([]) is None  # type: ignore[arg-type]


def test_parse_coerces_a_sparse_payload_safely() -> None:
    verdict = parse_sufficiency_response({"sufficient": "yes"})

    assert verdict is not None
    assert verdict.sufficient is True
    assert verdict.outcome_ids_touched == []
    assert verdict.confidence == 0.0


def test_parse_clamps_confidence_and_drops_non_string_ids() -> None:
    verdict = parse_sufficiency_response(
        {"sufficient": True, "confidence": 7.5, "outcome_ids_touched": [TARGET, 3, None, "  "]}
    )

    assert verdict is not None
    assert verdict.confidence == 1.0
    assert verdict.outcome_ids_touched == [TARGET]


def test_fallback_verdict_is_inert() -> None:
    verdict = fallback_verdict()

    assert verdict.sufficient is False
    assert verdict.confidence == 0.0
    assert verdict_to_evidence(verdict, turn_id=TURN, target_outcome_id=TARGET) == []


# ── the probe call ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_returns_the_parsed_verdict() -> None:
    gateway = _gateway({"sufficient": True, "outcome_ids_touched": [TARGET], "confidence": 0.85})

    verdict = await sufficiency_logic.probe_sufficiency(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="Explain B-tree height.",
        student_answer="It stays logarithmic because the fanout is high.",
        outcome_id=TARGET,
        outcome_text="Explains logarithmic height",
        gateway=gateway,  # type: ignore[arg-type]
    )

    assert verdict.sufficient is True
    assert verdict.outcome_ids_touched == [TARGET]
    assert verdict.confidence == 0.85


@pytest.mark.asyncio
async def test_probe_prompt_carries_no_rubric_content() -> None:
    """The probe holds the question, the answer and the outcome statements only."""
    calls: list[dict[str, Any]] = []
    gateway = _gateway({"sufficient": False, "outcome_ids_touched": [], "confidence": 0.5}, calls)

    await sufficiency_logic.probe_sufficiency(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="q",
        student_answer="a",
        outcome_id=TARGET,
        outcome_text="Explains logarithmic height",
        other_outcomes=[{"id": OTHER, "text": "Communicates trade-offs"}],
        gateway=gateway,  # type: ignore[arg-type]
    )

    payload = json.loads(calls[0]["user_prompt"])
    assert set(payload) == {"current_question", "student_answer", "outcome", "other_outcomes"}
    assert "expected_evidence" not in payload
    assert "common_misconceptions" not in payload
    assert "supplementary_instructions" not in payload


@pytest.mark.asyncio
async def test_probe_failure_degrades_safely() -> None:
    """A dead gateway must not raise and must not invent coverage."""
    coverage = OutcomeCoverageState(outcome_id=TARGET)

    verdict = await sufficiency_logic.probe_sufficiency(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="q",
        student_answer="a real answer",
        outcome_id=TARGET,
        gateway=_exploding_gateway(),  # type: ignore[arg-type]
    )

    assert verdict == fallback_verdict()
    _apply(verdict, coverage)
    assert coverage.coverage_points == 0
    assert coverage.evidence_count == 0
    assert coverage.status is CoverageStatus.NOT_STARTED


@pytest.mark.asyncio
async def test_probe_malformed_payload_degrades_safely() -> None:
    verdict = await sufficiency_logic.probe_sufficiency(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="q",
        student_answer="a real answer",
        outcome_id=TARGET,
        gateway=_gateway("not a mapping"),  # type: ignore[arg-type]
    )

    assert verdict == fallback_verdict()


@pytest.mark.asyncio
async def test_probe_skips_the_call_for_an_empty_answer() -> None:
    calls: list[dict[str, Any]] = []
    gateway = _gateway({"sufficient": True}, calls)

    verdict = await sufficiency_logic.probe_sufficiency(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="q",
        student_answer="   ",
        outcome_id=TARGET,
        gateway=gateway,  # type: ignore[arg-type]
    )

    assert verdict == fallback_verdict()
    assert calls == []


# ── reconciliation against the authoritative full analysis ───────────────────


def test_reconcile_revokes_a_tick_the_full_analysis_does_not_support() -> None:
    """Given the probe ticked the outcome, when the full analysis finds only
    partial support, then the tick is revoked."""
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    probe = SufficiencyVerdict(sufficient=True, outcome_ids_touched=[TARGET], confidence=0.9)
    _apply(probe, coverage)
    assert is_provisionally_sufficient(coverage.coverage_points)
    data = _state(coverage)

    result = reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=probe,
        analysis=_analysis(
            OutcomeEvidence(
                outcome_id=TARGET,
                turn_id=TURN,
                evidence_type=EvidenceType.PARTIALLY_SUPPORTS,
                confidence=0.9,
            )
        ),
        target_outcome_id=TARGET,
    )

    assert coverage.coverage_points == 1
    assert not is_provisionally_sufficient(coverage.coverage_points)
    assert result.revoked_outcome_ids == [TARGET]
    assert result.changed is True
    assert coverage.evidence_count == 1


def test_reconcile_grants_coverage_the_probe_missed() -> None:
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    probe = SufficiencyVerdict(sufficient=False, outcome_ids_touched=[TARGET], confidence=0.9)
    _apply(probe, coverage)
    data = _state(coverage)

    result = reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=probe,
        analysis=_analysis(
            OutcomeEvidence(
                outcome_id=TARGET,
                turn_id=TURN,
                evidence_type=EvidenceType.SUPPORTS,
                confidence=0.9,
            )
        ),
        target_outcome_id=TARGET,
    )

    assert coverage.coverage_points == COVERAGE_SUFFICIENT_POINTS
    assert result.granted_outcome_ids == [TARGET]


def test_reconcile_is_a_no_op_when_probe_and_full_analysis_agree() -> None:
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    probe = SufficiencyVerdict(sufficient=True, outcome_ids_touched=[TARGET], confidence=0.9)
    _apply(probe, coverage)
    data = _state(coverage)

    result = reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=probe,
        analysis=_analysis(
            OutcomeEvidence(
                outcome_id=TARGET,
                turn_id=TURN,
                evidence_type=EvidenceType.SUPPORTS,
                confidence=0.9,
            )
        ),
        target_outcome_id=TARGET,
    )

    assert coverage.coverage_points == COVERAGE_SUFFICIENT_POINTS
    assert result.changed is False
    assert result.revoked_outcome_ids == []
    assert result.granted_outcome_ids == []


def test_reconcile_revokes_when_the_full_analysis_is_not_assessable() -> None:
    """A zero-confidence full read is a statement that the turn proved nothing."""
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    probe = SufficiencyVerdict(sufficient=True, outcome_ids_touched=[TARGET], confidence=0.9)
    _apply(probe, coverage)
    data = _state(coverage)

    result = reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=probe,
        analysis=_analysis(confidence=0.0),
        target_outcome_id=TARGET,
    )

    assert coverage.coverage_points == 0
    assert coverage.supporting_turn_ids == []
    assert result.revoked_outcome_ids == [TARGET]


def test_reconcile_preserves_coverage_earned_by_other_turns() -> None:
    """The reconciliation is a delta, so a later turn's points are untouched."""
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    probe = SufficiencyVerdict(sufficient=True, outcome_ids_touched=[TARGET], confidence=0.9)
    _apply(probe, coverage)
    apply_evidence_to_coverage(
        coverage,
        OutcomeEvidence(
            outcome_id=TARGET,
            turn_id="turn-2",
            evidence_type=EvidenceType.SUPPORTS,
            confidence=0.9,
        ),
    )
    assert coverage.coverage_points == 4
    data = _state(coverage)

    reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=probe,
        analysis=_analysis(confidence=0.0),
        target_outcome_id=TARGET,
    )

    assert coverage.coverage_points == 2
    assert coverage.supporting_turn_ids == ["turn-2"]


def test_reconcile_without_a_probe_applies_the_full_evidence_additively() -> None:
    coverage = OutcomeCoverageState(outcome_id=TARGET)
    data = _state(coverage)

    reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=None,
        analysis=_analysis(
            OutcomeEvidence(
                outcome_id=TARGET,
                turn_id=TURN,
                evidence_type=EvidenceType.SUPPORTS,
                confidence=0.9,
            )
        ),
        target_outcome_id=TARGET,
    )

    assert coverage.coverage_points == COVERAGE_SUFFICIENT_POINTS
    assert coverage.evidence_count == 1


def test_reconcile_creates_coverage_for_an_outcome_only_the_full_analysis_saw() -> None:
    data = _state()

    reconcile_turn_coverage(
        data,
        turn_id=TURN,
        probe_verdict=None,
        analysis=_analysis(
            OutcomeEvidence(
                outcome_id=OTHER,
                turn_id=TURN,
                evidence_type=EvidenceType.SUPPORTS,
                confidence=0.9,
                secondary=True,
            )
        ),
        target_outcome_id=TARGET,
        allowed_other=(OTHER,),
    )

    assert data.outcome_coverage[OTHER].coverage_points == COVERAGE_SUFFICIENT_POINTS
