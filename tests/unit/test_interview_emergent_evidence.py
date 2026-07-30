"""Unit tests for emergent (non-target) outcome evidence.

A candidate answering about LO1 often demonstrates LO3 in passing. Before this
feature that evidence was discarded: the analyzer only ever saw the ONE outcome
linked to the current question and could only attribute evidence to it.

Two halves are covered here, both pure (no DB, no LLM):

1. ``coverage.evidence_points`` / ``apply_evidence_to_coverage`` — emergent
   (``secondary=True``) evidence must clear a HIGHER confidence bar before it
   earns coverage points, while still being recorded for traceability.
2. ``analysis_logic._sanitize_evidence`` — the whitelist that stops a
   hallucinated / unknown outcome id from creating a phantom outcome, and that
   forces the ``secondary`` flag rather than trusting what the model claimed.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    EvidenceType,
    OutcomeEvidence,
)
from abridgeai.features.interviews.orchestrator.analysis_logic import _sanitize_evidence
from abridgeai.features.interviews.orchestrator.coverage import (
    EMERGENT_EVIDENCE_CONFIDENCE_MIN,
    EVIDENCE_CONFIDENCE_MIN,
    apply_evidence_to_coverage,
    evidence_points,
)
from abridgeai.features.interviews.orchestrator.state import (
    CoverageStatus,
    OutcomeCoverageState,
)


def _ev(
    *,
    confidence: float,
    outcome_id: str = "o1",
    etype: EvidenceType = EvidenceType.SUPPORTS,
    secondary: bool = False,
) -> OutcomeEvidence:
    return OutcomeEvidence(
        outcome_id=outcome_id,
        turn_id="t1",
        evidence_type=etype,
        confidence=confidence,
        secondary=secondary,
    )


# ── the emergent confidence bar ──────────────────────────────────────────────


def test_emergent_bar_is_stricter_than_target_bar() -> None:
    """Sanity-check the constants themselves, since the whole gate rests on it."""
    assert EMERGENT_EVIDENCE_CONFIDENCE_MIN > EVIDENCE_CONFIDENCE_MIN


def test_confidence_between_bars_counts_for_target_but_not_emergent() -> None:
    """The interesting band: good enough when asked, not good enough unprompted."""
    mid = (EVIDENCE_CONFIDENCE_MIN + EMERGENT_EVIDENCE_CONFIDENCE_MIN) / 2
    ev = _ev(confidence=mid)
    assert evidence_points(ev, secondary=False) == 2
    assert evidence_points(ev, secondary=True) == 0


def test_confident_emergent_evidence_counts_normally() -> None:
    ev = _ev(confidence=0.95)
    assert evidence_points(ev, secondary=True) == 2


def test_emergent_partial_support_still_worth_one_point() -> None:
    """The 2/1/0 type weighting is unchanged for emergent evidence."""
    ev = _ev(confidence=0.95, etype=EvidenceType.PARTIALLY_SUPPORTS)
    assert evidence_points(ev, secondary=True) == 1


def test_default_is_target_behaviour() -> None:
    """Omitting ``secondary`` must reproduce v1 behaviour exactly."""
    ev = _ev(confidence=EVIDENCE_CONFIDENCE_MIN)
    assert evidence_points(ev) == evidence_points(ev, secondary=False)


# ── apply_evidence_to_coverage with emergent evidence ────────────────────────


def test_rejected_emergent_evidence_is_still_recorded() -> None:
    """0 points, but the attempt stays auditable (count + turn id + status)."""
    cov = OutcomeCoverageState(outcome_id="o2")
    ev = _ev(confidence=0.6, outcome_id="o2", secondary=True)

    apply_evidence_to_coverage(cov, ev, secondary=True)

    assert cov.coverage_points == 0
    assert cov.evidence_count == 1  # recorded for traceability
    assert cov.supporting_turn_ids == ["t1"]
    assert cov.status is CoverageStatus.INSUFFICIENT  # saw evidence, earned nothing


def test_confident_emergent_evidence_reaches_sufficiency() -> None:
    cov = OutcomeCoverageState(outcome_id="o2")
    ev = _ev(confidence=0.9, outcome_id="o2", secondary=True)

    apply_evidence_to_coverage(cov, ev, secondary=True)

    assert cov.coverage_points == 2
    assert cov.status is CoverageStatus.SUFFICIENT


# ── _sanitize_evidence: the id whitelist ─────────────────────────────────────


def _analysis(*evidence: OutcomeEvidence) -> AnswerAnalysis:
    return AnswerAnalysis(evidence=list(evidence))


def test_unknown_outcome_id_is_dropped() -> None:
    """A hallucinated id must never create a phantom outcome in the map."""
    analysis = _analysis(_ev(confidence=0.9, outcome_id="ghost"))

    _sanitize_evidence(analysis, target_outcome_id="o1", allowed_other={"o2"})

    assert analysis.evidence == []


def test_target_evidence_is_kept_and_marked_primary() -> None:
    analysis = _analysis(_ev(confidence=0.9, outcome_id="o1", secondary=True))

    _sanitize_evidence(analysis, target_outcome_id="o1", allowed_other={"o2"})

    assert len(analysis.evidence) == 1
    assert analysis.evidence[0].secondary is False  # forced, model claim ignored


def test_whitelisted_other_outcome_is_forced_secondary() -> None:
    """Even if the model forgot the flag, a non-target id is secondary."""
    analysis = _analysis(_ev(confidence=0.9, outcome_id="o2", secondary=False))

    _sanitize_evidence(analysis, target_outcome_id="o1", allowed_other={"o2"})

    assert len(analysis.evidence) == 1
    assert analysis.evidence[0].secondary is True


def test_missing_id_is_repaired_to_target() -> None:
    """Real evidence for the question actually asked is not thrown away."""
    analysis = _analysis(_ev(confidence=0.9, outcome_id=""))

    _sanitize_evidence(analysis, target_outcome_id="o1", allowed_other=set())

    assert len(analysis.evidence) == 1
    assert analysis.evidence[0].outcome_id == "o1"
    assert analysis.evidence[0].secondary is False


def test_feature_off_drops_all_non_target_evidence() -> None:
    """Empty whitelist (flag off) → v1 behaviour: only target evidence survives."""
    analysis = _analysis(
        _ev(confidence=0.9, outcome_id="o1"),
        _ev(confidence=0.9, outcome_id="o2"),
        _ev(confidence=0.9, outcome_id="o3"),
    )

    _sanitize_evidence(analysis, target_outcome_id="o1", allowed_other=set())

    assert [e.outcome_id for e in analysis.evidence] == ["o1"]


def test_no_target_outcome_keeps_only_whitelisted() -> None:
    """A question with no linked outcome: nothing to repair to, so blanks drop."""
    analysis = _analysis(
        _ev(confidence=0.9, outcome_id=""),
        _ev(confidence=0.9, outcome_id="o2"),
    )

    _sanitize_evidence(analysis, target_outcome_id=None, allowed_other={"o2"})

    assert [e.outcome_id for e in analysis.evidence] == ["o2"]
    assert analysis.evidence[0].secondary is True
