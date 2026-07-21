"""Weighted provisional coverage + answer-quality policy (Slice 2).

Pure decision helpers — NO DB, NO LLM, NO state mutation. They encode two
things the brief specifies and that were previously either flat-counted or not
computed at all:

1. **Weighted coverage.** A single piece of evidence no longer counts as one
   flat point. Confident supporting evidence is worth 2 coverage points,
   confident partial support 1, and contradiction / insufficient / low-confidence
   evidence 0. An outcome becomes *provisionally sufficient* at
   ``COVERAGE_SUFFICIENT_POINTS`` (2). This is runtime selection guidance ONLY —
   the post-session evaluator re-judges the transcript independently and is never
   bound by these values.

2. **Strong / weak answer classification.** Used by difficulty adaptation
   (Slice 3): a *strong* answer is relevant, complete, correct-or-mostly-correct,
   and analyzed with confidence >= ``STRONG_CONFIDENCE_MIN``. A *weak* answer is
   confidently incorrect, insufficient, or vague. Everything else (mixed /
   low-confidence) is neutral and must not aggressively move either streak.

Keeping this pure mirrors ``decision.py`` / ``selection.py`` and lets the
weighting rules be unit-tested with plain enums, independent of the DB path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator.analysis import (
    Completeness,
    Correctness,
    EvidenceType,
    Relevance,
    Specificity,
)
from abridgeai.features.interviews.orchestrator.state import CoverageStatus

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.analysis import (
        AnswerAnalysis,
        OutcomeEvidence,
    )
    from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState

# ── weighted coverage ────────────────────────────────────────────────────────

# Coverage points awarded per confident evidence item, keyed by evidence type.
# Contradiction / insufficient contribute nothing. Low-confidence evidence is
# gated out separately (see EVIDENCE_CONFIDENCE_MIN) before this map applies.
_COVERAGE_POINTS: dict[EvidenceType, int] = {
    EvidenceType.SUPPORTS: 2,
    EvidenceType.PARTIALLY_SUPPORTS: 1,
    EvidenceType.CONTRADICTS: 0,
    EvidenceType.INSUFFICIENT: 0,
}

# An outcome is provisionally sufficient at (and above) this many coverage points.
COVERAGE_SUFFICIENT_POINTS = 2

# Evidence below this confidence establishes no coverage (it may still be
# recorded for traceability, but contributes 0 points).
EVIDENCE_CONFIDENCE_MIN = 0.5

# An answer is "strong" only when the analyzer is at least this confident.
STRONG_CONFIDENCE_MIN = 0.65


def evidence_points(evidence: OutcomeEvidence) -> int:
    """Coverage points a single evidence item is worth.

    Low-confidence evidence contributes 0 regardless of type, so a hedged or
    uncertain read never establishes provisional coverage on its own.
    """
    if evidence.confidence < EVIDENCE_CONFIDENCE_MIN:
        return 0
    return _COVERAGE_POINTS.get(evidence.evidence_type, 0)


def is_provisionally_sufficient(coverage_points: int) -> bool:
    """Whether an outcome has earned enough weighted coverage to be sufficient."""
    return coverage_points >= COVERAGE_SUFFICIENT_POINTS


def _coverage_status_for(points: int, *, had_evidence: bool) -> CoverageStatus:
    """Map accumulated weighted points to a coverage status.

    ``had_evidence`` distinguishes "no evidence seen yet" (NOT_STARTED) from
    "evidence seen but it established no coverage" (INSUFFICIENT), e.g. a turn
    that only contradicted or was insufficient.
    """
    if is_provisionally_sufficient(points):
        return CoverageStatus.SUFFICIENT
    if points > 0:
        return CoverageStatus.PARTIAL
    if had_evidence:
        return CoverageStatus.INSUFFICIENT
    return CoverageStatus.NOT_STARTED


def apply_evidence_to_coverage(
    coverage: OutcomeCoverageState,
    evidence: OutcomeEvidence,
    *,
    now: str | None = None,
) -> None:
    """Fold one evidence item into an outcome's running coverage (in place).

    Updates the raw ``evidence_count`` (traceability), the weighted
    ``coverage_points`` (the value that actually gates sufficiency), the derived
    ``status``, and appends the originating turn id. This is the single place
    the 2/1/0 weighting is applied so the REST, adaptive, and controlled-end
    paths can never drift apart. No DB, no save — the caller owns persistence.
    """
    coverage.evidence_count += 1
    coverage.coverage_points += evidence_points(evidence)
    coverage.status = _coverage_status_for(coverage.coverage_points, had_evidence=True)
    if now is not None:
        coverage.last_updated_at = now
    if evidence.turn_id not in coverage.supporting_turn_ids:
        coverage.supporting_turn_ids.append(evidence.turn_id)


# ── strong / weak answer classification ──────────────────────────────────────


def is_strong_answer(analysis: AnswerAnalysis | None) -> bool:
    """Relevant + complete + (mostly) correct + confidently analyzed.

    A missing analysis (non-academic turn or analyzer failure) is never strong.
    """
    if analysis is None:
        return False
    return (
        analysis.confidence >= STRONG_CONFIDENCE_MIN
        and analysis.relevance is Relevance.RELEVANT
        and analysis.completeness is Completeness.COMPLETE
        and analysis.correctness in (Correctness.CORRECT, Correctness.MOSTLY_CORRECT)
    )


def is_weak_answer(analysis: AnswerAnalysis | None) -> bool:
    """Confidently incorrect, insufficient, or vague.

    Low-confidence analysis is NOT weak — an uncertain read is neutral so it
    can't drag the difficulty streak down on noise. A missing analysis is
    likewise neutral (returns False).

    Strong and weak are mutually exclusive: an answer that already qualifies as
    strong (relevant + complete + (mostly) correct) is never also weak, so a
    merely-vague *phrasing* of a complete, correct answer does not count against
    the streak. Weakness is judged only among non-strong answers.
    """
    if analysis is None or analysis.confidence < STRONG_CONFIDENCE_MIN:
        return False
    if is_strong_answer(analysis):
        return False
    return (
        analysis.correctness is Correctness.INCORRECT
        or analysis.completeness is Completeness.INSUFFICIENT
        or analysis.specificity is Specificity.VAGUE
    )


__all__ = [
    "COVERAGE_SUFFICIENT_POINTS",
    "EVIDENCE_CONFIDENCE_MIN",
    "STRONG_CONFIDENCE_MIN",
    "apply_evidence_to_coverage",
    "evidence_points",
    "is_provisionally_sufficient",
    "is_strong_answer",
    "is_weak_answer",
]
