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

# Emergent (non-target) evidence bar. Evidence attributed to an outcome that was
# NOT the current question's target is opportunistic: nobody asked about it, so a
# hedged read is far likelier to be the analyzer over-reaching than a real
# demonstration. Such evidence must clear this HIGHER bar to contribute coverage
# points; below it the item is still recorded (traceability) but is worth 0.
# Only consulted when the caller marks an item secondary (see
# :func:`evidence_points`); target evidence keeps using
# ``EVIDENCE_CONFIDENCE_MIN``.
EMERGENT_EVIDENCE_CONFIDENCE_MIN = 0.75


def evidence_points(evidence: OutcomeEvidence, *, secondary: bool = False) -> int:
    """Coverage points a single evidence item is worth.

    Low-confidence evidence contributes 0 regardless of type, so a hedged or
    uncertain read never establishes provisional coverage on its own.

    ``secondary=True`` marks *emergent* evidence — attributed to an outcome the
    current question did not target. It must clear the stricter
    ``EMERGENT_EVIDENCE_CONFIDENCE_MIN`` instead, so an unprompted attribution
    needs to be a confident read before it counts toward coverage.
    """
    floor = EMERGENT_EVIDENCE_CONFIDENCE_MIN if secondary else EVIDENCE_CONFIDENCE_MIN
    if evidence.confidence < floor:
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
    secondary: bool = False,
) -> None:
    """Fold one evidence item into an outcome's running coverage (in place).

    Updates the raw ``evidence_count`` (traceability), the weighted
    ``coverage_points`` (the value that actually gates sufficiency), the derived
    ``status``, and appends the originating turn id. This is the single place
    the 2/1/0 weighting is applied so the REST, adaptive, and controlled-end
    paths can never drift apart. No DB, no save — the caller owns persistence.

    ``secondary=True`` routes the item through the stricter emergent-evidence
    confidence bar (see :func:`evidence_points`). The item is always counted in
    ``evidence_count`` and its turn recorded, even when it earns 0 points, so a
    rejected attribution stays auditable.
    """
    coverage.evidence_count += 1
    coverage.coverage_points += evidence_points(evidence, secondary=secondary)
    coverage.status = _coverage_status_for(coverage.coverage_points, had_evidence=True)
    if now is not None:
        coverage.last_updated_at = now
    if evidence.turn_id not in coverage.supporting_turn_ids:
        coverage.supporting_turn_ids.append(evidence.turn_id)


def revoke_evidence_from_coverage(
    coverage: OutcomeCoverageState,
    evidence: OutcomeEvidence,
    *,
    now: str | None = None,
    secondary: bool = False,
) -> int:
    """Undo one previously-applied evidence item (in place); return points removed.

    The exact inverse of :func:`apply_evidence_to_coverage`, sharing its weighting
    so a revocation can never remove a different number of points than the
    application added. This exists for the async reconciliation path: the fast
    sufficiency probe establishes provisional coverage on the turn, and when the
    full extraction later disagrees, the probe's contribution has to come back
    out. A tick CAN therefore be revoked mid-session, which is legitimate —
    coverage is runtime selection guidance only and the post-session evaluator
    re-judges the transcript independently.

    ``supporting_turn_ids`` is deliberately NOT pruned here: only the caller
    knows whether the replacement evidence re-cites the same turn.
    """
    removed = evidence_points(evidence, secondary=secondary)
    coverage.evidence_count = max(0, coverage.evidence_count - 1)
    coverage.coverage_points = max(0, coverage.coverage_points - removed)
    coverage.status = _coverage_status_for(
        coverage.coverage_points, had_evidence=coverage.evidence_count > 0
    )
    if now is not None:
        coverage.last_updated_at = now
    return removed


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


def is_confidently_wrong(analysis: AnswerAnalysis | None) -> bool:
    """Relevant + specific + confidently INCORRECT/MIXED — the case to challenge.

    A candidate who committed assertively to a concrete claim that is wrong is
    exactly whom a real interviewer leans in on (Slice 16): the answer is on
    topic, specific enough to engage, and the analyzer is confident it is wrong
    or mixed. This is deliberately narrower than :func:`is_weak_answer`:

    * VAGUE answers are excluded — those are a clarify/probe case, not a
      challenge case (you can't challenge reasoning that wasn't really given).
    * OFF_TOPIC / PARTIALLY_RELEVANT answers are excluded — redirected elsewhere.
    * Low-confidence reads are excluded — we never lean in on a shaky signal.

    A missing analysis is never confidently wrong (returns False).
    """
    if analysis is None or analysis.confidence < STRONG_CONFIDENCE_MIN:
        return False
    return (
        analysis.relevance is Relevance.RELEVANT
        and analysis.correctness in (Correctness.INCORRECT, Correctness.MIXED)
        and analysis.specificity is not Specificity.VAGUE
    )


__all__ = [
    "COVERAGE_SUFFICIENT_POINTS",
    "EMERGENT_EVIDENCE_CONFIDENCE_MIN",
    "EVIDENCE_CONFIDENCE_MIN",
    "STRONG_CONFIDENCE_MIN",
    "apply_evidence_to_coverage",
    "evidence_points",
    "is_confidently_wrong",
    "is_provisionally_sufficient",
    "is_strong_answer",
    "is_weak_answer",
    "revoke_evidence_from_coverage",
]
