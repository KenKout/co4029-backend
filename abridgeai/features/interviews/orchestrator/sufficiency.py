"""Fast sufficiency verdict — pure types, parser, and coverage projection.

The blocking turn path needs exactly one thing from the model: did this answer
move the current learning outcome toward sufficiency. That is what a
:class:`SufficiencyVerdict` carries, and nothing else — no evidence prose, no
contradictions, no per-outcome summaries. Those belong to the full extraction
(:mod:`orchestrator.analysis_logic`), which now runs off the turn path and is
reconciled in afterwards.

The projection back onto coverage deliberately reuses the EXISTING weighting in
:mod:`orchestrator.coverage` rather than introducing a second scale: a verdict is
turned into ordinary :class:`OutcomeEvidence` items, so ``evidence_points`` /
``apply_evidence_to_coverage`` / ``is_provisionally_sufficient`` behave exactly
as they do for a full analysis. There is no probe-specific arithmetic anywhere.

The I/O lives in :mod:`orchestrator.sufficiency_logic`, mirroring the
analysis / extraction / intent split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from abridgeai.features.interviews.orchestrator.analysis import (
    EvidenceType,
    OutcomeEvidence,
)

# Evidence summary attached to a projected item. Deliberately a fixed string:
# the probe returns no prose, and inventing a summary here would put words in
# the audit record that no model produced.
PROBE_EVIDENCE_SUMMARY = "fast sufficiency probe"


@dataclass
class SufficiencyVerdict:
    """The minimal per-turn coverage signal.

    ``sufficient`` is the model's judgement that the touched outcomes are NOW
    adequately demonstrated. ``outcome_ids_touched`` is the set of outcomes this
    answer gave real evidence for — an empty list is the correct, common answer
    for an off-topic or wrong turn and establishes no coverage at all.
    """

    sufficient: bool = False
    outcome_ids_touched: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "outcome_ids_touched": list(self.outcome_ids_touched),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SufficiencyVerdict:
        data = data or {}
        return cls(
            sufficient=bool(data.get("sufficient", False)),
            outcome_ids_touched=_id_list(data.get("outcome_ids_touched")),
            confidence=_confidence(data.get("confidence")),
        )


def parse_sufficiency_response(payload: Mapping[str, Any] | None) -> SufficiencyVerdict | None:
    """Coerce the gateway JSON into a verdict, or None when it is not a mapping.

    Returning None (rather than a defaulted verdict) lets the caller distinguish
    "the model said nothing usable" from "the model said not-sufficient", which
    are different facts even though both establish no coverage.
    """
    if not isinstance(payload, Mapping):
        return None
    return SufficiencyVerdict.from_dict(payload)


def fallback_verdict() -> SufficiencyVerdict:
    """The safe verdict for a failed or unusable probe.

    Zero confidence and no touched outcomes, so :func:`verdict_to_evidence`
    yields nothing and coverage cannot move. A probe failure must never invent
    coverage, and must never block the turn either — the full extraction will
    supply the real evidence when it lands.
    """
    return SufficiencyVerdict()


def verdict_to_evidence(
    verdict: SufficiencyVerdict,
    *,
    turn_id: str,
    target_outcome_id: str | None = None,
    allowed_other: Iterable[str] = (),
) -> list[OutcomeEvidence]:
    """Project a verdict onto the evidence items the coverage weighting consumes.

    * ``sufficient`` → ``SUPPORTS`` (2 points), which flips
      :func:`coverage.is_provisionally_sufficient` in a single turn.
    * not sufficient but the outcome was touched → ``PARTIALLY_SUPPORTS``
      (1 point), so two partial turns accumulate to sufficiency exactly as they
      do under the full analysis.
    * no outcomes touched → no items at all → coverage unchanged.

    The verdict's confidence is carried onto every item, so the existing
    confidence floors (``EVIDENCE_CONFIDENCE_MIN`` for the target outcome,
    ``EMERGENT_EVIDENCE_CONFIDENCE_MIN`` for a secondary one) gate a hedged
    probe down to 0 points without any extra logic here.

    A verdict with zero confidence makes no claim about any outcome and yields
    nothing — not even a 0-point item. This mirrors ``turn_state`` skipping the
    whole evidence fold for a zero-confidence analysis, and it is what keeps a
    failed probe out of ``evidence_count`` and out of ``supporting_turn_ids``.

    Ids are sanitized against the same allowlist rule as the full analysis: the
    target outcome, or an id explicitly offered in ``allowed_other``. Anything
    else is dropped, so a hallucinated id can never create a phantom outcome.

    There is deliberately NO repair when nothing survives the filter. Both ways
    that happens are untrustworthy signals: a verdict naming no outcome is the
    model reporting that the answer demonstrated none, and a verdict naming only
    unknown ids is a model that could not repeat an id it was handed — in which
    case its ``sufficient`` claim is not worth acting on either. In a graded
    assessment an ambiguous signal must award nothing: a missed credit costs one
    more follow-up, while phantom credit ticks an outcome the candidate never
    demonstrated and the interview advances past it.
    """
    if verdict.confidence <= 0.0:
        return []

    target = str(target_outcome_id) if target_outcome_id else None
    allowed = {str(o) for o in allowed_other if o}
    evidence_type = EvidenceType.SUPPORTS if verdict.sufficient else EvidenceType.PARTIALLY_SUPPORTS

    ids = [oid for oid in verdict.outcome_ids_touched if oid == target or oid in allowed]

    seen: set[str] = set()
    out: list[OutcomeEvidence] = []
    for oid in ids:
        if oid in seen:
            continue
        seen.add(oid)
        out.append(
            OutcomeEvidence(
                outcome_id=oid,
                turn_id=turn_id,
                evidence_type=evidence_type,
                summary=PROBE_EVIDENCE_SUMMARY,
                confidence=verdict.confidence,
                secondary=oid != target,
            )
        )
    return out


def _id_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned[:200])
    return out


def _confidence(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


__all__ = [
    "PROBE_EVIDENCE_SUMMARY",
    "SufficiencyVerdict",
    "fallback_verdict",
    "parse_sufficiency_response",
    "verdict_to_evidence",
]
