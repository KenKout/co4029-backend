"""Reconcile the fast probe's provisional coverage against the full analysis.

Pure — NO DB, NO LLM, NO state save. The caller owns persistence, exactly like
:mod:`orchestrator.coverage` and :mod:`orchestrator.turn_state`.

The split this module closes: on the turn path a
:class:`~orchestrator.sufficiency.SufficiencyVerdict` established provisional
coverage from a ~30-token judgement, because the candidate cannot wait for the
full evidence extraction. That extraction then runs in the worker and is
authoritative. When the two disagree, the probe's contribution is removed and
the full analysis's evidence is applied in its place.

Coverage can therefore go DOWN — an outcome the agent was told was covered can
become uncovered again. That is by design, and it is safe for the reason stated
in :mod:`orchestrator.coverage`'s docstring: these numbers are runtime SELECTION
GUIDANCE only, and the post-session evaluator re-judges the transcript
independently and is never bound by them. The alternative — letting a cheap
one-bit read permanently outvote the full extraction — would silently degrade
question selection for the rest of the session.

Applying a delta (revoke the probe's items, apply the full items) rather than
recomputing the outcome from scratch is what makes this correct to run late:
addition is commutative, so turns that landed in between are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator.coverage import (
    apply_evidence_to_coverage,
    is_provisionally_sufficient,
    revoke_evidence_from_coverage,
)
from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState
from abridgeai.features.interviews.orchestrator.sufficiency import verdict_to_evidence

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis
    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
    from abridgeai.features.interviews.orchestrator.sufficiency import SufficiencyVerdict


@dataclass(frozen=True)
class ReconciliationResult:
    """What the reconciliation changed, for logging and for tests.

    ``revoked_outcome_ids`` are the outcomes that were provisionally sufficient
    before and are not any more — the tick the agent may already have acted on.
    """

    changed: bool = False
    revoked_outcome_ids: list[str] = field(default_factory=list)
    granted_outcome_ids: list[str] = field(default_factory=list)
    points_before: dict[str, int] = field(default_factory=dict)
    points_after: dict[str, int] = field(default_factory=dict)


def reconcile_turn_coverage(
    data: InterviewRuntimeStateData,
    *,
    turn_id: str,
    probe_verdict: SufficiencyVerdict | None,
    analysis: AnswerAnalysis,
    target_outcome_id: str | None = None,
    allowed_other: tuple[str, ...] = (),
    now: str | None = None,
) -> ReconciliationResult:
    """Replace one turn's probe-derived coverage with the full analysis's (in place).

    ``probe_verdict`` of None means no probe ran for this turn, so there is
    nothing to revoke and the full evidence is applied additively.

    A full analysis with zero confidence is treated the same way the runtime path
    treats it (see ``turn_state.apply_state_updates``): its evidence is not
    applied. The probe's contribution is STILL revoked in that case — a
    not-assessable full read is a positive statement that the turn established
    nothing, so leaving the probe's optimistic tick standing would be the exact
    disagreement this function exists to resolve.
    """
    probe_evidence = (
        verdict_to_evidence(
            probe_verdict,
            turn_id=turn_id,
            target_outcome_id=target_outcome_id,
            allowed_other=allowed_other,
        )
        if probe_verdict is not None
        else []
    )
    full_evidence = list(analysis.evidence) if analysis.confidence > 0.0 else []

    touched = {ev.outcome_id for ev in probe_evidence} | {ev.outcome_id for ev in full_evidence}
    points_before = {
        oid: cov.coverage_points for oid, cov in data.outcome_coverage.items() if oid in touched
    }
    sufficient_before = {
        oid for oid, points in points_before.items() if is_provisionally_sufficient(points)
    }

    for ev in probe_evidence:
        cov = data.outcome_coverage.get(ev.outcome_id)
        if cov is None:
            continue
        revoke_evidence_from_coverage(cov, ev, now=now, secondary=ev.secondary)

    for ev in full_evidence:
        cov = data.outcome_coverage.get(ev.outcome_id)
        if cov is None:
            cov = OutcomeCoverageState(outcome_id=ev.outcome_id)
            data.outcome_coverage[ev.outcome_id] = cov
        apply_evidence_to_coverage(cov, ev, now=now, secondary=ev.secondary)

    _prune_orphaned_turn_ids(
        data,
        turn_id=turn_id,
        revoked_outcome_ids=[ev.outcome_id for ev in probe_evidence],
        retained_outcome_ids={ev.outcome_id for ev in full_evidence},
    )

    points_after = {
        oid: cov.coverage_points for oid, cov in data.outcome_coverage.items() if oid in touched
    }
    sufficient_after = {
        oid for oid, points in points_after.items() if is_provisionally_sufficient(points)
    }
    return ReconciliationResult(
        changed=points_before != points_after,
        revoked_outcome_ids=sorted(sufficient_before - sufficient_after),
        granted_outcome_ids=sorted(sufficient_after - sufficient_before),
        points_before=points_before,
        points_after=points_after,
    )


def _prune_orphaned_turn_ids(
    data: InterviewRuntimeStateData,
    *,
    turn_id: str,
    revoked_outcome_ids: list[str],
    retained_outcome_ids: set[str],
) -> None:
    """Drop the turn from outcomes whose only citation of it was the probe's.

    Without this the audit trail would claim a turn supports an outcome that,
    after reconciliation, has no evidence from it at all.
    """
    for oid in revoked_outcome_ids:
        if oid in retained_outcome_ids:
            continue
        cov = data.outcome_coverage.get(oid)
        if cov is not None and turn_id in cov.supporting_turn_ids:
            cov.supporting_turn_ids.remove(turn_id)


__all__ = ["ReconciliationResult", "reconcile_turn_coverage"]
