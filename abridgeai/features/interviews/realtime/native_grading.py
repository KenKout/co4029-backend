"""Grade one native-path turn: fast probe now, full re-analysis later.

This is the seam between the two halves of the native architecture. The agent
supplies the conversation; ``orchestrator/sufficiency_logic`` supplies a cheap
verdict; ``features/interviews/workers/analysis`` supplies the authoritative
re-analysis. Without something joining them, ``outcome_coverage`` never moves on a
spoken turn: the state note the agent reads says "NOT yet covered" for the whole
interview, ``interview_next_question`` refuses until its bounded budget gives way,
and the hard-stop timer becomes the only thing that can end the session.

Why the split, restated because it is the load-bearing decision: the candidate is
waiting, so the turn spends ~0.7s on a ~59-token verdict that is just enough to
move coverage points. The full extraction (evidence text, contradictions,
per-outcome summaries — ~1.3s and ~300 tokens) is enqueued and reconciled off the
turn path, where it may take minutes and may REVOKE what the probe awarded. That
is legitimate: ``orchestrator/coverage.py`` documents these numbers as runtime
selection guidance only, never the grade.

Dependencies arrive as callables rather than being imported here, so this module
stays free of DB and gateway concerns and the ordering guarantees above are
testable with plain objects.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from abridgeai.features.interviews.orchestrator.coverage import apply_evidence_to_coverage
from abridgeai.features.interviews.orchestrator.state import OutcomeCoverageState
from abridgeai.features.interviews.orchestrator.sufficiency import verdict_to_evidence

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
    from abridgeai.features.interviews.orchestrator.sufficiency import SufficiencyVerdict

logger = logging.getLogger(__name__)


class _Probe(Protocol):
    async def __call__(
        self,
        *,
        question_text: str,
        answer_text: str,
        outcome_id: str | None,
        allowed_other: tuple[str, ...],
    ) -> SufficiencyVerdict: ...


async def grade_native_turn(
    *,
    state: InterviewRuntimeStateData,
    answer_text: str,
    question_text: str,
    turn_id: str,
    probe: _Probe,
    enqueue_reconcile: Callable[..., Awaitable[None]],
    save_state: Callable[[], Awaitable[None]],
    allowed_other_outcome_ids: tuple[str, ...] = (),
) -> None:
    """Fold this answer into coverage, then defer the authoritative analysis.

    Never raises. A dead gateway must not end a live graded interview, so a failed
    probe leaves coverage untouched — no phantom points — and the turn continues.
    State is persisted either way, because the tools mutated it this turn (hint
    ladder, refusal counters) and a rejoin that reset those bounds would hand a
    stubborn model a fresh budget to argue with.
    """
    if not answer_text.strip():
        # Silence or an empty transcript is not an answer; grading it would cost a
        # call and could only ever produce "touched nothing".
        await save_state()
        return

    verdict: SufficiencyVerdict | None = None
    try:
        verdict = await probe(
            question_text=question_text,
            answer_text=answer_text,
            outcome_id=state.current_outcome_id,
            allowed_other=allowed_other_outcome_ids,
        )
    except Exception:  # noqa: BLE001 -- grading is best-effort; the turn is not
        logger.exception("sufficiency probe failed; turn continues ungraded (turn=%s)", turn_id)

    if verdict is not None:
        _apply_verdict(state, verdict, turn_id=turn_id, allowed=allowed_other_outcome_ids)
        try:
            # Enqueued AFTER the fold so the worker's delta is computed against
            # exactly what the probe awarded.
            await enqueue_reconcile(turn_id=turn_id, probe_verdict=verdict.to_dict())
        except Exception:  # noqa: BLE001 -- a lost reconcile costs precision, not the turn
            logger.exception("could not enqueue turn reconciliation (turn=%s)", turn_id)

    await save_state()


def _apply_verdict(
    state: InterviewRuntimeStateData,
    verdict: SufficiencyVerdict,
    *,
    turn_id: str,
    allowed: tuple[str, ...],
) -> None:
    """Fold the verdict's evidence into ``outcome_coverage``.

    Uses the SAME weighting the full analysis does (``apply_evidence_to_coverage``)
    rather than probe-specific arithmetic, so the two paths cannot drift and the
    worker's later reconciliation is a clean delta.
    """
    for evidence in verdict_to_evidence(
        verdict,
        turn_id=turn_id,
        target_outcome_id=state.current_outcome_id,
        allowed_other=allowed,
    ):
        coverage = state.outcome_coverage.get(evidence.outcome_id)
        if coverage is None:
            coverage = OutcomeCoverageState(outcome_id=evidence.outcome_id)
            state.outcome_coverage[evidence.outcome_id] = coverage
        apply_evidence_to_coverage(coverage, evidence, secondary=evidence.secondary)


__all__ = ["grade_native_turn"]
