"""Per-outcome verdict data shapes for the interview EVALUATION stage.

Implements the thesis §4.3 contract: post-session AI judges the transcript
against EACH teacher-defined ``InterviewOutcome`` and returns a binary
**met / not-met** verdict with hidden reasoning + evidence — NOT a numeric
rubric score. The session pass/fail is later derived from how many outcomes
were met vs. the teacher-configured ``min_outcomes_to_pass`` (services layer).

This module is intentionally **pure**: no LLM calls, no DB I/O, no Jinja
rendering. The stage's :mod:`logic` module owns the side-effecting work and
hands the parsed verdicts here for aggregation.

Why separate from ``rubric.py``
-------------------------------
The rubric (``rubric.py``) is retained as a teacher-facing *diagnostic*
feeding the Gap Report; it no longer gates pass/fail. Keeping the two shapes
apart makes the "verdict gates, rubric informs" split explicit and keeps the
downstream Gap Report import surface unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class OutcomeVerdict:
    """One judge verdict on one outcome for the whole session.

    ``met`` is the binary pass signal for this outcome. ``reasoning`` is the
    judge's hidden justification (audit only — never shown to the student).
    ``evidence`` is an optional short quote from the candidate's answers that
    anchors the verdict.
    """

    outcome_id: UUID
    met: bool
    reasoning: str
    evidence: str | None = None


@dataclass(frozen=True)
class OutcomeVerdicts:
    """Stage output: one :class:`OutcomeVerdict` per configured outcome.

    ``met_count`` is the number of outcomes judged met — the input to the
    ``met_count >= min_outcomes_to_pass`` gate in the services layer.
    """

    verdicts: list[OutcomeVerdict] = field(default_factory=list)

    @property
    def met_count(self) -> int:
        return sum(1 for verdict in self.verdicts if verdict.met)

    @property
    def total(self) -> int:
        return len(self.verdicts)


def build_outcome_verdicts(verdicts: Sequence[OutcomeVerdict]) -> OutcomeVerdicts:
    """Wrap a sequence of verdicts into the aggregated stage output."""
    return OutcomeVerdicts(verdicts=list(verdicts))


__all__ = [
    "OutcomeVerdict",
    "OutcomeVerdicts",
    "build_outcome_verdicts",
]
