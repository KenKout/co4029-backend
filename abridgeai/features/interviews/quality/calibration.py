"""Coverage calibration: runtime self-belief vs the evaluator's verdicts.

Rule-based and pure — no LLM, no DB. The caller supplies the two things that
already exist in the database and this module compares them:

* the adaptive interviewer's *runtime* coverage points per outcome (what it
  believed while deciding what to ask next), and
* the post-session evaluator's independent ``verdict_met`` per outcome.

Why this matters: ``coverage.py`` deliberately documents runtime coverage as
"selection guidance ONLY — the post-session evaluator re-judges the transcript
independently". That independence is the right design, but it means nothing
verifies the runtime belief. If the selector routinely declares an outcome
sufficiently covered and moves on, while the evaluator later says the outcome
was NOT met, the interview is spending its turns badly and students are being
under-examined on exactly the outcomes they fail.

Two error directions, and they are not symmetric:

``over_confident``
    Runtime said sufficient, evaluator said not met. The costly one: the
    interviewer stopped probing an outcome the student had not actually
    demonstrated.

``under_confident``
    Runtime said insufficient, evaluator said met. Wasteful rather than unfair —
    turns were spent re-probing something already demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from abridgeai.features.interviews.orchestrator.coverage import (
    COVERAGE_SUFFICIENT_POINTS,
    is_provisionally_sufficient,
)


@dataclass(frozen=True)
class OutcomeCalibration:
    """Runtime belief vs evaluator verdict for ONE outcome."""

    outcome_id: str
    coverage_points: int
    runtime_sufficient: bool
    verdict_met: bool
    # Number of evidence items the runtime attributed (traceability only).
    evidence_count: int = 0

    @property
    def agrees(self) -> bool:
        return self.runtime_sufficient == self.verdict_met

    @property
    def over_confident(self) -> bool:
        """Runtime thought it had enough; the evaluator disagreed."""
        return self.runtime_sufficient and not self.verdict_met

    @property
    def under_confident(self) -> bool:
        """Runtime kept probing something the evaluator counted as met."""
        return not self.runtime_sufficient and self.verdict_met


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregate calibration for one session."""

    session_id: str
    outcomes: list[OutcomeCalibration] = field(default_factory=list)

    @property
    def scored(self) -> int:
        """Outcomes with BOTH a runtime state and an evaluator verdict."""
        return len(self.outcomes)

    @property
    def agreements(self) -> int:
        return sum(1 for o in self.outcomes if o.agrees)

    @property
    def over_confident(self) -> list[OutcomeCalibration]:
        return [o for o in self.outcomes if o.over_confident]

    @property
    def under_confident(self) -> list[OutcomeCalibration]:
        return [o for o in self.outcomes if o.under_confident]

    @property
    def agreement_rate(self) -> float | None:
        """Fraction of comparable outcomes where runtime and evaluator agree.

        ``None`` when nothing is comparable (no verdicts yet / no outcomes), so
        callers can distinguish "no data" from "perfectly miscalibrated".
        """
        if not self.outcomes:
            return None
        return self.agreements / len(self.outcomes)

    @property
    def over_confidence_rate(self) -> float | None:
        """Fraction of comparable outcomes the interview over-claimed.

        The headline number: this is the share of outcomes where the interviewer
        stopped probing but the evaluator found the outcome unmet.
        """
        if not self.outcomes:
            return None
        return len(self.over_confident) / len(self.outcomes)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "sufficiency_threshold": COVERAGE_SUFFICIENT_POINTS,
            "scored_outcomes": self.scored,
            "agreements": self.agreements,
            "agreement_rate": self.agreement_rate,
            "over_confident": [o.outcome_id for o in self.over_confident],
            "under_confident": [o.outcome_id for o in self.under_confident],
            "over_confidence_rate": self.over_confidence_rate,
            "outcomes": [
                {
                    "outcome_id": o.outcome_id,
                    "coverage_points": o.coverage_points,
                    "evidence_count": o.evidence_count,
                    "runtime_sufficient": o.runtime_sufficient,
                    "verdict_met": o.verdict_met,
                    "agrees": o.agrees,
                }
                for o in self.outcomes
            ],
        }


def compute_calibration(
    *,
    session_id: str,
    runtime_coverage: dict[str, dict[str, object]],
    verdicts: dict[str, bool],
) -> CalibrationReport:
    """Compare runtime coverage against evaluator verdicts.

    ``runtime_coverage`` is the ``outcome_coverage`` map straight out of
    ``interview_runtime_states.state_json`` — ``{outcome_id: {coverage_points,
    evidence_count, ...}}``. Extra keys are ignored so the state schema can keep
    evolving without breaking this.

    ``verdicts`` is ``{outcome_id: verdict_met}`` from
    ``interview_outcome_evaluations``.

    Only outcomes present in BOTH are scored: an outcome with no verdict has not
    been judged yet (nothing to calibrate against), and a verdict with no
    runtime entry means the adaptive path never ran for it (e.g. legacy session).
    Both are silently skipped rather than guessed at.
    """
    rows: list[OutcomeCalibration] = []
    for outcome_id, verdict_met in verdicts.items():
        state = runtime_coverage.get(outcome_id)
        if state is None:
            continue
        points = _as_int(state.get("coverage_points"))
        rows.append(
            OutcomeCalibration(
                outcome_id=outcome_id,
                coverage_points=points,
                runtime_sufficient=is_provisionally_sufficient(points),
                verdict_met=bool(verdict_met),
                evidence_count=_as_int(state.get("evidence_count")),
            )
        )
    rows.sort(key=lambda o: o.outcome_id)
    return CalibrationReport(session_id=session_id, outcomes=rows)


def _as_int(value: object) -> int:
    """Coerce a JSONB value to int, defaulting to 0 on anything unusable."""
    if isinstance(value, bool):  # bool is an int subclass; not a coverage count
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


__all__ = [
    "CalibrationReport",
    "OutcomeCalibration",
    "compute_calibration",
]
