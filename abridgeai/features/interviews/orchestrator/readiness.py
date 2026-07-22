"""Adaptive-readiness analysis for the teacher authoring workspace (Slice 5).

Pure analysis — NO DB, NO LLM. Given a decoupled snapshot of an interview
config's approved questions + outcomes, it produces the set of *readiness
warnings* the authoring UI surfaces so a teacher can tell whether the adaptive
interviewer has enough structured material to adapt well:

* approved questions with no linked outcome (their evidence can't be scored),
* required outcomes with no approved question (can never be covered),
* approved questions missing a difficulty label (difficulty adaptation can't
  place them),
* lack of difficulty diversity (all questions the same level → no ramp),
* insufficient question coverage for the configured duration.

These are ADVISORY. They never block publishing — only the existing hard
publish requirements (>=1 approved question, >=1 outcome) do that, enforced
separately in ``services.authoring.publish_interview_config``. Keeping this
pure mirrors ``decision.py`` / ``selection.py`` / ``coverage.py`` and lets the
rules be unit-tested with plain objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Rough authoring heuristic: how many minutes of interview one approved
# question is expected to fill (opening + answer + a follow-up or two). Used
# only to advise when the pool looks thin for the configured duration.
_MINUTES_PER_QUESTION = 4


class ReadinessLevel(str, Enum):  # noqa: UP042 -- match codebase convention
    INFO = "info"
    WARNING = "warning"


class ReadinessCode(str, Enum):  # noqa: UP042 -- match codebase convention
    QUESTIONS_WITHOUT_OUTCOME = "questions_without_outcome"
    OUTCOMES_WITHOUT_QUESTION = "outcomes_without_question"
    QUESTIONS_MISSING_DIFFICULTY = "questions_missing_difficulty"
    LOW_DIFFICULTY_DIVERSITY = "low_difficulty_diversity"
    INSUFFICIENT_QUESTION_COVERAGE = "insufficient_question_coverage"


@dataclass(frozen=True)
class ReadinessQuestion:
    """An approved question, decoupled from the ORM row."""

    question_id: str
    linked_outcome_id: str | None
    difficulty: str | None


@dataclass(frozen=True)
class ReadinessOutcome:
    """A learning outcome, decoupled from the ORM row."""

    outcome_id: str
    importance_weight: int = 1


@dataclass(frozen=True)
class ReadinessInputs:
    """Everything the analysis needs, decoupled from the DB.

    ``questions`` are the APPROVED questions only (the pool the adaptive
    selector can actually draw from). ``time_limit_minutes`` is None for an
    untimed interview, in which case the coverage-for-duration check is skipped.
    """

    questions: list[ReadinessQuestion]
    outcomes: list[ReadinessOutcome]
    time_limit_minutes: int | None = None


@dataclass(frozen=True)
class ReadinessWarning:
    code: ReadinessCode
    level: ReadinessLevel
    # Machine-usable detail; the frontend localizes the human message from code.
    affected_ids: list[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "level": self.level.value,
            "affected_ids": list(self.affected_ids),
            "count": self.count,
        }


def analyze_readiness(inputs: ReadinessInputs) -> list[ReadinessWarning]:
    """Compute advisory adaptive-readiness warnings. Never raises, never blocks.

    The order is stable (declaration order below) so the UI renders
    deterministically and tests can assert positionally.
    """
    warnings: list[ReadinessWarning] = []
    questions = inputs.questions
    outcomes = inputs.outcomes

    # 1. Approved questions with no linked outcome — their evidence is orphaned.
    orphan_qs = [q.question_id for q in questions if not q.linked_outcome_id]
    if orphan_qs:
        warnings.append(
            ReadinessWarning(
                code=ReadinessCode.QUESTIONS_WITHOUT_OUTCOME,
                level=ReadinessLevel.WARNING,
                affected_ids=orphan_qs,
                count=len(orphan_qs),
            )
        )

    # 2. Outcomes with no approved question — can never be covered.
    linked_outcome_ids = {q.linked_outcome_id for q in questions if q.linked_outcome_id}
    uncovered_outcomes = [o.outcome_id for o in outcomes if o.outcome_id not in linked_outcome_ids]
    if uncovered_outcomes:
        warnings.append(
            ReadinessWarning(
                code=ReadinessCode.OUTCOMES_WITHOUT_QUESTION,
                level=ReadinessLevel.WARNING,
                affected_ids=uncovered_outcomes,
                count=len(uncovered_outcomes),
            )
        )

    # 3. Approved questions missing a difficulty label — can't be placed on the
    #    easy→hard ramp, so difficulty adaptation degrades to neutral for them.
    missing_difficulty = [q.question_id for q in questions if not q.difficulty]
    if missing_difficulty:
        warnings.append(
            ReadinessWarning(
                code=ReadinessCode.QUESTIONS_MISSING_DIFFICULTY,
                level=ReadinessLevel.WARNING,
                affected_ids=missing_difficulty,
                count=len(missing_difficulty),
            )
        )

    # 4. Lack of difficulty diversity — with 2+ questions all at one level there
    #    is no ramp for the streak adaptation to move along. Only meaningful once
    #    every question HAS a difficulty (else #3 is the actionable signal).
    labelled = [q.difficulty for q in questions if q.difficulty]
    if len(labelled) >= 2 and len(set(labelled)) == 1:
        warnings.append(
            ReadinessWarning(
                code=ReadinessCode.LOW_DIFFICULTY_DIVERSITY,
                level=ReadinessLevel.INFO,
                count=len(labelled),
            )
        )

    # 5. Insufficient question coverage for the configured duration.
    if inputs.time_limit_minutes is not None:
        expected = max(1, inputs.time_limit_minutes // _MINUTES_PER_QUESTION)
        if len(questions) < expected:
            warnings.append(
                ReadinessWarning(
                    code=ReadinessCode.INSUFFICIENT_QUESTION_COVERAGE,
                    level=ReadinessLevel.INFO,
                    count=len(questions),
                )
            )

    return warnings


__all__ = [
    "ReadinessCode",
    "ReadinessInputs",
    "ReadinessLevel",
    "ReadinessOutcome",
    "ReadinessQuestion",
    "ReadinessWarning",
    "analyze_readiness",
]
