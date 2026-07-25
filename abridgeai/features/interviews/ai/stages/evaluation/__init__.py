"""Interview EVALUATION stage (T6.8).

Public API:
    evaluate_session(...)        — orchestrator entry point
    RubricScores                 — stage output dataclass (consumed by T6.9, T6.11)
    ResponseEvaluation           — per-response evaluation record
    CriterionScore               — single criterion judgement
    aggregate_rubric_scores(...) — pure aggregation helper (no LLM)
    parse_evaluation_response(...) — pure parser (testable in isolation)
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.evaluation.logic import (
    EVALUATION_STAGE_NAME,
    evaluate_outcomes,
    evaluate_session,
)
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    OutcomeVerdicts,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.parsers import (
    parse_evaluation_response,
)
from abridgeai.features.interviews.ai.stages.evaluation.parsers_outcome_verdicts import (
    parse_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    DEFAULT_CRITERIA,
    SUPPLEMENTARY_RUBRIC_KEY,
    CriterionScore,
    ResponseEvaluation,
    RubricDefinition,
    RubricScores,
    aggregate_rubric_scores,
    build_criterion_score,
    resolve_rubric,
    resolve_rubric_definition,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "EVALUATION_STAGE_NAME",
    "SUPPLEMENTARY_RUBRIC_KEY",
    "CriterionScore",
    "OutcomeVerdict",
    "OutcomeVerdicts",
    "ResponseEvaluation",
    "RubricDefinition",
    "RubricScores",
    "aggregate_rubric_scores",
    "build_criterion_score",
    "build_outcome_verdicts",
    "evaluate_outcomes",
    "evaluate_session",
    "parse_evaluation_response",
    "parse_outcome_verdicts",
    "resolve_rubric",
    "resolve_rubric_definition",
]
