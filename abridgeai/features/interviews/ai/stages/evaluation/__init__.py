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
    evaluate_session,
)
from abridgeai.features.interviews.ai.stages.evaluation.parsers import (
    parse_evaluation_response,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    DEFAULT_CRITERIA,
    CriterionScore,
    ResponseEvaluation,
    RubricScores,
    aggregate_rubric_scores,
    build_criterion_score,
    resolve_rubric,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "EVALUATION_STAGE_NAME",
    "CriterionScore",
    "ResponseEvaluation",
    "RubricScores",
    "aggregate_rubric_scores",
    "build_criterion_score",
    "evaluate_session",
    "parse_evaluation_response",
    "resolve_rubric",
]
