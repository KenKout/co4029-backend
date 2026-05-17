"""LLM-as-judge scoring for the eval framework (T8.3).

Public re-exports. ``criteria`` exposes declarative criterion catalogs
per scenario capability; ``judge`` runs one LLM call per criterion and
returns a ``JudgeScore`` or ``PairwiseVerdict``.
"""

from __future__ import annotations

from eval.judges.criteria import (
    Criterion,
    criteria_for_capability,
    gap_report_criteria,
    interview_criteria,
    quiz_criteria,
)
from eval.judges.judge import (
    JudgeScore,
    PairwiseVerdict,
    judge_pairwise,
    judge_response,
)

__all__ = [
    "Criterion",
    "JudgeScore",
    "PairwiseVerdict",
    "criteria_for_capability",
    "gap_report_criteria",
    "interview_criteria",
    "judge_pairwise",
    "judge_response",
    "quiz_criteria",
]
