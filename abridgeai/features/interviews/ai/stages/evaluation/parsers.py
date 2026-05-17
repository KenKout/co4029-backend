"""Parser for the interview EVALUATION stage LLM output (T6.8).

The stage prompt asks the judge to return a strict JSON object::

    {
      "criterion_scores": [
        {"criterion": "technical_accuracy",
         "score": 4,
         "justification": "Correctly identified ..."},
        ...
      ]
    }

This parser is intentionally permissive (mirrors the quiz validation
parser philosophy in §T5.7): malformed criterion rows are dropped
rather than fatal, and unknown criteria are kept so the rubric can
evolve without code changes. The caller (logic.py) is responsible for
filling missing criteria with a default 0-score so the rubric matrix
stays dense.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    CriterionScore,
    build_criterion_score,
)


def parse_evaluation_response(
    payload: Mapping[str, Any] | None,
    *,
    expected_criteria: tuple[str, ...] | list[str],
) -> list[CriterionScore]:
    """Normalise the raw LLM JSON into a list of :class:`CriterionScore`.

    Returns one entry per criterion in ``expected_criteria`` (in order).
    Missing or malformed entries default to ``score=0`` with an
    explanatory justification — better to record "no judgement" than
    silently inflate the average by skipping the criterion.
    """

    if not expected_criteria:
        return []

    by_criterion: dict[str, CriterionScore] = {}
    raw = payload.get("criterion_scores") if isinstance(payload, Mapping) else None
    if isinstance(raw, list):
        for entry in raw:
            score = _coerce_entry(entry)
            if score is not None:
                by_criterion[score.criterion] = score

    return [
        by_criterion.get(
            criterion,
            build_criterion_score(
                criterion=criterion,
                score=0.0,
                justification="Judge did not return a score for this criterion.",
            ),
        )
        for criterion in expected_criteria
    ]


def _coerce_entry(entry: object) -> CriterionScore | None:
    if not isinstance(entry, dict):
        return None

    criterion_raw = entry.get("criterion")
    if not isinstance(criterion_raw, str):
        return None
    criterion = criterion_raw.strip()
    if not criterion:
        return None

    score_raw = entry.get("score")
    if isinstance(score_raw, bool) or not isinstance(score_raw, (int, float, str)):
        return None
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        return None

    justification_raw = entry.get("justification") or entry.get("reason") or ""
    justification = str(justification_raw).strip() or "No justification provided."

    return build_criterion_score(
        criterion=criterion,
        score=score,
        justification=justification,
    )


__all__ = ["parse_evaluation_response"]
