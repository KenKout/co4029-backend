"""Rubric data shapes + aggregation helpers for the interview EVALUATION stage (T6.8).

This module is intentionally **pure**: no LLM calls, no DB I/O, no Jinja
rendering. The stage's :mod:`logic` module owns the side-effecting work
(LLM round-trips, parsing) and hands raw per-response evaluations to the
helpers here, which fold them into a stable :class:`RubricScores` shape.

Why a separate module
---------------------
T6.9 (Gap Report) and T6.11 (services persistence) both consume
:class:`RubricScores`. Keeping the dataclasses + aggregation here means
they can be imported without dragging in the LLM gateway or Jinja env,
which keeps test surfaces small for the downstream stages.

Default rubric
--------------
When ``InterviewConfig`` does not pin a custom rubric we fall back to
four standard criteria (matching the thesis UC-LEARN-02 step 9 rubric):

* ``technical_accuracy`` — factual correctness vs. source material.
* ``communication`` — clarity / structure of the response.
* ``problem_solving`` — reasoning quality, decomposition, trade-offs.
* ``professionalism`` — tone, hedging, recovery from gaps.

Scoring scale
-------------
Per-response scores are 0-5 (LLM judges). The aggregated ``total_score``
is rescaled to 0-100 percent so it slots cleanly into
``interview_sessions.total_score`` (NUMERIC(5,2) in baseline).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


DEFAULT_CRITERIA: tuple[str, ...] = (
    "technical_accuracy",
    "communication",
    "problem_solving",
    "professionalism",
)
"""Fallback rubric used when ``InterviewConfig`` ships no custom criteria.

Order is stable so JSONB output is deterministic — useful for diffing
fixtures and for the Gap Report which renders criteria in this order.
"""

_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 5.0
_PERCENT_SCALE: float = 100.0 / _SCORE_MAX  # 5 → 100


@dataclass(frozen=True)
class CriterionScore:
    """One judge verdict on one criterion for one response.

    ``score`` is clipped to ``[0, 5]`` at construction time via
    :func:`build_criterion_score` — the dataclass itself stays trivial
    so direct construction in tests / fixtures remains painless.
    """

    criterion: str
    score: float
    justification: str


@dataclass(frozen=True)
class ResponseEvaluation:
    """All criterion scores produced by a single LLM call for a single response."""

    session_question_id: UUID
    criterion_scores: list[CriterionScore] = field(default_factory=list)


@dataclass(frozen=True)
class RubricScores:
    """Stage output: per-response evaluations + aggregated session-level scores.

    ``aggregated`` maps each criterion → mean score (0-5) across all
    responses. ``total_score`` is the weighted aggregate rescaled to
    0-100 so it can be persisted directly to
    ``interview_sessions.total_score``.
    """

    response_evaluations: list[ResponseEvaluation]
    aggregated: dict[str, float]
    total_score: float


def resolve_rubric(config: Mapping[str, object] | None) -> dict[str, float]:
    """Read criterion → weight pairs from config, falling back to defaults.

    Recognised config shapes (precedence order):

    1. ``rubric_weights = {criterion: weight, ...}`` — explicit weights.
    2. ``rubric_criteria = [criterion, ...]`` — equal-weight list.

    Anything else returns the four-criterion default with equal weight.

    Weights are normalised to sum to ``1.0`` so callers don't have to
    care whether the config stored fractions, percentages, or raw
    counts. Negative or zero-sum inputs are rejected by falling back to
    equal weights — silently downgrading is preferable to crashing the
    evaluation pipeline on a malformed config.
    """

    explicit_weights = _coerce_weight_mapping(
        (config or {}).get("rubric_weights") if isinstance(config, Mapping) else None
    )
    if explicit_weights is not None:
        return _normalise_weights(explicit_weights)

    criteria_list = _coerce_criterion_list(
        (config or {}).get("rubric_criteria") if isinstance(config, Mapping) else None
    )
    if criteria_list:
        equal = 1.0 / len(criteria_list)
        return {criterion: equal for criterion in criteria_list}

    equal = 1.0 / len(DEFAULT_CRITERIA)
    return {criterion: equal for criterion in DEFAULT_CRITERIA}


def aggregate_rubric_scores(
    response_evaluations: Sequence[ResponseEvaluation],
    weights: Mapping[str, float],
) -> RubricScores:
    """Fold per-response evaluations into a session-level :class:`RubricScores`.

    Algorithm
    ---------
    1. For each criterion in ``weights``, average the scores across all
       responses that produced that criterion. Missing scores for a
       given response do not pull the average down — they're simply
       skipped (otherwise a single missing field would zero the
       criterion).
    2. ``total_score = SUM(criterion_mean * weight) * 20`` rescales the
       0-5 weighted aggregate to a 0-100 percent.
    3. Criteria with zero responses score 0 to keep the JSONB shape
       stable; the aggregate still works because their weight is
       multiplied by 0.
    """

    normalised_weights = _normalise_weights(dict(weights)) if weights else {}
    aggregated: dict[str, float] = {criterion: 0.0 for criterion in normalised_weights}

    for criterion in aggregated:
        scores = [
            cs.score
            for evaluation in response_evaluations
            for cs in evaluation.criterion_scores
            if cs.criterion == criterion
        ]
        if scores:
            aggregated[criterion] = sum(scores) / len(scores)

    total = (
        sum(
            aggregated[criterion] * normalised_weights[criterion]
            for criterion in normalised_weights
        )
        * _PERCENT_SCALE
    )

    return RubricScores(
        response_evaluations=list(response_evaluations),
        aggregated=aggregated,
        total_score=round(total, 2),
    )


def build_criterion_score(criterion: str, score: float, justification: str) -> CriterionScore:
    """Construct a :class:`CriterionScore` with the score clipped to ``[0, 5]``.

    The LLM occasionally over-shoots (e.g. emits ``5.5`` for a perfect
    answer or ``-1`` for a non-response). Clipping at the boundary keeps
    downstream aggregation honest without rejecting otherwise-valid
    rows.
    """

    clipped = max(_SCORE_MIN, min(_SCORE_MAX, float(score)))
    return CriterionScore(
        criterion=criterion,
        score=clipped,
        justification=justification.strip(),
    )


def _coerce_weight_mapping(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    weights: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weights[key] = weight
    return weights or None


def _coerce_criterion_list(value: object) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weight for weight in weights.values() if weight > 0)
    if total <= 0:
        equal = 1.0 / len(weights) if weights else 1.0
        return {criterion: equal for criterion in weights}
    return {criterion: weight / total for criterion, weight in weights.items() if weight > 0}


__all__ = [
    "DEFAULT_CRITERIA",
    "CriterionScore",
    "ResponseEvaluation",
    "RubricScores",
    "aggregate_rubric_scores",
    "build_criterion_score",
    "resolve_rubric",
]
