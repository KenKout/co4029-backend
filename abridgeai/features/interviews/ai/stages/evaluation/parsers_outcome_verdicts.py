"""Parser for the per-outcome verdict LLM output (thesis §4.3).

The verdict prompt asks the judge to return strict JSON::

    {
      "verdicts": [
        {"outcome_id": "<uuid>",
         "met": true,
         "reasoning": "Candidate explained ...",
         "evidence": "the part where they said ..."},
        ...
      ]
    }

This parser is permissive (mirrors the rubric parser philosophy): malformed
rows are dropped, unknown outcome ids are ignored, and any expected outcome the
judge omitted defaults to ``met=False`` with an explanatory reasoning. Defaulting
a missing outcome to *not met* is the safe direction — it never inflates a pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
)

_MISSING_REASONING = "Judge returned no verdict for this outcome; defaulting to not met."
_REASONING_LIMIT = 2000
_EVIDENCE_LIMIT = 500


def parse_outcome_verdicts(
    payload: Mapping[str, Any] | None,
    *,
    expected_outcome_ids: Sequence[UUID],
) -> list[OutcomeVerdict]:
    """Normalise raw LLM JSON into one :class:`OutcomeVerdict` per expected id.

    Returns verdicts in the order of ``expected_outcome_ids``. Outcomes the
    judge did not score (or scored malformed) default to ``met=False``.
    """
    if not expected_outcome_ids:
        return []

    by_outcome: dict[UUID, OutcomeVerdict] = {}
    raw = payload.get("verdicts") if isinstance(payload, Mapping) else None
    if isinstance(raw, list):
        for entry in raw:
            verdict = _coerce_entry(entry)
            if verdict is not None:
                by_outcome[verdict.outcome_id] = verdict

    return [
        by_outcome.get(
            outcome_id,
            OutcomeVerdict(
                outcome_id=outcome_id,
                met=False,
                reasoning=_MISSING_REASONING,
                evidence=None,
            ),
        )
        for outcome_id in expected_outcome_ids
    ]


def _coerce_entry(entry: object) -> OutcomeVerdict | None:
    if not isinstance(entry, dict):
        return None

    outcome_id = _coerce_uuid(entry.get("outcome_id"))
    if outcome_id is None:
        return None

    met = _coerce_bool(entry.get("met"))
    if met is None:
        return None

    reasoning_raw = entry.get("reasoning") or entry.get("justification") or ""
    reasoning = str(reasoning_raw).strip()[:_REASONING_LIMIT] or "No reasoning provided."

    evidence_raw = entry.get("evidence")
    evidence: str | None = None
    if isinstance(evidence_raw, str) and evidence_raw.strip():
        evidence = evidence_raw.strip()[:_EVIDENCE_LIMIT]

    return OutcomeVerdict(
        outcome_id=outcome_id,
        met=met,
        reasoning=reasoning,
        evidence=evidence,
    )


def _coerce_uuid(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value.strip())
        except (ValueError, AttributeError):
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    """Accept true/false JSON booleans and common string spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("true", "met", "yes", "pass", "passed"):
            return True
        if cleaned in ("false", "not_met", "not met", "no", "fail", "failed"):
            return False
    return None


__all__ = ["parse_outcome_verdicts"]
