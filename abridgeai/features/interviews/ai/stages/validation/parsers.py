"""LLM verdict parser for the interview VALIDATION stage (T6.6).

The validation pipeline runs four deterministic checks in Python
(GROUNDED / DIFFICULTY_COHERENT / TYPE_MATCHES_CONFIG /
LENGTH_REASONABLE) plus one LLM judgement (NOT_LEADING). This module
isolates the LLM-side parsing so :mod:`logic` stays focused on
combining all five signals.

The parser is permissive by design: a flaky validator must never
silently reject every question, so missing or malformed entries
default to "not leading" with a synthetic rationale. Callers can
distinguish "validator was silent" from "validator approved" via the
returned ``LeadingVerdict.is_default`` flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LeadingVerdict:
    """The validator's decision on a single question's neutrality."""

    question_index: int
    not_leading: bool
    rationale: str = ""
    is_default: bool = False


_DEFAULT_RATIONALE = "Validator did not return a verdict; accepted by default."


def parse_leading_verdicts(
    payload: Mapping[str, Any] | None,
    *,
    question_count: int,
) -> list[LeadingVerdict]:
    """Normalise the LLM JSON into one :class:`LeadingVerdict` per question.

    The returned list is positional — entry ``i`` corresponds to draft
    ``i``. Bad rows are dropped silently and missing positions get a
    default-accept verdict so a single bad row never poisons the batch.
    """

    if question_count <= 0:
        return []

    by_index: dict[int, LeadingVerdict] = {}
    raw = payload.get("verdicts") if isinstance(payload, Mapping) else None
    if isinstance(raw, list):
        for entry in raw:
            verdict = _coerce_entry(entry, question_count)
            if verdict is not None:
                by_index[verdict.question_index] = verdict

    return [
        by_index.get(
            index,
            LeadingVerdict(
                question_index=index,
                not_leading=True,
                rationale=_DEFAULT_RATIONALE,
                is_default=True,
            ),
        )
        for index in range(question_count)
    ]


def _coerce_entry(entry: object, question_count: int) -> LeadingVerdict | None:
    if not isinstance(entry, dict):
        return None

    index = _coerce_index(entry, question_count)
    if index is None:
        return None

    not_leading = _coerce_bool(entry.get("not_leading"))
    if not_leading is None:
        return None

    rationale_raw = entry.get("rationale")
    rationale = rationale_raw.strip() if isinstance(rationale_raw, str) else ""

    return LeadingVerdict(
        question_index=index,
        not_leading=not_leading,
        rationale=rationale[:400],
    )


def _coerce_index(entry: dict[str, Any], question_count: int) -> int | None:
    raw = entry.get("question_index")
    if raw is None:
        raw = entry.get("position")
        if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= question_count:
            return raw - 1
    if isinstance(raw, bool) or raw is None:
        return None
    if not isinstance(raw, (int, str, float)):
        return None
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return None
    if 0 <= index < question_count:
        return index
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "yes", "accept", "ok", "neutral"}:
            return True
        if cleaned in {"false", "no", "reject", "leading", "biased"}:
            return False
    return None


__all__ = ["LeadingVerdict", "parse_leading_verdicts"]
