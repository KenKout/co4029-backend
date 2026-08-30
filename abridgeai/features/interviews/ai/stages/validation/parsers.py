"""LLM verdict parser for the interview VALIDATION stage (T6.6).

The validation pipeline combines four deterministic checks
(GROUNDED / DIFFICULTY_COHERENT / TYPE_MATCHES_CONFIG /
LENGTH_REASONABLE) with two LLM judgements: per-question NOT_LEADING
and complete-group VARIANT_TOPIC_COHERENT.

Leading verdict parsing is permissive: malformed per-question entries
default to "not leading". Complete variant-group coherence is deliberately
fail-closed: a missing, malformed, duplicate, or out-of-scope group verdict
rejects that expected group atomically.
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


@dataclass(frozen=True)
class GroupCoherenceVerdict:
    group_index: int
    topic_coherent: bool
    outlier_question_indices: tuple[int, ...] = ()
    rationale: str = ""
    is_default: bool = False


def parse_group_coherence_verdicts(
    payload: Mapping[str, Any] | None,
    *,
    expected_groups: Mapping[int, frozenset[int]],
) -> list[GroupCoherenceVerdict]:
    """Return one strict semantic-coherence verdict per expected group."""
    by_group: dict[int, GroupCoherenceVerdict] = {}
    duplicate_groups: set[int] = set()
    raw = payload.get("group_verdicts") if isinstance(payload, Mapping) else None
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            raw_index = entry.get("group_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                continue
            if raw_index not in expected_groups:
                continue
            if raw_index in by_group:
                duplicate_groups.add(raw_index)
                continue
            coherent = entry.get("topic_coherent")
            if not isinstance(coherent, bool):
                continue
            raw_outliers = entry.get("outlier_question_indices", [])
            if not isinstance(raw_outliers, list) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in raw_outliers
            ):
                continue
            outliers = tuple(dict.fromkeys(raw_outliers))
            if not set(outliers).issubset(expected_groups[raw_index]):
                continue
            if coherent and outliers:
                continue
            rationale_raw = entry.get("rationale")
            rationale = rationale_raw.strip()[:400] if isinstance(rationale_raw, str) else ""
            by_group[raw_index] = GroupCoherenceVerdict(
                group_index=raw_index,
                topic_coherent=coherent,
                outlier_question_indices=outliers,
                rationale=rationale,
            )

    defaults = duplicate_groups
    return [
        GroupCoherenceVerdict(
            group_index=group_index,
            topic_coherent=False,
            rationale="Validator did not return a valid group coherence verdict.",
            is_default=True,
        )
        if group_index in defaults or group_index not in by_group
        else by_group[group_index]
        for group_index in sorted(expected_groups)
    ]


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


__all__ = [
    "GroupCoherenceVerdict",
    "LeadingVerdict",
    "parse_group_coherence_verdicts",
    "parse_leading_verdicts",
]
