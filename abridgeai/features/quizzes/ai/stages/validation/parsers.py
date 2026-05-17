"""Verdict normalisation for the quiz validation stage (T5.7).

Ports ``normalize_validation_verdicts`` from
``backend/app/ai/haystack/mappers/quiz.py:217-243`` into the new package
layout. The parser is intentionally permissive: it skips malformed
verdict entries (rather than raising) so a single bad row from the
validator does not poison the whole batch. Missing positions default
to ``accept`` because rejecting silently would drop perfectly good
questions on a flaky validator.

The :class:`Verdict` dataclass extends the legacy dict shape with
``reasons: list[str]`` (multiple defect codes per rejection) and
``evidence_excerpt`` (auditable snippet from the validator). Plain
``dict`` access is preserved via ``to_dict`` so legacy callers can keep
walking results by key while new callers get a typed handle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_ACCEPT_VALUES: frozenset[str] = frozenset({"accept", "approve", "ok", "pass"})
_REJECT_VALUES: frozenset[str] = frozenset({"reject", "deny", "fail"})

_DEFAULT_REASON = "Validator did not return a verdict; accepted by default."

_DEFECT_CODES: tuple[str, ...] = (
    "SOURCE_LEAK",
    "ANSWER_LEAK",
    "SHALLOW",
    "LENGTH_TELL",
    "SHAPE",
    "UNGROUNDED",
    "AMBIGUOUS",
    "META",
)


@dataclass(frozen=True)
class Verdict:
    """One validator decision for a single question.

    ``position`` is 1-based to mirror the prompt format. ``reasons`` is a
    list because the prompt allows multiple defect codes joined into one
    free-form ``reason`` field (e.g. ``"SHAPE; AMBIGUOUS: ..."``); the
    parser splits these into atomic codes for downstream filtering and
    keeps the original prose as the first element when no codes match.
    """

    position: int
    verdict: str  # "accept" | "reject"
    reasons: list[str] = field(default_factory=list)
    evidence_excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Mirror the legacy dict shape so callers can keep using ``[]``."""

        return {
            "position": self.position,
            "verdict": self.verdict,
            "reason": self.reasons[0] if self.reasons else None,
            "reasons": list(self.reasons),
            "evidence_excerpt": self.evidence_excerpt,
        }


def parse_validation_response(
    payload: Mapping[str, Any] | None,
    *,
    question_count: int,
) -> list[Verdict]:
    """Normalise the raw LLM JSON into a positional list of verdicts.

    Returns ``len(questions)`` entries (one per question, in order).
    Missing or malformed entries default to ``accept`` with a synthetic
    reason — see module docstring for rationale.
    """

    if question_count <= 0:
        return []

    by_position: dict[int, Verdict] = {}
    raw = payload.get("verdicts") if isinstance(payload, Mapping) else None
    if isinstance(raw, list):
        for entry in raw:
            verdict = _coerce_entry(entry, question_count)
            if verdict is not None:
                by_position[verdict.position] = verdict

    return [
        by_position.get(
            position,
            Verdict(
                position=position,
                verdict="accept",
                reasons=[_DEFAULT_REASON],
                evidence_excerpt=None,
            ),
        )
        for position in range(1, question_count + 1)
    ]


def _coerce_entry(entry: object, question_count: int) -> Verdict | None:
    if not isinstance(entry, dict):
        return None

    position_raw = entry.get("position")
    if position_raw is None or isinstance(position_raw, bool):
        return None
    if not isinstance(position_raw, (int, str, float)):
        return None
    try:
        position = int(position_raw)
    except (TypeError, ValueError):
        return None
    if not 1 <= position <= question_count:
        return None

    verdict = _normalize_verdict_value(entry.get("verdict"))
    if verdict is None:
        return None

    reason_raw = entry.get("reason")
    reasons: list[str] = []
    if isinstance(reason_raw, str):
        cleaned = reason_raw.strip()
        if cleaned:
            reasons = _split_defect_codes(cleaned)

    evidence_raw = entry.get("evidence_excerpt") or entry.get("evidence")
    evidence_excerpt: str | None = None
    if isinstance(evidence_raw, str):
        cleaned_evidence = evidence_raw.strip()
        if cleaned_evidence:
            evidence_excerpt = cleaned_evidence[:400]

    return Verdict(
        position=position,
        verdict=verdict,
        reasons=reasons,
        evidence_excerpt=evidence_excerpt,
    )


def _normalize_verdict_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned in _ACCEPT_VALUES:
        return "accept"
    if cleaned in _REJECT_VALUES:
        return "reject"
    return None


def _split_defect_codes(reason: str) -> list[str]:
    """Split a free-form reason into atomic ``CODE: prose`` fragments.

    The prompt invites validators to separate multiple codes with ``;``
    or ``,`` (e.g. ``"SHAPE; AMBIGUOUS: only B is correct"``). When no
    defect code is recognised we keep the original prose intact as a
    single-element list — better to preserve the validator's words than
    drop the rejection rationale entirely.
    """

    candidates = [fragment.strip() for fragment in reason.replace(";", ",").split(",")]
    cleaned: list[str] = []
    for fragment in candidates:
        if not fragment:
            continue
        head = fragment.split(":", 1)[0].strip().upper()
        if head in _DEFECT_CODES:
            cleaned.append(fragment)
    if cleaned:
        return cleaned
    return [reason]


__all__ = ["Verdict", "parse_validation_response"]
