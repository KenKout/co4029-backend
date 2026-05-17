"""Parse the interview-followup LLM JSON response into a typed verdict.

The follow-up stage runs at session runtime (T6.7), so the parser is
intentionally permissive: a malformed payload from the (small-tier) model
should never crash ``take_session_step``. Instead we return a verdict that
says ``is_sufficient=True`` with a rationale explaining the fallback, which
quietly suppresses a follow-up rather than blocking the student.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_DEFAULT_FALLBACK_RATIONALE = (
    "Follow-up stage could not parse the LLM response; treating answer as sufficient."
)


@dataclass(frozen=True)
class FollowupVerdict:
    """Outcome of the follow-up sufficiency judgement.

    ``is_sufficient`` mirrors the LLM's boolean. ``followup`` is the
    probing question text when ``is_sufficient=False`` — None otherwise.
    ``rationale`` is a short, free-form audit string used for telemetry
    only (never shown to the student).
    """

    is_sufficient: bool
    followup: str | None
    rationale: str


def parse_followup_response(payload: Mapping[str, Any] | None) -> FollowupVerdict:
    """Coerce the gateway JSON dict into a :class:`FollowupVerdict`.

    The model contract is::

        {"is_sufficient": bool, "followup": str | null, "rationale": str}

    We accept common deviations: missing keys default to a sufficient
    verdict, ``is_sufficient`` may arrive as a string ("true"/"false"),
    and a non-empty ``followup`` flips ``is_sufficient`` to False so a
    contradictory model still yields a usable follow-up.
    """

    if not isinstance(payload, Mapping):
        return FollowupVerdict(
            is_sufficient=True,
            followup=None,
            rationale=_DEFAULT_FALLBACK_RATIONALE,
        )

    is_sufficient = _coerce_bool(payload.get("is_sufficient"))
    followup = _coerce_followup_text(payload.get("followup"))
    rationale = _coerce_rationale(payload.get("rationale"))

    if followup is None:
        return FollowupVerdict(is_sufficient=True, followup=None, rationale=rationale)

    if is_sufficient is True:
        return FollowupVerdict(is_sufficient=True, followup=None, rationale=rationale)

    return FollowupVerdict(is_sufficient=False, followup=followup, rationale=rationale)


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "yes", "1"}:
            return True
        if cleaned in {"false", "no", "0"}:
            return False
    return None


def _coerce_followup_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:1000]


def _coerce_rationale(value: object) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned[:500]
    return _DEFAULT_FALLBACK_RATIONALE


__all__ = ["FollowupVerdict", "parse_followup_response"]
