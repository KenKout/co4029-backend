"""Parser + result shape for the PERSONA ADHERENCE stage.

Permissive by design (mirrors the followup / gap-report parser philosophy):
this is a post-session diagnostic, never a scoring gate, so a malformed field
degrades to a safe default rather than raising. A judge that returns garbage
must not crash the evaluation pipeline it runs alongside.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# The exact violation tags the prompt is allowed to emit. Anything else the
# model invents is dropped — the set is a closed vocabulary so downstream
# consumers (teacher UI, metrics) can rely on it.
_ALLOWED_VIOLATIONS: frozenset[str] = frozenset(
    {
        "over_polite_for_strict",
        "cold_for_supportive",
        "declared_answer",
        "repeated_candidate_answer",
        "identity_confusion",
    }
)

_TRAIT_MIN = 0
_TRAIT_MAX = 4
_SCORE_MIN = 0
_SCORE_MAX = 10
_MAX_REASONING_LEN = 2000
_MAX_DRIFT_TURNS = 100


@dataclass(frozen=True)
class PersonaAdherence:
    """One tone-only audit of an interview transcript against its persona.

    ``tone_consistency`` is the headline 0–10 score; ``*_observed`` are the
    judge's read of the traits actually displayed (0–4), for intent-vs-reality
    comparison. ``drift_turns`` and ``violations`` localise the problems.
    ``available`` is False when the judge produced nothing usable, so callers
    can distinguish "audited, clean" from "no audit".
    """

    tone_consistency: int
    reasoning: str
    warmth_observed: int
    directness_observed: int
    verbosity_observed: int
    formality_observed: int
    drift_turns: list[int] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    available: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "tone_consistency": self.tone_consistency,
            "reasoning": self.reasoning,
            "warmth_observed": self.warmth_observed,
            "directness_observed": self.directness_observed,
            "verbosity_observed": self.verbosity_observed,
            "formality_observed": self.formality_observed,
            "drift_turns": list(self.drift_turns),
            "violations": list(self.violations),
            "available": self.available,
        }


def unavailable() -> PersonaAdherence:
    """The sentinel returned when no usable judgement exists (no turns, LLM down)."""
    return PersonaAdherence(
        tone_consistency=0,
        reasoning="",
        warmth_observed=0,
        directness_observed=0,
        verbosity_observed=0,
        formality_observed=0,
        drift_turns=[],
        violations=[],
        available=False,
    )


def parse_persona_adherence(payload: Mapping[str, Any] | None) -> PersonaAdherence:
    """Coerce the gateway JSON into a :class:`PersonaAdherence`.

    Any missing / malformed field degrades to a safe default; only a payload
    that is not a mapping at all yields the ``unavailable()`` sentinel.
    """
    if not isinstance(payload, Mapping):
        return unavailable()

    tone = payload.get("tone_consistency")
    score = 0
    reasoning = ""
    if isinstance(tone, Mapping):
        score = _clamp_int(tone.get("score"), _SCORE_MIN, _SCORE_MAX)
        reasoning = _clean_str(tone.get("reasoning"), _MAX_REASONING_LEN)
    else:
        # Tolerate a flattened shape: {"tone_consistency": 7, "reasoning": "..."}
        score = _clamp_int(tone, _SCORE_MIN, _SCORE_MAX)
        reasoning = _clean_str(payload.get("reasoning"), _MAX_REASONING_LEN)

    return PersonaAdherence(
        tone_consistency=score,
        reasoning=reasoning,
        warmth_observed=_clamp_int(payload.get("warmth_observed"), _TRAIT_MIN, _TRAIT_MAX),
        directness_observed=_clamp_int(payload.get("directness_observed"), _TRAIT_MIN, _TRAIT_MAX),
        verbosity_observed=_clamp_int(payload.get("verbosity_observed"), _TRAIT_MIN, _TRAIT_MAX),
        formality_observed=_clamp_int(payload.get("formality_observed"), _TRAIT_MIN, _TRAIT_MAX),
        drift_turns=_coerce_drift_turns(payload.get("drift_turns")),
        violations=_coerce_violations(payload.get("violations")),
        available=True,
    )


def _clamp_int(raw: object, low: int, high: int) -> int:
    if isinstance(raw, bool):
        return low
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def _clean_str(raw: object, limit: int) -> str:
    return raw.strip()[:limit] if isinstance(raw, str) else ""


def _coerce_drift_turns(raw: object) -> list[int]:
    if not isinstance(raw, (list, tuple)):
        return []
    turns: list[int] = []
    seen: set[int] = set()
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            turn = int(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if turn < 0 or turn in seen:
            continue
        seen.add(turn)
        turns.append(turn)
        if len(turns) >= _MAX_DRIFT_TURNS:
            break
    return turns


def _coerce_violations(raw: object) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = item.strip()
        if tag in _ALLOWED_VIOLATIONS and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


__all__ = [
    "PersonaAdherence",
    "parse_persona_adherence",
    "unavailable",
]
