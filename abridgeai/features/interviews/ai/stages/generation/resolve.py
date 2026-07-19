"""Config-resolution helpers for the interview GENERATION stage (T6.5).

Split out of ``logic.py`` to keep that module under the feature's LOC
budget. Pure functions only — no LLM calls, no DB access — so they stay
trivially unit-testable in isolation from the gateway.
"""

from __future__ import annotations

import json
from typing import Any

_DEFAULT_TYPE_MIX: dict[str, int] = {"technical": 60, "behavioral": 30, "situational": 10}
_DEFAULT_QUESTION_COUNT = 8
_MIN_QUESTION_COUNT = 1
_MAX_QUESTION_COUNT = 50


def resolve_type_mix(supplementary: str | None) -> dict[str, int]:
    """Return weights summing to 100 — fall back to the 60/30/10 default."""
    parsed = _try_parse_rubric(supplementary)
    if parsed is None:
        return dict(_DEFAULT_TYPE_MIX)
    raw_weights = parsed.get("rubric_weights")
    if not isinstance(raw_weights, dict):
        return dict(_DEFAULT_TYPE_MIX)
    cleaned: dict[str, int] = {key: 0 for key in _DEFAULT_TYPE_MIX}
    for key, value in raw_weights.items():
        if not isinstance(key, str):
            continue
        normalised_key = key.strip().lower()
        if normalised_key == "behavioural":  # accept BrEng spelling
            normalised_key = "behavioral"
        if normalised_key not in cleaned:
            continue
        try:
            cleaned[normalised_key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    total = sum(cleaned.values())
    if total <= 0:
        return dict(_DEFAULT_TYPE_MIX)
    return {key: round(value * 100 / total) for key, value in cleaned.items()}


def resolve_question_count(
    *,
    run_config_json: dict[str, Any] | None,
    supplementary: str | None,
) -> int:
    """Resolve question count, clamped to [1, 50].

    Precedence: form value (``run_config_json["question_count"]``) →
    ``supplementary_instructions`` JSON override → default.

    Public (no leading underscore) so the T6.10 pipeline can compute the
    same target count up front for its backfill loop without duplicating
    the resolution precedence here.
    """
    from_form = _coerce_question_count(
        run_config_json.get("question_count") if isinstance(run_config_json, dict) else None
    )
    if from_form is not None:
        return from_form

    parsed = _try_parse_rubric(supplementary)
    if parsed is not None:
        from_supplementary = _coerce_question_count(parsed.get("question_count"))
        if from_supplementary is not None:
            return from_supplementary

    return _DEFAULT_QUESTION_COUNT


def _coerce_question_count(raw: object) -> int | None:
    """Parse + clamp a raw count to [1, 50]; None if unusable."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        count = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(_MIN_QUESTION_COUNT, min(_MAX_QUESTION_COUNT, count))


def _try_parse_rubric(supplementary: str | None) -> dict[str, Any] | None:
    """Best-effort JSON parse of the supplementary-instructions field."""
    if not supplementary:
        return None
    stripped = supplementary.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = ["resolve_question_count", "resolve_type_mix"]
