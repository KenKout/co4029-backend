"""Config-resolution helpers for the interview GENERATION stage (T6.5).

Split out of ``logic.py`` to keep that module under the feature's LOC
budget. Pure functions only — no LLM calls, no DB access — so they stay
trivially unit-testable in isolation from the gateway.
"""

from __future__ import annotations

import json
from typing import Any, cast

_DEFAULT_TYPE_MIX: dict[str, int] = {"technical": 60, "behavioral": 30, "situational": 10}
_DEFAULT_QUESTION_COUNT = 5
_MIN_QUESTION_COUNT = 1
_MAX_QUESTION_COUNT = 50

#: The question angles a role-conditioned bank generates — one per non-generic
#: interviewer role (see ``orchestrator.role_question_filter``).
VARIANT_ANGLES: tuple[str, ...] = (
    "technical",
    "system_design",
    "situational",
    "behavioral",
)

#: Ceiling on the LOGICAL question count when ``all_angles`` is in play.
#:
#: ``all_angles`` turns each logical question into one row per angle, so the
#: teacher's count is multiplied by ``len(VARIANT_ANGLES)`` to get the real row
#: budget. That multiplication used to happen AFTER the [1, 50] clamp, so a
#: typed 50 asked the pipeline for 200 rows — and the pipeline demands an EXACT
#: hit (``accepted != target_count`` raises) with only ``MAX_BACKFILL_ATTEMPTS``
#: rounds of recovery, where ``all_angles`` additionally only accepts whole
#: 4-angle groups sharing one outcome and difficulty. Such a run burns four
#: rounds of LLM spend and then fails. Capping the EFFECTIVE total at
#: ``_MAX_QUESTION_COUNT`` keeps the teacher's number meaning "logical
#: questions" while making the row budget honest.
MAX_ALL_ANGLES_LOGICAL_COUNT = _MAX_QUESTION_COUNT // len(VARIANT_ANGLES)


def max_logical_question_count(variant_strategy: str | None) -> int:
    """Upper bound on the count a teacher may request for this strategy.

    ``all_angles`` fans each logical question out into ``len(VARIANT_ANGLES)``
    rows, so its ceiling is proportionally lower; every other strategy maps
    1:1 and keeps the full ``_MAX_QUESTION_COUNT``.
    """
    if variant_strategy == "all_angles":
        return MAX_ALL_ANGLES_LOGICAL_COUNT
    return _MAX_QUESTION_COUNT


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
    if not isinstance(raw, (str, bytes, bytearray, int, float)):
        return None
    try:
        count = int(raw)
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
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


def resolve_avoid_topics(run_config_json: dict[str, Any] | None) -> list[str]:
    """Teacher-supplied exclusion list from the run's ``config_json``.

    Cleaned here rather than trusted raw: the value round-trips through JSONB,
    so a non-list or a list holding blanks/non-strings is possible. Blanks
    would render as empty ``-`` bullets and dilute the instruction.
    """
    if not isinstance(run_config_json, dict):
        return []
    raw = run_config_json.get("avoid_topics")
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def resolve_supplementary(
    run_config_json: dict[str, Any] | None,
    config_supplementary: str | None,
) -> str | None:
    """Pick the ``supplementary_instructions`` a generation run must use.

    Precedence: the run's own value (the teacher's Generate-tab form) over the
    saved config column. Empty string counts as "not supplied" — the form sends
    ``null`` when the field is blank, and a blank override must not wipe the
    config's rubric/prose.

    Why this exists: the request carried ``supplementary_instructions`` since
    the feature shipped, but every stage read ``config.supplementary_instructions``
    instead, so the value was accepted and ignored. This is the ONE resolver all
    call sites share, because the field feeds three different consumers —
    :func:`resolve_type_mix` (question type mix), :func:`resolve_question_count`,
    and the prose injected into the prompt. If generation honoured the override
    while the run's ``type_weights`` (computed by the authoring service and read
    back by the VALIDATION stage) still came from the config, validation would
    reject the very drafts generation was told to produce.

    Note the SCORING rubric (``evaluation_rubric``) is deliberately NOT affected
    at grading time: ``services/evaluation.py`` reads the config column, so a
    per-run generation override can never change how a sitting is graded.
    """
    if isinstance(run_config_json, dict):
        raw = run_config_json.get("supplementary_instructions")
        if isinstance(raw, str) and raw.strip():
            return raw
    return config_supplementary


def resolve_persona(
    run_config_json: dict[str, Any] | None,
    config_persona: str | None,
) -> str:
    """Pick the persona label for the generation prompt, defaulting to neutral.

    Same precedence and same rationale as :func:`resolve_supplementary` — the
    request field was accepted and dropped. Only the three authored labels are
    honoured; anything else (a stale client, a hand-rolled API call) falls back
    to the config rather than reaching the prompt, since ``persona`` is rendered
    verbatim and a free-text value here would be an injection surface.

    Generation-time only: the persona that CONDUCTS the interview still comes
    from the config (``services/taking.py`` / ``orchestrator.persona``), so this
    cannot change the tone a candidate actually meets.
    """
    allowed = ("strict", "neutral", "supportive")
    if isinstance(run_config_json, dict):
        raw = run_config_json.get("persona")
        if isinstance(raw, str) and raw.strip().lower() in allowed:
            return raw.strip().lower()
    return config_persona or "neutral"


def resolve_variant_strategy(run_config_json: dict[str, Any] | None) -> str | None:
    """Resolve the variant generation strategy from the run's form values.

    ``all_angles`` = one question per angle per logical question (4x the
    typed count); ``role_only`` = every question of the config role's
    preferred type (1x). Returns ``None`` (legacy mixed generation) when
    the form value is absent or unrecognised.
    """
    if not isinstance(run_config_json, dict):
        return None
    raw = run_config_json.get("variant_strategy")
    if isinstance(raw, str):
        normalised = raw.strip().lower()
        if normalised in ("all_angles", "role_only"):
            return normalised
    return None


__all__ = [
    "MAX_ALL_ANGLES_LOGICAL_COUNT",
    "VARIANT_ANGLES",
    "max_logical_question_count",
    "resolve_avoid_topics",
    "resolve_persona",
    "resolve_question_count",
    "resolve_supplementary",
    "resolve_type_mix",
    "resolve_variant_strategy",
]
