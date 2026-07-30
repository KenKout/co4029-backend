"""Unit tests for ``profile_from_config`` — teacher per-trait persona overrides.

Phase 3 lets a teacher layer optional per-trait overrides on top of a persona
preset via ``interview_configs.persona_profile_json``. ``profile_from_config``
resolves the effective :class:`PersonaProfile` with this order:

1. persona_profile_json present → preset merged with the overrides (clamped;
   unknown keys ignored; non-numeric values skipped).
2. persona only → the preset.
3. neither → neutral.

These tests lock the resolution order, the clamp, and — critically — that a
malformed override can never raise and never yields an out-of-range trait, so a
bad override degrades to "slightly different tone", never a crash or a
scoring/fairness change (persona is tone-only).
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.persona import (
    PRESETS,
    TRAIT_MAX,
    TRAIT_MIN,
    OpeningStyle,
    profile_from_config,
)


# ── Resolution order ─────────────────────────────────────────────────────────


def test_neither_persona_nor_overrides_yields_neutral() -> None:
    profile = profile_from_config(None, None)
    assert profile.key == "neutral"
    assert profile == PRESETS["neutral"]


def test_persona_only_yields_the_preset() -> None:
    for key in ("strict", "neutral", "supportive"):
        assert profile_from_config(key, None) == PRESETS[key]


def test_empty_override_dict_is_treated_as_no_override() -> None:
    assert profile_from_config("strict", {}) == PRESETS["strict"]


# ── Merge: overrides layer on the preset ─────────────────────────────────────


def test_single_trait_override_merges_over_preset() -> None:
    # strict preset has warmth=0; bump ONLY warmth and leave the rest intact.
    profile = profile_from_config("strict", {"warmth": 3})
    assert profile.warmth == 3
    # Every other trait still equals the strict preset.
    assert profile.directness == PRESETS["strict"].directness
    assert profile.verbosity == PRESETS["strict"].verbosity
    assert profile.formality == PRESETS["strict"].formality
    assert profile.ack_frequency == PRESETS["strict"].ack_frequency
    # The legacy key is preserved — the preset still keys the fallback tables.
    assert profile.key == "strict"


def test_multiple_traits_override() -> None:
    profile = profile_from_config(
        "neutral", {"warmth": 4, "directness": 0, "verbosity": 1}
    )
    assert profile.warmth == 4
    assert profile.directness == 0
    assert profile.verbosity == 1
    assert profile.formality == PRESETS["neutral"].formality


def test_key_is_never_overridable() -> None:
    # Even if a caller sneaks a 'key' into the override dict, the preset key
    # stands — the override can only reshape tone dials, never the selector.
    profile = profile_from_config("strict", {"key": "supportive", "warmth": 2})
    assert profile.key == "strict"


# ── Clamp: overrides are forced into [TRAIT_MIN, TRAIT_MAX] ───────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(99, TRAIT_MAX), (-5, TRAIT_MIN), (TRAIT_MAX + 1, TRAIT_MAX), (0, 0), (4, 4)],
)
def test_override_values_are_clamped(raw: int, expected: int) -> None:
    profile = profile_from_config("neutral", {"warmth": raw})
    assert profile.warmth == expected
    assert TRAIT_MIN <= profile.warmth <= TRAIT_MAX


# ── Robustness: malformed overrides never raise, never go out of range ────────


def test_unknown_keys_are_ignored() -> None:
    profile = profile_from_config("neutral", {"nonsense": 3, "warmth": 1})
    assert profile.warmth == 1
    # The bogus key did not become an attribute or crash construction.
    assert not hasattr(profile, "nonsense")


def test_non_numeric_override_is_skipped_keeping_preset_value() -> None:
    profile = profile_from_config("supportive", {"warmth": "hot", "verbosity": None})
    # Both bad values skipped → supportive preset values stand.
    assert profile.warmth == PRESETS["supportive"].warmth
    assert profile.verbosity == PRESETS["supportive"].verbosity


def test_string_numeric_override_is_coerced() -> None:
    # int("3") works, so a stringified number is accepted and clamped.
    profile = profile_from_config("strict", {"warmth": "3"})
    assert profile.warmth == 3


def test_float_override_is_coerced_via_int() -> None:
    profile = profile_from_config("strict", {"warmth": 2.9})
    assert profile.warmth == 2  # int(2.9) == 2


def test_non_dict_override_is_ignored() -> None:
    # A list / string / int in the JSON column must not crash resolution.
    for bad in ([1, 2, 3], "warmth=4", 7):
        assert profile_from_config("strict", bad) == PRESETS["strict"]  # type: ignore[arg-type]


# ── opening_style override ───────────────────────────────────────────────────


def test_valid_opening_style_override() -> None:
    profile = profile_from_config("strict", {"opening_style": "comfort"})
    assert profile.opening_style is OpeningStyle.COMFORT


def test_invalid_opening_style_is_ignored() -> None:
    profile = profile_from_config("strict", {"opening_style": "bogus"})
    assert profile.opening_style is PRESETS["strict"].opening_style


# ── The resolved profile is always in range (property-ish sweep) ─────────────


def test_resolved_profile_is_always_clampable_and_serializable() -> None:
    profile = profile_from_config(
        "supportive", {"warmth": 99, "directness": -3, "opening_style": "brief"}
    )
    traits = profile.clamped().as_prompt_traits()
    for field in ("warmth", "directness", "verbosity", "formality", "ack_frequency"):
        assert TRAIT_MIN <= int(traits[field]) <= TRAIT_MAX  # type: ignore[call-overload]
    assert traits["opening_style"] == "brief"
    assert traits["key"] == "supportive"
