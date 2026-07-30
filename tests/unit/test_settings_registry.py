"""Registry validation and resolution precedence.

The precedence chain is the whole point of this feature, and it is where a
mistake is quietest: getting it wrong does not raise, it just silently ignores
what an operator set. Each level is pinned here.
"""

from __future__ import annotations

import pytest

from abridgeai.core.settings_registry import (
    SETTINGS_REGISTRY,
    SettingValidationError,
    coerce_and_validate,
)


class TestRegistryShape:
    def test_every_spec_key_matches_its_map_key(self) -> None:
        for key, spec in SETTINGS_REGISTRY.items():
            assert spec.key == key

    def test_defaults_satisfy_their_own_bounds(self) -> None:
        """A default outside its bounds would make a fresh install unwritable."""
        for key, spec in SETTINGS_REGISTRY.items():
            assert coerce_and_validate(key, spec.default) == spec.default

    def test_env_var_names_are_unique(self) -> None:
        seen: dict[str, str] = {}
        for key, spec in SETTINGS_REGISTRY.items():
            if spec.env_var is None:
                continue
            assert spec.env_var not in seen, (
                f"{spec.env_var} claimed by both {seen[spec.env_var]} and {key}"
            )
            seen[spec.env_var] = key


class TestValidation:
    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.nope", 1)

    def test_bounds_are_enforced_both_ways(self) -> None:
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.parallelism", 0)
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.parallelism", 999)

    def test_int_setting_rejects_a_fraction(self) -> None:
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.max_tokens", 800.5)

    def test_int_setting_accepts_a_whole_float(self) -> None:
        assert coerce_and_validate("chunking.max_tokens", 800.0) == 800

    def test_bool_setting_rejects_an_int(self) -> None:
        """1 is not True here.

        Accepting it would let a checkbox round-trip as a number and read back
        wrong in the UI.
        """
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.llm_boundary_enabled", 1)

    def test_numeric_setting_rejects_a_bool(self) -> None:
        """The subtle one: ``bool`` subclasses ``int`` in Python.

        Without an explicit guard ``isinstance(True, int)`` passes and ``True``
        is stored as a token count of 1.
        """
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.max_tokens", True)

    def test_string_is_rejected(self) -> None:
        with pytest.raises(SettingValidationError):
            coerce_and_validate("chunking.max_tokens", "800")
