"""Pydantic validator tests for :class:`LessonUnlockConfig` (T3.1).

Per plan §3927 the application boundary must reject out-of-range
``ef_min_unlock`` / ``tau_unlock`` values BEFORE issuing INSERT /
UPDATE. These tests exercise the Pydantic v2 ``field_validator``
gates without touching the database.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from abridgeai.features.courses.schemas import LessonUnlockConfig


def test_defaults_pass() -> None:
    config = LessonUnlockConfig()
    assert config.ef_min_unlock == 2.0
    assert config.tau_unlock == 0.8
    assert config.requires_interview_pass is False


@pytest.mark.parametrize("ef_min_unlock", [1.3, 1.5, 2.0, 2.4, 2.5])
def test_ef_min_unlock_in_range_passes(ef_min_unlock: float) -> None:
    config = LessonUnlockConfig(ef_min_unlock=ef_min_unlock, tau_unlock=0.8)
    assert config.ef_min_unlock == ef_min_unlock


@pytest.mark.parametrize("ef_min_unlock", [-1.0, 0.0, 1.0, 1.299, 2.501, 3.0, 100.0])
def test_ef_min_unlock_out_of_range_rejected(ef_min_unlock: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        LessonUnlockConfig(ef_min_unlock=ef_min_unlock, tau_unlock=0.8)
    assert "ef_min_unlock" in str(exc_info.value)


@pytest.mark.parametrize("tau_unlock", [0.01, 0.1, 0.5, 0.8, 0.999, 1.0])
def test_tau_unlock_in_range_passes(tau_unlock: float) -> None:
    config = LessonUnlockConfig(ef_min_unlock=2.0, tau_unlock=tau_unlock)
    assert config.tau_unlock == tau_unlock


@pytest.mark.parametrize("tau_unlock", [-1.0, 0.0, 1.0001, 1.5, 2.0, 100.0])
def test_tau_unlock_out_of_range_rejected(tau_unlock: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        LessonUnlockConfig(ef_min_unlock=2.0, tau_unlock=tau_unlock)
    assert "tau_unlock" in str(exc_info.value)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        LessonUnlockConfig(ef_min_unlock=2.0, tau_unlock=0.8, bogus_field=42)


def test_requires_interview_pass_accepts_bool() -> None:
    enabled = LessonUnlockConfig(requires_interview_pass=True)
    assert enabled.requires_interview_pass is True
    disabled = LessonUnlockConfig(requires_interview_pass=False)
    assert disabled.requires_interview_pass is False
