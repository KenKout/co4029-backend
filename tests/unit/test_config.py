from __future__ import annotations

import pytest
from pydantic import ValidationError

from abridgeai.core.config import Settings

_DEV_SECRET_32 = "dev-only-change-me-padding-to-32-chars-xx"  # noqa: S105
_STRONG_PROD_SECRET = "x" * 64  # noqa: S105


def test_jwt_secret_min_length_32() -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(jwt_secret_key="short", environment="dev")  # noqa: S106


def test_jwt_secret_dev_default_allowed_in_dev() -> None:
    s = Settings(jwt_secret_key=_DEV_SECRET_32, environment="dev")
    assert s.jwt_secret_key.startswith("dev-only")


def test_jwt_secret_dev_default_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="dev-|production"):
        Settings(jwt_secret_key=_DEV_SECRET_32, environment="production")


def test_jwt_secret_strong_value_accepted_in_production() -> None:
    s = Settings(jwt_secret_key=_STRONG_PROD_SECRET, environment="production")
    assert len(s.jwt_secret_key) == 64


def test_voice_load_threshold_defaults_above_the_sdk_value() -> None:
    """Default must clear the SDK's 0.7.

    The worker's load signal is whole-host CPU, so on a shared box 0.7 is
    tripped by neighbouring processes and LiveKit stops dispatching interviews
    to an otherwise idle worker.
    """
    s = Settings(jwt_secret_key=_DEV_SECRET_32, environment="dev")
    assert s.interview_voice_load_threshold == 0.9
    assert s.interview_voice_load_threshold > 0.7


@pytest.mark.parametrize("value", [1.0, 1.5, 0.0, -0.1])
def test_voice_load_threshold_rejects_out_of_range(value: float) -> None:
    """livekit-agents raises on >= 1 outside dev mode, so reject it at config time."""
    with pytest.raises(ValidationError):
        Settings(
            jwt_secret_key=_DEV_SECRET_32,
            environment="dev",
            interview_voice_load_threshold=value,
        )
