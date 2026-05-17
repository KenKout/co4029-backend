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
