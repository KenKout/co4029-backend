"""Unit tests for the Phase 10/11 adaptive rollout + shadow-mode resolvers.

Covers two Settings methods layered on top of ``adaptive_enabled_for_mode``:

  * ``adaptive_enabled_for_student`` (Phase 11) — the static mode gate AND a
    deterministic percentage rollout keyed on a stable (student, config) hash.
  * ``shadow_enabled_for_mode`` (Phase 10) — whether to COMPUTE the adaptive
    decision for comparison without letting it drive the student; only
    meaningful for a mode that is NOT already live.

Both default to a no-op posture: rollout_percent=100 preserves "everyone who
passes the static gate is enabled"; shadow defaults OFF.
"""

from __future__ import annotations

from typing import Any

import pytest

from abridgeai.core.config import Settings


@pytest.fixture(autouse=True)
def _clear_adaptive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic Settings(): strip ambient ADAPTIVE_* env (e.g. exported to run
    the live agent) so declared defaults hold."""
    for var in (
        "ADAPTIVE_INTERVIEWER_ENABLED",
        "ADAPTIVE_INTERVIEWER_TEXT_ENABLED",
        "ADAPTIVE_INTERVIEWER_HYBRID_ENABLED",
        "ADAPTIVE_INTERVIEWER_VOICE_ENABLED",
        "ADAPTIVE_INTERVIEWER_SHADOW_ENABLED",
        "ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jwt_secret_key": "x" * 40,
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
        "test_database_url": "postgresql+psycopg://u:p@localhost:5432/db_test",
    }
    base.update(overrides)
    return Settings(**base)


# ── Percentage rollout (Phase 11) ────────────────────────────────────────────


def test_rollout_default_100_enables_everyone_past_static_gate() -> None:
    """Default rollout_percent=100 → identical to the plain mode gate."""
    s = _settings(adaptive_interviewer_enabled=True)  # text/hybrid default on
    assert s.adaptive_interviewer_rollout_percent == 100
    for i in range(20):
        assert (
            s.adaptive_enabled_for_student("text", student_id=f"u{i}", config_id="c1")
            is True
        )


def test_rollout_zero_gates_everyone_out_without_touching_mode_flag() -> None:
    s = _settings(adaptive_interviewer_enabled=True, adaptive_interviewer_rollout_percent=0)
    # Mode is still statically enabled...
    assert s.adaptive_enabled_for_mode("text") is True
    # ...but nobody passes the per-student gate.
    for i in range(20):
        assert (
            s.adaptive_enabled_for_student("text", student_id=f"u{i}", config_id="c1")
            is False
        )


def test_rollout_static_gate_off_means_off_regardless_of_percent() -> None:
    """If the mode isn't statically enabled, 100% rollout still yields False."""
    s = _settings(adaptive_interviewer_enabled=False, adaptive_interviewer_rollout_percent=100)
    assert (
        s.adaptive_enabled_for_student("text", student_id="u1", config_id="c1") is False
    )


def test_rollout_is_deterministic_for_same_pair() -> None:
    """A given (student, config) pair always resolves the same way — a student's
    experience must not flip between turns/attempts."""
    s = _settings(adaptive_interviewer_enabled=True, adaptive_interviewer_rollout_percent=50)
    first = s.adaptive_enabled_for_student("text", student_id="stable-user", config_id="c1")
    for _ in range(50):
        assert (
            s.adaptive_enabled_for_student("text", student_id="stable-user", config_id="c1")
            == first
        )


def test_rollout_bucket_is_stable_and_bounded() -> None:
    """The bucket is in [0,99], salt-independent, and stable across calls."""
    b1 = Settings._rollout_bucket("user-a", "config-x")
    b2 = Settings._rollout_bucket("user-a", "config-x")
    assert b1 == b2
    assert 0 <= b1 < 100


def test_rollout_partial_splits_population() -> None:
    """A 50% rollout enables roughly half of a large population and, crucially,
    strictly fewer than 100% and more than 0%."""
    s = _settings(adaptive_interviewer_enabled=True, adaptive_interviewer_rollout_percent=50)
    enabled = sum(
        1
        for i in range(500)
        if s.adaptive_enabled_for_student("text", student_id=f"user-{i}", config_id="c1")
    )
    # Deterministic hash over 500 users should land near half; keep the bound
    # loose enough to never flake but tight enough to prove it's splitting.
    assert 150 < enabled < 350


def test_rollout_percent_clamped_by_field_bounds() -> None:
    """The field is constrained to [0,100]; out-of-range raises at construction."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _settings(adaptive_interviewer_rollout_percent=150)
    with pytest.raises(pydantic.ValidationError):
        _settings(adaptive_interviewer_rollout_percent=-1)


# ── Shadow mode (Phase 10) ────────────────────────────────────────────────────


def test_shadow_default_off() -> None:
    s = _settings(adaptive_interviewer_enabled=True)
    assert s.adaptive_interviewer_shadow_enabled is False
    assert s.shadow_enabled_for_mode("voice") is False


def test_shadow_on_for_a_mode_that_is_not_live() -> None:
    """Shadow voice while voice is NOT statically enabled → shadow runs."""
    s = _settings(
        adaptive_interviewer_enabled=True,  # text/hybrid live
        adaptive_interviewer_voice_enabled=False,  # voice NOT live
        adaptive_interviewer_shadow_enabled=True,
    )
    assert s.shadow_enabled_for_mode("voice") is True


def test_shadow_is_noop_for_a_mode_already_live() -> None:
    """You can't shadow a path that's already driving the student."""
    s = _settings(
        adaptive_interviewer_enabled=True,
        adaptive_interviewer_voice_enabled=True,  # voice IS live
        adaptive_interviewer_shadow_enabled=True,
    )
    assert s.adaptive_enabled_for_mode("voice") is True
    assert s.shadow_enabled_for_mode("voice") is False
    # text is also live under the master switch → no shadow either.
    assert s.shadow_enabled_for_mode("text") is False


def test_shadow_independent_of_master_switch() -> None:
    """Shadow can run even with the master switch off (nothing is live, so every
    recognised mode is eligible to be shadowed)."""
    s = _settings(
        adaptive_interviewer_enabled=False,
        adaptive_interviewer_shadow_enabled=True,
    )
    assert s.shadow_enabled_for_mode("voice") is True
    assert s.shadow_enabled_for_mode("text") is True
    assert s.shadow_enabled_for_mode("hybrid") is True


def test_shadow_unknown_mode_is_false() -> None:
    s = _settings(
        adaptive_interviewer_enabled=False,
        adaptive_interviewer_shadow_enabled=True,
    )
    assert s.shadow_enabled_for_mode("carrier-pigeon") is False
