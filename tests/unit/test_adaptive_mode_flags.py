"""Unit tests for mode-specific adaptive feature-flag precedence.

Covers ``Settings.adaptive_enabled_for_mode`` — the single resolver the
interview runtime uses to decide whether the adaptive brain runs for a given
input mode. The master switch (``adaptive_interviewer_enabled``) gates
everything; each mode then has its own flag layered on top.

Backward-compat contract:
  * text + hybrid default ON  → an existing deployment that only set the master
    switch keeps adaptive text/hybrid.
  * voice defaults OFF        → flipping the master switch alone must NOT enable
    adaptive voice (human sign-off gates it via
    ADAPTIVE_INTERVIEWER_VOICE_ENABLED).
"""

from __future__ import annotations

import pytest

from abridgeai.core.config import Settings


@pytest.fixture(autouse=True)
def _clear_adaptive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make Settings() hermetic: pydantic reads env, so an ambient
    ADAPTIVE_INTERVIEWER_*_ENABLED (e.g. exported to run the live agent) would
    otherwise override the declared defaults these tests assert on."""
    for var in (
        "ADAPTIVE_INTERVIEWER_ENABLED",
        "ADAPTIVE_INTERVIEWER_TEXT_ENABLED",
        "ADAPTIVE_INTERVIEWER_HYBRID_ENABLED",
        "ADAPTIVE_INTERVIEWER_VOICE_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**overrides: bool) -> Settings:
    # Dummy DB/JWT so Settings validates without env; never connected.
    base = {
        "jwt_secret_key": "x" * 40,
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
        "test_database_url": "postgresql+psycopg://u:p@localhost:5432/db_test",
    }
    base.update(overrides)  # type: ignore[arg-type]
    # _env_file=None disables .env file loading so these tests assert against
    # the declared code defaults, not an ambient host .env (the deployed .env
    # exports ADAPTIVE_INTERVIEWER_VOICE_ENABLED=true to run the live agent,
    # which would otherwise override the default this suite verifies). The
    # autouse fixture clears the process env; this closes the file-load hole.
    return Settings(_env_file=None, **base)  # type: ignore[arg-type,call-arg]


def test_master_off_disables_every_mode() -> None:
    """Global off → legacy for all three modes, even if per-mode flags are on."""
    s = _settings(
        adaptive_interviewer_enabled=False,
        adaptive_interviewer_text_enabled=True,
        adaptive_interviewer_hybrid_enabled=True,
        adaptive_interviewer_voice_enabled=True,
    )
    assert s.adaptive_enabled_for_mode("text") is False
    assert s.adaptive_enabled_for_mode("hybrid") is False
    assert s.adaptive_enabled_for_mode("voice") is False


def test_defaults_text_and_hybrid_on_voice_off() -> None:
    """Master ON + defaults: text/hybrid adaptive, voice legacy."""
    s = _settings(adaptive_interviewer_enabled=True)
    assert s.adaptive_enabled_for_mode("text") is True
    assert s.adaptive_enabled_for_mode("hybrid") is True
    # Voice must stay off until explicitly enabled — flipping the master switch
    # alone does not turn on adaptive voice.
    assert s.adaptive_enabled_for_mode("voice") is False


def test_master_on_voice_explicitly_enabled() -> None:
    s = _settings(
        adaptive_interviewer_enabled=True,
        adaptive_interviewer_voice_enabled=True,
    )
    assert s.adaptive_enabled_for_mode("voice") is True


def test_master_on_text_explicitly_disabled() -> None:
    """A mode can be independently turned OFF under an ON master switch."""
    s = _settings(
        adaptive_interviewer_enabled=True,
        adaptive_interviewer_text_enabled=False,
    )
    assert s.adaptive_enabled_for_mode("text") is False
    assert s.adaptive_enabled_for_mode("hybrid") is True


@pytest.mark.parametrize("mode", ["text", "hybrid", "voice"])
def test_master_off_beats_mode_on(mode: str) -> None:
    s = _settings(
        adaptive_interviewer_enabled=False,
        adaptive_interviewer_text_enabled=True,
        adaptive_interviewer_hybrid_enabled=True,
        adaptive_interviewer_voice_enabled=True,
    )
    assert s.adaptive_enabled_for_mode(mode) is False


def test_unknown_mode_is_disabled() -> None:
    """An unrecognized mode is treated conservatively as legacy."""
    s = _settings(adaptive_interviewer_enabled=True)
    assert s.adaptive_enabled_for_mode("carrier-pigeon") is False
