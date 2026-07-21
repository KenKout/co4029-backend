"""Behavior tests for strict-mode wiring inside ``run_scenario``.

These stub out all live/IO parts of the harness (credential check, DB
provisioning, the live LiveKit drive, DB-signal collection, and both cleanups)
so we can assert how the STRICT flag flips the final verdict — without touching
LiveKit, the gateway, or a database.

Covers the four required cases:
  * strict + adaptive ran            → pass
  * strict + legacy (no adaptive)    → fail, with a diagnostic error
  * non-strict + legacy              → pass (adaptive informational)
  * the machine-readable result accurately reports strict verification
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.voice_harness import run_harness


@pytest.fixture
def stub_live(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub every live/IO touchpoint of run_scenario. Returns a mutable dict so
    each test can dial the DB signals that come back."""
    signals: dict[str, Any] = {
        "transcript": [{"role": "user", "action": "", "text": "hi"}],
        "message_count": 3,
        "runtime_state_count": 1,
        "adaptive_actions": ["ask_for_example"],
        "state_version": 2,
        "decision_count": 1,
        "fallback_count": 0,
    }

    # Credentials always "present".
    monkeypatch.setattr(run_harness, "_require_credentials", lambda settings: None)

    import uuid

    cfg_id = uuid.uuid4()
    stu_id = uuid.uuid4()
    sess_id = uuid.uuid4()

    async def _fake_provision(db: Any) -> tuple[uuid.UUID, uuid.UUID]:
        return cfg_id, stu_id

    async def _fake_create(db: Any, c: uuid.UUID, s: uuid.UUID) -> uuid.UUID:
        return sess_id

    monkeypatch.setattr(run_harness, "_provision_throwaway", _fake_provision)
    monkeypatch.setattr(run_harness, "_create_voice_session", _fake_create)

    async def _fake_drive(*, result: run_harness.ScenarioResult, **kw: Any) -> None:
        # Pretend the agent spoke: populate the audio side of the result.
        result.agent_utterance_count = 1
        result.captured_audio_seconds = 5.0
        result.peak_amplitude = 12000

    monkeypatch.setattr(run_harness, "_drive_scenario", _fake_drive)

    async def _fake_signals(session_id: uuid.UUID) -> dict[str, Any]:
        return dict(signals)

    monkeypatch.setattr(run_harness, "_collect_db_signals", _fake_signals)

    async def _fake_cleanup(db: Any, config_id: uuid.UUID) -> None:
        return None

    monkeypatch.setattr(run_harness, "_cleanup_throwaway", _fake_cleanup)

    async def _fake_lk_delete(room_name: str, settings: Any) -> str:
        return "deleted"

    monkeypatch.setattr(run_harness, "_delete_livekit_room", _fake_lk_delete)

    # A minimal sessionmaker stub whose async-context yields a dummy db.
    class _DummyDB:
        async def __aenter__(self) -> _DummyDB:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(run_harness, "get_sessionmaker", lambda: lambda: _DummyDB())

    return signals


@pytest.mark.asyncio
async def test_strict_pass_when_adaptive_ran(stub_live: dict[str, Any], tmp_path: Path) -> None:
    result = await run_harness.run_scenario(
        out_dir=str(tmp_path / "o"),
        require_adaptive=True,
        cleanup_livekit=True,
    )
    assert result.adaptive_required is True
    assert result.adaptive_requirement_met is True
    assert result.ok is True
    assert result.error is None


@pytest.mark.asyncio
async def test_strict_fail_when_legacy(stub_live: dict[str, Any], tmp_path: Path) -> None:
    # Dial the signals to a legacy run: no runtime state, no decisions.
    stub_live.update(
        runtime_state_count=0,
        adaptive_actions=[],
        state_version=None,
        decision_count=0,
    )
    result = await run_harness.run_scenario(
        out_dir=str(tmp_path / "o"),
        require_adaptive=True,
    )
    assert result.adaptive_required is True
    assert result.adaptive_requirement_met is False
    assert result.ok is False
    assert result.error is not None
    assert "adaptive feature flag DISABLED" in result.error


@pytest.mark.asyncio
async def test_non_strict_legacy_still_passes(stub_live: dict[str, Any], tmp_path: Path) -> None:
    # Same legacy signals, but strict mode OFF → run still passes.
    stub_live.update(
        runtime_state_count=0,
        adaptive_actions=[],
        state_version=None,
        decision_count=0,
    )
    result = await run_harness.run_scenario(
        out_dir=str(tmp_path / "o"),
        require_adaptive=False,
    )
    assert result.adaptive_required is False
    # requirement_met stays True (never evaluated) and does not fail the run.
    assert result.adaptive_requirement_met is True
    assert result.ok is True
    assert result.adaptive_ran is False


@pytest.mark.asyncio
async def test_result_reports_strict_fields(stub_live: dict[str, Any], tmp_path: Path) -> None:
    result = await run_harness.run_scenario(
        out_dir=str(tmp_path / "o"),
        require_adaptive=True,
    )
    import json

    payload = json.loads(result.to_json())
    # The machine-readable result exposes every strict field.
    for key in (
        "adaptive_required",
        "adaptive_requirement_met",
        "fallback_count",
        "state_version",
        "decision_count",
    ):
        assert key in payload
    assert payload["adaptive_required"] is True
    assert payload["state_version"] == 2
    assert payload["decision_count"] == 1


@pytest.mark.asyncio
async def test_security_block_requirement_passes_from_persisted_events(
    stub_live: dict[str, Any], tmp_path: Path
) -> None:
    stub_live.update(
        security_assessment_count=3,
        security_blocked_count=3,
        security_categories=["answer_key_request", "system_prompt_request"],
        security_actions=["refuse_and_redirect", "warn_and_redirect"],
    )
    result = await run_harness.run_scenario(
        out_dir=str(tmp_path / "o"),
        require_security_blocks=3,
    )
    assert result.required_security_blocks == 3
    assert result.security_requirement_met is True
    assert result.security_blocked_count == 3
    assert result.ok is True


@pytest.mark.asyncio
async def test_security_block_requirement_fails_when_stt_turn_was_not_blocked(
    stub_live: dict[str, Any], tmp_path: Path
) -> None:
    stub_live.update(security_assessment_count=3, security_blocked_count=2)
    result = await run_harness.run_scenario(
        out_dir=str(tmp_path / "o"),
        require_security_blocks=3,
    )
    assert result.security_requirement_met is False
    assert result.ok is False
    assert result.error is not None
    assert "blocked=2" in result.error
