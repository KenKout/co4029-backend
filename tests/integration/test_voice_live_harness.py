"""Live LiveKit voice-interview smoke test (opt-in, costs money).

This is a THIN pytest wrapper around the canonical harness orchestrator in
``scripts.voice_harness.run_harness`` — it does NOT duplicate any harness logic.
It exists so the live smoke test is discoverable via pytest and can gate a
release, while staying completely out of the normal suite and CI.

Excluded by default two ways:
  1. ``pyproject.toml`` addopts carries ``-m 'not destructive and not voice_live'``
     so a bare ``pytest`` never collects it.
  2. A ``skipif`` requires ``RUN_LIVEKIT_VOICE_TESTS=1`` AND real credentials, so
     even ``pytest -m voice_live`` self-skips unless explicitly armed.

Run it (from backend/):
    RUN_LIVEKIT_VOICE_TESTS=1 .venv/bin/pytest \
      tests/integration/test_voice_live_harness.py -m voice_live -s -p no:cov

External services + cost: see ``docs/voice-adaptive-smoke-tests.md`` §4.1.

NOTE: exercising the ADAPTIVE path additionally requires the pm2
``abridgeai-interview-agent`` process to run with
``ADAPTIVE_INTERVIEWER_ENABLED=true``. This wrapper asserts the pipeline runs
end-to-end and captures audio; it reports ``adaptive_ran`` (from the
runtime-state row count) as an informational signal rather than hard-failing,
because the agent-process flag is outside the test process's control.
"""

from __future__ import annotations

import os

import pytest

from scripts.voice_harness.run_harness import run_scenario


def _voice_live_armed() -> bool:
    """Both the opt-in switch AND all required credentials must be present."""
    if os.getenv("RUN_LIVEKIT_VOICE_TESTS") != "1":
        return False
    required = (
        "LIVEKIT_WS_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    )
    return all(os.getenv(k) for k in required)


pytestmark = [
    pytest.mark.voice_live,
    pytest.mark.skipif(
        not _voice_live_armed(),
        reason=(
            "voice_live is opt-in: set RUN_LIVEKIT_VOICE_TESTS=1 and provide "
            "LIVEKIT_* + LLM_BASE_URL + LLM_API_KEY to run the live harness."
        ),
    ),
]


@pytest.mark.parametrize(
    ("language", "answers"),
    [
        pytest.param(
            "en",
            [
                "A fact table stores quantitative measurements for analysis",
                "It connects to dimension tables through foreign keys",
            ],
            id="english",
        ),
        pytest.param(
            "vi",
            [
                "Bảng dữ kiện lưu trữ các số đo định lượng để phân tích",
                "Nó liên kết với các bảng chiều thông qua khóa ngoại",
            ],
            id="vietnamese",
        ),
    ],
)
@pytest.mark.asyncio
async def test_voice_live_scenario(language: str, answers: list[str], tmp_path) -> None:
    """Drive one live voice scenario per language through the canonical harness.

    Asserts the pipeline ran end-to-end (agent spoke, answers transcribed and
    persisted) and that cleanup completed. Audio is kept under ``tmp_path`` on
    failure for post-mortem; deleted on success.
    """
    # Strict adaptive verification is opt-in via REQUIRE_ADAPTIVE_VOICE=1. When
    # armed, the scenario itself fails unless the adaptive brain genuinely ran;
    # otherwise adaptive is informational (a warning is printed).
    require_adaptive = os.getenv("REQUIRE_ADAPTIVE_VOICE") == "1"

    out_dir = tmp_path / f"voice-harness-out-{language}"
    result = await run_scenario(
        language=language,
        answers=answers,
        out_dir=str(out_dir),
        # Provision throwaway data and clean everything up afterwards.
        cleanup_db=True,
        cleanup_livekit=True,
        delete_audio_on_success=True,
        # Keep the whole scenario bounded so a stuck agent can't hang CI-by-hand.
        scenario_timeout_s=float(os.getenv("VOICE_LIVE_TIMEOUT_S", "240")),
        require_adaptive=require_adaptive,
    )

    # Emit the machine-readable result so `-s` runs surface it for triage.
    print("\n" + result.to_json())

    # ── Hard assertions: the end-to-end pipeline actually worked ─────────────
    assert result.error is None, f"scenario errored: {result.error}"
    assert result.agent_utterance_count >= 1, "agent produced no audio"
    assert result.captured_audio_seconds > 0, "captured audio was empty/silent"
    assert result.peak_amplitude > 200, (
        f"captured audio looks like silence (peak={result.peak_amplitude})"
    )
    assert result.message_count >= 1, "no student/agent turns persisted"
    assert result.language == language

    # ── Cleanup must have completed (no leaked DB/LiveKit resources) ─────────
    assert result.db_cleanup_status in ("cleaned", "n/a (used existing config)"), (
        f"db cleanup did not complete: {result.db_cleanup_status}"
    )
    assert result.livekit_cleanup_status in (
        "deleted",
    ) or result.livekit_cleanup_status.startswith("failed"), result.livekit_cleanup_status

    # ── Adaptive verification ────────────────────────────────────────────────
    if require_adaptive:
        # Strict mode: the adaptive brain MUST have genuinely run. run_scenario
        # already set result.ok=False + a diagnostic error, but assert here too
        # so the failure is attributed to this specific expectation.
        assert result.adaptive_requirement_met, (
            f"strict adaptive verification failed: {result.error}"
        )
        assert result.adaptive_ran
        assert result.runtime_state_count >= 1
        assert result.decision_count >= 1
        assert result.adaptive_actions
        assert result.state_version is not None
    elif not result.adaptive_ran:
        # Informational: the agent-process flag is outside this process's
        # control, so a legacy run is a loud warning rather than a failure.
        print(
            "WARNING: adaptive brain did NOT run (runtime_state_count=0). "
            "Ensure the abridgeai-interview-agent process has "
            "ADAPTIVE_INTERVIEWER_ENABLED=true AND "
            "ADAPTIVE_INTERVIEWER_VOICE_ENABLED=true to exercise the adaptive path."
        )
