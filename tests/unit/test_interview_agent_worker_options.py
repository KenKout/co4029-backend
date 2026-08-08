"""The agent worker's `WorkerOptions` wiring.

Covers only what `run()` hands to livekit-agents. The entrypoint itself needs a
live `JobContext` and is exercised by the voice harness, not here.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("livekit.agents", reason="requires the interview-agent extra")

from abridgeai.features.interviews.realtime import agent as agent_module  # noqa: E402


@pytest.fixture
def captured_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run `run()` with `cli.run_app` replaced, and return the WorkerOptions."""
    captured: dict[str, Any] = {}

    def fake_run_app(options: Any) -> None:
        captured["options"] = options

    monkeypatch.setattr(agent_module.cli, "run_app", fake_run_app)
    agent_module.run()
    return captured


def test_load_threshold_reaches_worker_options(captured_options: dict[str, Any]) -> None:
    """The configured threshold must actually be handed to the SDK.

    Regression guard with a specific failure in mind: the setting existing while
    `WorkerOptions` silently keeps the SDK default of 0.7 looks identical from
    the outside -- the worker still marks itself unavailable under neighbouring
    CPU load and LiveKit still withholds interviews from it.
    """
    options = captured_options["options"]
    settings = agent_module.get_settings()

    assert options.load_threshold == settings.interview_voice_load_threshold
    assert options.load_threshold > 0.7


def test_agent_name_reaches_worker_options(captured_options: dict[str, Any]) -> None:
    """Worker registration name must match the name the token dispatches to.

    `services/real_time.py` puts `settings.livekit_agent_name` in the token's
    `RoomAgentDispatch`; if registration used anything else, dispatch would
    address an agent nobody is listening for.
    """
    options = captured_options["options"]
    settings = agent_module.get_settings()

    assert options.agent_name == settings.livekit_agent_name
