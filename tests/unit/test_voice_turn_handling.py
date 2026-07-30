"""Unit tests for voice turn-taking configuration.

These pin two decisions that are easy to undo by accident and impossible to
notice from the code alone.

The first is that endpointing is DYNAMIC with a raised floor. The SDK default is
a fixed 0.5s, tuned for assistants answering one-sentence requests. An interview
answer is not that: candidates stop mid-thought, and at 0.5s the interviewer
talks over them. Published studies of AI-run oral assessment name pacing as the
dominant complaint, and the ASR-fairness literature identifies the same
endpointing threshold as what cuts off disfluent and stuttered speech — so this
setting is simultaneously a realism and an accessibility property.

The second is a negative: interruption mode is deliberately NOT set. Auto-detect
selects the ML detector on the English path, where Deepgram reports word-aligned
transcripts. Naming ``adaptive`` explicitly would not improve English and would
make the Vietnamese session (whose STT reports no alignment) log a warning and
disable the feature outright.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.realtime import session_runtime as sr


def _session(language: str) -> Any:
    """Build a real AgentSession with only the VAD model stubbed out."""
    with patch.object(sr.silero.VAD, "load", return_value=MagicMock()):
        return sr.build_agent_session(
            get_settings(),
            language=language,
            voice=None,
            persona="neutral",
            verbosity=2,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "vi"])
async def test_endpointing_is_dynamic_on_both_language_paths(language: str) -> None:
    """Asserts the resolved session, not the dict we passed in.

    Building the options object proves nothing: the value has to survive
    AgentSession's own merge with its defaults to actually take effect.
    """
    endpointing = _session(language)._opts.endpointing

    assert endpointing["mode"] == "dynamic"
    assert endpointing["min_delay"] == sr._ENDPOINTING_MIN_DELAY_S
    assert endpointing["max_delay"] == sr._ENDPOINTING_MAX_DELAY_S


def test_endpointing_floor_is_above_the_sdk_default() -> None:
    """The whole point is more room to think than an assistant would give.

    If someone lowers this back to 0.5 the change is silent — nothing errors,
    interviews just start clipping people mid-sentence again.
    """
    assert sr._ENDPOINTING_MIN_DELAY_S > 0.5
    assert sr._ENDPOINTING_MAX_DELAY_S > sr._ENDPOINTING_MIN_DELAY_S
    # Sanity ceiling: a long tail is fine, dead air is not.
    assert sr._ENDPOINTING_MAX_DELAY_S <= 6.0


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "vi"])
async def test_interruption_mode_is_left_to_auto_detect(language: str) -> None:
    """Deliberate omission — see the module docstring.

    Setting ``mode`` explicitly breaks the Vietnamese path, so this test exists
    to stop a future reader "completing" the config.
    """
    interruption = _session(language)._opts.interruption

    assert "mode" not in interruption


@pytest.mark.asyncio
async def test_false_interruption_resume_stays_on() -> None:
    """A cough must not truncate the question a candidate is being asked.

    This is an SDK default rather than something we set, which is exactly why
    it is worth pinning: a future config change could switch it off without
    anyone noticing until a nervous candidate loses half a question.
    """
    interruption = _session("en")._opts.interruption

    assert interruption["resume_false_interruption"] is True
    assert interruption["false_interruption_timeout"] is not None


def test_turn_handling_sets_only_endpointing() -> None:
    """Everything else is intentionally the SDK's choice, not ours."""
    assert set(sr._turn_handling()) == {"endpointing"}
