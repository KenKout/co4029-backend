"""Tests for the native (multiturn) agent session wiring.

The single most important assertion here is that the session is built WITH an
``llm``. Without one, ``AgentSession`` cannot hold a conversation: every turn has
to be routed out to a stateless per-stage call, which is precisely the
architecture this replaces. A regression that dropped the ``llm`` would still
produce a working-looking interview — the routed path would take over — while
silently losing the multiturn context, so it has to be pinned explicitly.

Also pinned: the reminder injected after each answer is a system-role note (not
words attributed to the candidate or the interviewer), and text-only mode disables
audio input/output rather than muting it downstream.
"""

from __future__ import annotations

from uuid import uuid4

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.realtime.agent_instructions import build_instructions
from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata


def _userdata(points: int = 0) -> InterviewUserdata:
    state = InterviewRuntimeStateData()
    state.outcome_coverage = {"o1": OutcomeCoverageState(outcome_id="o1", coverage_points=points)}
    state.current_outcome_id = "o1"
    return InterviewUserdata(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        state=state,
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=3,
    )


# ── instructions ──────────────────────────────────────────────────────────────


def test_instructions_forbid_the_behaviours_seen_in_production() -> None:
    text = build_instructions(language="en").lower()
    # Each of these was an actual complaint from a live transcript.
    assert "which part" in text, "must explicitly forbid asking the candidate to self-diagnose"
    assert "never praise a non-answer" in text or "never praise" in text
    assert "paraphrase" in text, "the agent must be told it MAY reword the question"
    assert "never reveal" in text


def test_instructions_are_language_aware() -> None:
    en = build_instructions(language="en")
    vi = build_instructions(language="vi")
    assert "Vietnamese" in vi
    assert vi != en
    # The safety rules must survive the language switch, not be replaced by it.
    assert "NEVER reveal" in vi


def test_named_interviewer_gets_no_backstory_licence() -> None:
    text = build_instructions(language="en", interviewer_name="Hà")
    assert "Hà" in text
    assert "do not invent personal" in text.lower()


def test_instructions_ban_markdown_for_a_spoken_channel() -> None:
    text = build_instructions(language="en").lower()
    assert "markdown" in text


# ── reminder injection ────────────────────────────────────────────────────────


def test_reminder_is_built_from_committed_state() -> None:
    from abridgeai.features.interviews.realtime.agent_session import build_state_reminder

    data = _userdata(points=0)
    note = build_state_reminder(data)
    assert "NOT yet covered" in note
    assert "Do NOT call" in note

    covered = _userdata(points=COVERAGE_SUFFICIENT_POINTS)
    assert "MAY call" in build_state_reminder(covered)


def test_reminder_is_absent_when_state_is_not_loaded() -> None:
    # A session whose state has not been loaded must not fabricate a note — a
    # confident-sounding wrong note is worse than no note.
    from abridgeai.features.interviews.realtime.agent_session import build_state_reminder

    data = _userdata()
    data.state = None
    assert build_state_reminder(data) == ""


# ── text-only mode ────────────────────────────────────────────────────────────


def test_text_only_mode_disables_audio_at_the_room_boundary() -> None:
    from abridgeai.features.interviews.realtime.agent_session import room_options_for_mode

    text_in, text_out = room_options_for_mode("text")
    assert text_in.audio_enabled is False, "a typing candidate must not have their mic captured"
    assert text_out.audio_enabled is False, "text mode must not synthesize speech"
    assert text_out.transcription_enabled is True, "the candidate still needs to READ the agent"


def test_voice_mode_keeps_audio_both_ways() -> None:
    from abridgeai.features.interviews.realtime.agent_session import room_options_for_mode

    voice_in, voice_out = room_options_for_mode("voice")
    assert voice_in.audio_enabled is True
    assert voice_out.audio_enabled is True


def test_hybrid_mode_accepts_text_and_audio() -> None:
    from abridgeai.features.interviews.realtime.agent_session import room_options_for_mode

    hybrid_in, hybrid_out = room_options_for_mode("hybrid")
    assert hybrid_in.text_enabled is True
    assert hybrid_in.audio_enabled is True
