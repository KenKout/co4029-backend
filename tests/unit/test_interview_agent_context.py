"""Tests for the native interview agent itself (context wiring).

Three behaviours are pinned, all of which were bugs in the routed architecture:

  * **The state note is injected as a system message.** If it were appended as a
    user message the model would read its own budget as something the CANDIDATE
    said, and a candidate could then imitate it — "you may call next_question" is
    a prompt injection if it arrives in the user role.
  * **Onboarding is carried into the conversation.** Onboarding runs over REST
    before the agent is dispatched (the backend refuses dispatch until
    ``onboarding_stage == "completed"``), so without seeding, the agent opens by
    re-asking for the candidate's name and language — which is exactly what the
    live transcript showed.
  * **The reminder does not accumulate.** It is appended every turn; if stale
    copies are left behind, the context fills with contradictory budgets and the
    oldest one is as likely to be obeyed as the newest.
"""

from __future__ import annotations

from uuid import uuid4

from livekit.agents import ChatContext

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.realtime.agent_context import (
    REMINDER_PREFIX,
    inject_state_reminder,
    seed_onboarding_history,
)
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


def _roles(ctx: ChatContext) -> list[str]:
    return [item.role for item in ctx.items if item.type == "message"]


def _texts(ctx: ChatContext) -> list[str]:
    return [(item.text_content or "") for item in ctx.items if item.type == "message"]


# ── reminder injection ────────────────────────────────────────────────────────


def test_reminder_is_injected_as_a_system_message() -> None:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="I think it's about transactions.")
    inject_state_reminder(ctx, _userdata())

    assert "system" in _roles(ctx), "reminder must not be attributed to a participant"
    note = next(t for t in _texts(ctx) if REMINDER_PREFIX in t)
    assert "NOT yet covered" in note


def test_reminder_replaces_the_previous_one() -> None:
    ctx = ChatContext.empty()
    data = _userdata(points=0)
    inject_state_reminder(ctx, data)
    # Same question, now covered — the note must flip, not pile up.
    assert data.state is not None
    data.state.outcome_coverage["o1"].coverage_points = COVERAGE_SUFFICIENT_POINTS
    inject_state_reminder(ctx, data)

    notes = [t for t in _texts(ctx) if REMINDER_PREFIX in t]
    assert len(notes) == 1, f"stale reminders accumulated: {len(notes)}"
    assert "MAY call" in notes[0]


def test_reminder_is_skipped_when_state_is_unavailable() -> None:
    ctx = ChatContext.empty()
    data = _userdata()
    data.state = None
    inject_state_reminder(ctx, data)
    assert not [t for t in _texts(ctx) if REMINDER_PREFIX in t]


def test_reminder_injection_preserves_the_conversation() -> None:
    ctx = ChatContext.empty()
    ctx.add_message(role="assistant", content="What is an index?")
    ctx.add_message(role="user", content="A lookup structure.")
    inject_state_reminder(ctx, _userdata())

    texts = _texts(ctx)
    assert "What is an index?" in texts
    assert "A lookup structure." in texts


# ── onboarding seeding ────────────────────────────────────────────────────────


def test_onboarding_history_is_seeded_in_order() -> None:
    ctx = ChatContext.empty()
    seed_onboarding_history(
        ctx,
        [
            ("assistant", "Can you confirm I'm speaking with Duy?"),
            ("user", "Duy"),
            ("assistant", "Which language would you like?"),
            ("user", "English"),
        ],
    )
    assert _roles(ctx) == ["assistant", "user", "assistant", "user"]
    assert _texts(ctx)[1] == "Duy"


def test_onboarding_seeding_ignores_blank_and_unknown_roles() -> None:
    ctx = ChatContext.empty()
    seed_onboarding_history(
        ctx,
        [("assistant", "   "), ("system", "internal note"), ("user", "Duy")],
    )
    # Blank text carries nothing; a system row from the transcript store must not
    # become an instruction the model obeys.
    assert _roles(ctx) == ["user"]
    assert _texts(ctx) == ["Duy"]


def test_onboarding_seeding_is_idempotent() -> None:
    # A rejoin re-runs setup; seeding twice must not duplicate the greeting.
    ctx = ChatContext.empty()
    turns = [("assistant", "Confirm your name?"), ("user", "Duy")]
    seed_onboarding_history(ctx, turns)
    seed_onboarding_history(ctx, turns)
    assert _texts(ctx).count("Duy") == 1
