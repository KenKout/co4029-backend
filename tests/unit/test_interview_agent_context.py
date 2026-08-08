"""Context wiring for the native interview agent.

  * **Onboarding is carried into the conversation.** Onboarding runs over REST
    before the agent is dispatched (the backend refuses dispatch until
    ``onboarding_stage == "completed"``), so without seeding, the agent opens by
    re-asking for the candidate's name and language — which is exactly what the
    live transcript showed.
  * **The join context ends on the candidate's turn.** This gateway rejects any
    request whose last message is not a user turn, so the generated opening 400d
    and the candidate heard nothing after confirming they were ready.

The state note is deliberately NOT tested here: it is no longer part of the
conversation. It lives in the agent's SYSTEM instructions, because Gemini
discards a mid-conversation system message — see
``test_interview_native_agent.test_on_user_turn_completed_folds_the_note_into_the_instructions``.
"""

from __future__ import annotations

from uuid import uuid4

from livekit.agents import ChatContext

from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.realtime.agent_context import (
    end_on_user_turn,
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


def test_join_context_ends_on_the_candidate_s_turn() -> None:
    """The reported dead air: ready confirmed, then nothing at all.

    The opening is GENERATED, and this gateway refuses a request whose last
    message is not a user turn ("Requests ending with a model turn are not
    supported"). The seeded history ended with the interviewer's own ceremony
    line, so the generation 400d and the interview opened in silence.
    """
    ctx = ChatContext.empty()
    seed_onboarding_history(
        ctx,
        [
            ("assistant", "Are you ready to begin?"),
            ("user", "I'm ready to begin."),
            ("assistant", "Great—the introduction is complete. Here is your first question."),
        ],
    )

    end_on_user_turn(ctx)

    assert _texts(ctx) == ["Are you ready to begin?", "I'm ready to begin."]


def test_end_on_user_turn_leaves_a_conforming_context_alone() -> None:
    ctx = ChatContext.empty()
    seed_onboarding_history(ctx, [("assistant", "Ready?"), ("user", "Yes.")])

    end_on_user_turn(ctx)

    assert _texts(ctx) == ["Ready?", "Yes."]


def test_end_on_user_turn_tolerates_a_context_with_no_candidate_turn() -> None:
    ctx = ChatContext.empty()
    seed_onboarding_history(ctx, [("assistant", "Ready?")])

    end_on_user_turn(ctx)

    assert _texts(ctx) == []
