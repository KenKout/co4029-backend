"""Conversation-context helpers for the native interview agent.

Two jobs, both about what the agent can SEE:

* :func:`seed_onboarding_history` replays the REST onboarding exchange into the
  agent's context. Onboarding cannot run through the agent (the backend refuses to
  dispatch it until ``onboarding_stage == "completed"``, because the language
  choice made during onboarding is what shapes the dispatch), so without seeding
  the agent starts blind and re-asks for the candidate's name.

* :func:`end_on_user_turn` trims the seeded history so the join request ends with
  the candidate speaking, which this gateway requires.

The state note is NOT here. It belongs in the agent's SYSTEM instructions —
see :meth:`NativeInterviewAgent.refresh_state_note`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import ChatContext


# Roles that may be replayed from a stored transcript. `system` is deliberately
# EXCLUDED: a stored system row would become a live instruction the model obeys,
# which turns the transcript store into an injection vector.
_REPLAYABLE_ROLES = frozenset({"user", "assistant"})


def end_on_user_turn(chat_ctx: ChatContext) -> None:
    """Drop trailing messages until the context ends with the candidate speaking.

    Gemini — and therefore the gateway in front of it — refuses any request whose
    last message is not a user turn: "Requests ending with a model turn are not
    supported" (a 400, and a trailing SYSTEM message counts too). Verified against
    the live gateway.

    Every mid-interview turn satisfies this for free, because the SDK appends the
    candidate's message AFTER ``on_user_turn_completed`` has injected the state
    note. The JOIN is the exception: nothing is arriving, so the context ends with
    the onboarding ceremony's last line plus the note, and the opening generation
    failed with a 400 — the candidate confirmed they were ready and then heard
    nothing at all.

    What gets dropped is exactly what should not be there anyway: the
    ``ready_transition`` line ("here is your first question"), which is withheld
    from the candidate everywhere else and which left the model believing it had
    already announced the question. The note is re-injected on the first real turn.
    """
    while chat_ctx.items:
        last = chat_ctx.items[-1]
        if last.type == "message" and last.role == "user":
            return
        chat_ctx.items.pop()


def seed_onboarding_history(
    chat_ctx: ChatContext, turns: Sequence[tuple[str, str]] | Iterable[tuple[str, str]]
) -> None:
    """Replay the REST onboarding exchange into the agent's context.

    Idempotent by text: a rejoin re-runs setup, and duplicating the greeting would
    have the agent believe it introduced itself twice.

    Blank content is skipped, and only ``user`` / ``assistant`` rows are replayed
    (see ``_REPLAYABLE_ROLES``).
    """
    existing = {
        (item.role, (item.text_content or "").strip())
        for item in chat_ctx.items
        if item.type == "message"
    }
    for role, text in turns:
        content = (text or "").strip()
        if not content or role not in _REPLAYABLE_ROLES:
            continue
        if (role, content) in existing:
            continue
        chat_ctx.add_message(role=role, content=content)
        existing.add((role, content))


__all__ = [
    "end_on_user_turn",
    "seed_onboarding_history",
]
