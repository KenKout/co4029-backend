"""Conversation-context helpers for the native interview agent.

Two jobs, both about what the agent can SEE:

* :func:`inject_state_reminder` pushes the server's view of progress into the
  conversation after each candidate answer, so the common turn (probe / follow-up)
  needs no tool round-trip. It is injected as a SYSTEM message and the previous
  copy is removed first — see the function docs for why both matter.

* :func:`seed_onboarding_history` replays the REST onboarding exchange into the
  agent's context. Onboarding cannot run through the agent (the backend refuses to
  dispatch it until ``onboarding_stage == "completed"``, because the language
  choice made during onboarding is what shapes the dispatch), so without seeding
  the agent starts blind and re-asks for the candidate's name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from abridgeai.features.interviews.realtime.agent_session import build_state_reminder

if TYPE_CHECKING:
    from livekit.agents import ChatContext

    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

# Marker that identifies a reminder so the previous one can be found and dropped.
# Part of the injected text (not metadata) because it must survive any context
# copy the SDK performs — `chat_ctx.copy()` preserves content, not side channels.
REMINDER_PREFIX = "[interview state]"

# Roles that may be replayed from a stored transcript. `system` is deliberately
# EXCLUDED: a stored system row would become a live instruction the model obeys,
# which turns the transcript store into an injection vector.
_REPLAYABLE_ROLES = frozenset({"user", "assistant"})


def inject_state_reminder(chat_ctx: ChatContext, data: InterviewUserdata) -> None:
    """Append the current progress note, replacing any earlier one.

    SYSTEM role, not user: as a user message the model would read its own budget
    as something the candidate said — and a candidate could then type the same
    words to grant themselves an advance.

    The previous note is removed first because these are appended every turn. Left
    to accumulate, the context ends up holding several contradictory budgets and
    there is no reason the model should prefer the newest.

    A missing runtime state produces no note at all rather than a guess: the agent
    acting on a fabricated budget is worse than it having to call
    ``interview_get_progress``.
    """
    note = build_state_reminder(data)
    if not note:
        return

    chat_ctx.items = [
        item
        for item in chat_ctx.items
        if not (
            item.type == "message"
            and item.role == "system"
            and REMINDER_PREFIX in (item.text_content or "")
        )
    ]
    chat_ctx.add_message(role="system", content=f"{REMINDER_PREFIX} {note}")


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


__all__ = ["REMINDER_PREFIX", "inject_state_reminder", "seed_onboarding_history"]
